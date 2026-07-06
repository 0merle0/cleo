"""On-the-fly composing dataset: target x scaffold x CDR-lengths per rollout (SPEC 6.9).

The dataset-driven harness (``DesignDataset``) serves rows from a materialized JSONL
manifest. For antibody design that manifest would be a full **cross-product** of every
antigen target against every framework scaffold against every CDR-length draw — far too
large to materialize and static across a run. ``ComposingDataset`` instead holds two
small pools and **composes one example per** ``sample()`` **call**, drawing the three
axes independently:

  - **target**  — an antigen from ``data/pinder_epitopes_trainable.csv`` (chain ``T``
    antigen PDB + epitope residues + antigen MSA + net charge), filtered to one split.
  - **scaffold**— a framework rep from ``data/scaffold_pool.csv`` (VHH = 1 design chain
    ``H``; Fv = 2 chains ``H``/``L``), each carrying its per-chain framework MSA + native
    CDR spans. The VHH/Fv mix is a tunable knob (``vhh_fraction``; user default 0.5).
  - **CDR lengths** — sampled fresh per call (length diversity; SPEC 6.8) and pinned into
    the row as a ``cdr_lengths`` override so the featurizer's gapping and the reward's
    per-chain sequence split use the *same* draw.

The composed ``Example`` is byte-compatible with everything downstream: the featurizer
reads ``params.cdr_spans`` (native, for gapping) + ``params.cdr_lengths`` (the pinned
draw); the reward reads a uniform ``params.design_chains`` = ``[{length, framework_msa_dir,
cdr_spans}, ...]`` (one entry for VHH, two for Fv) that the generalized Protenix oracle
splits the decoded sequence by and gaps each chain's framework MSA with. Because the
lengths are pinned at ``sample()`` time, the post-gap per-chain ``length`` here is exactly
the decoded segment length the oracle's ``_split_seq`` expects.
"""
from __future__ import annotations

import csv
import json
import os
import random

from cleo.design.data.dataset import DesignDataset, Example
from cleo.design.data.mask import _cdr_key, as_chain_list, sample_cdr_lengths

#: chain letter -> which scaffold-pool MSA column holds that chain's framework MSA dir.
_CHAIN_MSA_COL = {"H": "msa_dir_H", "L": "msa_dir_L"}


class ComposerError(ValueError):
    """Raised when a pool row is malformed or a composed example is inconsistent."""


def _parse_epitope_residues(raw: str) -> list[int]:
    """Manifest epitope residues are a space-separated string of chain-T seqid numbers."""
    return [int(x) for x in str(raw).split()] if raw else []


def _load_targets(csv_path: str, split: str | None) -> list[dict]:
    """Antigen pool: one row per (target, antigen-chain), optionally filtered to a split."""
    rows: list[dict] = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if split is not None and r.get("split") != split:
                continue
            epi = _parse_epitope_residues(r.get("epitope_residues", ""))
            if not epi:
                continue  # nothing to score against; skip defensively
            nc = r.get("epitope_net_charge", "")
            rows.append({
                "target_id": r["id"],
                "antigen_file": r["antigen_file"],
                "antigen_msa_dir": r.get("msa_dir") or None,
                "epitope_residues": epi,
                "epitope_net_charge": int(nc) if nc not in ("", None) else None,
            })
    if not rows:
        raise ComposerError(f"target pool {csv_path} yielded no rows (split={split!r})")
    return rows


def _load_scaffolds(csv_path: str) -> tuple[list[dict], list[dict], int]:
    """Framework pool split by modality. Drops reps missing a required per-chain MSA dir."""
    vhh, fv, dropped = [], [], 0
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            design_chain = r["design_chain"]
            chains = as_chain_list(design_chain)
            msa_dirs = {c: (r.get(_CHAIN_MSA_COL[c]) or None) for c in chains}
            if any(msa_dirs[c] is None for c in chains):
                dropped += 1
                continue
            entry = {
                "scaffold_id": r["scaffold_id"],
                "kind": r["kind"],
                "structure_file": r["structure_file"],
                "design_chain": design_chain,
                "cdr_spans": json.loads(r["cdr_spans"]),
                "msa_dirs": msa_dirs,
            }
            (vhh if r["kind"] == "vhh" else fv).append(entry)
    if not (vhh or fv):
        raise ComposerError(f"scaffold pool {csv_path} yielded no usable reps")
    return vhh, fv, dropped


class ComposingDataset(DesignDataset):
    """Compose (target x scaffold x CDR-lengths) into an Example per ``sample()`` call."""

    def __init__(
        self,
        targets: list[dict],
        scaffolds_vhh: list[dict],
        scaffolds_fv: list[dict],
        reward: str,
        reward_dir: str,
        *,
        vhh_fraction: float = 0.5,
        cdr_length_ranges: dict | None = None,
        native=None,
        rng: random.Random | None = None,
    ):
        super().__init__(rows=[], reward_dir=reward_dir, native=native)
        if not (0.0 <= vhh_fraction <= 1.0):
            raise ComposerError(f"vhh_fraction must be in [0, 1], got {vhh_fraction}")
        self.targets = targets
        self.scaffolds = {"vhh": scaffolds_vhh, "fv": scaffolds_fv}
        self.reward = reward
        self.vhh_fraction = vhh_fraction
        self.cdr_length_ranges = cdr_length_ranges or {}
        self.rng = rng or random.Random()

    # -- loading ------------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        targets_csv: str,
        scaffold_pool_csv: str,
        reward: str,
        reward_dir: str,
        *,
        split: str | None = None,
        vhh_fraction: float = 0.5,
        cdr_length_ranges: dict | None = None,
        rng: random.Random | None = None,
    ) -> "ComposingDataset":
        """Read both pools, validate the reward package exists, return a composer."""
        targets = _load_targets(targets_csv, split)
        vhh, fv, dropped = _load_scaffolds(scaffold_pool_csv)
        ds = cls(
            targets, vhh, fv, reward, reward_dir,
            vhh_fraction=vhh_fraction, cdr_length_ranges=cdr_length_ranges, rng=rng,
        )
        ds._package(reward)  # fail fast if the reward package is missing / unreadable
        print(
            f"ComposingDataset: {len(targets)} targets (split={split!r}), "
            f"{len(vhh)} VHH + {len(fv)} Fv scaffolds"
            + (f" ({dropped} dropped: missing framework MSA)" if dropped else "")
            + f", vhh_fraction={vhh_fraction}"
        )
        return ds

    # -- composition --------------------------------------------------------- #
    def _pick_modality(self) -> str:
        """VHH with prob ``vhh_fraction`` (falls back to the non-empty pool)."""
        want_vhh = self.rng.random() < self.vhh_fraction
        if want_vhh and self.scaffolds["vhh"]:
            return "vhh"
        if not want_vhh and self.scaffolds["fv"]:
            return "fv"
        return "vhh" if self.scaffolds["vhh"] else "fv"

    def _design_chain_records(self, scaffold: dict, cdr_lengths: dict) -> list[dict]:
        """Per design chain (H-then-L order): post-gap length, framework MSA dir, native
        CDR spans. ``length`` = native VD length - sum(native span widths) + sum(sampled
        lengths) for that chain's CDRs = the decoded segment length the oracle splits by."""
        spans = scaffold["cdr_spans"]
        records = []
        for chain in as_chain_list(scaffold["design_chain"]):
            chain_spans = {k: v for k, v in spans.items() if _cdr_key(k).startswith(chain)}
            native_len = len(self.native.seq(scaffold["structure_file"], chain))
            delta = sum(cdr_lengths[k] - (int(v[1]) - int(v[0])) for k, v in chain_spans.items())
            length = native_len + delta
            if length <= 0:
                raise ComposerError(
                    f"scaffold {scaffold['scaffold_id']} chain {chain!r} post-gap length "
                    f"{length} <= 0 (native {native_len}, delta {delta})"
                )
            records.append({
                "length": length,
                "framework_msa_dir": scaffold["msa_dirs"][chain],
                "cdr_spans": chain_spans,
            })
        return records

    def _compose_row(self) -> dict:
        target = self.rng.choice(self.targets)
        scaffold = self.rng.choice(self.scaffolds[self._pick_modality()])

        # One CDR-length draw, shared by the featurizer (gapping) and the reward (split).
        cdr_lengths = sample_cdr_lengths(
            {"cdr_spans": scaffold["cdr_spans"], "cdr_length_ranges": self.cdr_length_ranges},
            rng=self.rng,
        )
        design_chains = self._design_chain_records(scaffold, cdr_lengths)

        return {
            "id": f"{scaffold['scaffold_id']}__{target['target_id']}",
            "task": "antibody_design",
            "reward": self.reward,
            "structure": scaffold["structure_file"],
            "design_chain": scaffold["design_chain"],
            "antigen_structure": target["antigen_file"],
            "params": {
                "cdr_spans": scaffold["cdr_spans"],       # native positional spans (featurizer gap)
                "cdr_lengths": cdr_lengths,               # pinned draw -> featurizer reuses it
                "design_chains": design_chains,           # reward: split + per-chain fw-MSA gap
                "antigen_msa_dir": target["antigen_msa_dir"],
                "epitope_residues": target["epitope_residues"],
                "epitope_net_charge": target["epitope_net_charge"],
                # provenance (logged per rollout; not fed to the model)
                "scaffold_id": scaffold["scaffold_id"],
                "kind": scaffold["kind"],
                "target_id": target["target_id"],
            },
        }

    def sample(self, k=None):
        n = 1 if k is None else k
        exs = [self._to_example(self._compose_row()) for _ in range(n)]
        return exs[0] if k is None else exs

    def __len__(self):
        """Nominal size = pool cross-product (never materialized; for logging/introspection)."""
        return len(self.targets) * (len(self.scaffolds["vhh"]) + len(self.scaffolds["fv"]))
