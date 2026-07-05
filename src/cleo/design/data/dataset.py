"""Dataset-driven training harness data layer (SPEC 6).

``DesignDataset`` loads a JSONL manifest into a list of rows, validates **every**
row against its reward package's declared contract (``requires`` + ``steps.*.inputs``)
at load, and serves ``Example`` rows to the training loop. Validation is a **hard
error** at load (SPEC 6.3): a dataset that loads is guaranteed that every example
fully satisfies its reward. The single-PDB CLEO flow is the degenerate 1-row case.

Validation is deliberately **cheap** (SPEC 6.3): the structure file must exist and
the referenced chains must be **present** (a light chain-ID scan of ATOM records — no
full atom parse), and every binding must resolve. Actual per-chain **sequences** are
parsed lazily (gemmi) and cached only when an example is first sampled/bound.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import gemmi
from omegaconf import OmegaConf

from cleo.design.data.binding import (
    BindingError,
    native_seq_fn,
    resolve_inputs,
)
from cleo.design.data.mask import as_chain_list, resolve_fixed_residues


class DesignDatasetError(ValueError):
    """Raised when a dataset row fails structural / schema validation at load."""


_REQUIRED_FIELDS = ("id", "task", "reward", "structure", "design_chain")


# --------------------------------------------------------------------------- #
# structure I/O
# --------------------------------------------------------------------------- #
def scan_chain_ids(path: str) -> set[str]:
    """Light chain-ID scan of a PDB (column 22 of ATOM/HETATM records). No atom parse."""
    chains: set[str] = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                cid = line[21].strip()
                if cid:
                    chains.add(cid)
    return chains


class NativeSeqProvider:
    """Lazy per-chain one-letter sequence parsing from a structure, cached per path."""

    def __init__(self):
        self._cache: dict[str, dict[str, str]] = {}

    def _load(self, path: str) -> dict[str, str]:
        if path not in self._cache:
            st = gemmi.read_structure(path)
            chains: dict[str, str] = {}
            for ch in st[0]:
                names = []
                for r in ch:
                    info = gemmi.find_tabulated_residue(r.name)
                    if info is not None and info.is_amino_acid():
                        names.append(r.name)
                chains[ch.name] = gemmi.one_letter_code(names)
            self._cache[path] = chains
        return self._cache[path]

    def seq(self, path: str, chain: str) -> str:
        chains = self._load(path)
        if chain not in chains:
            raise BindingError(
                f"structure {path} has no chain {chain!r} (present: {sorted(chains)})"
            )
        return chains[chain]

    def chains(self, path: str) -> set[str]:
        return set(self._load(path).keys())


# --------------------------------------------------------------------------- #
# example
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    """One dataset row — the *what* (SPEC 6). Target-specific; reward-agnostic."""

    id: str
    task: str
    reward: str
    structure: str                  # framework scaffold PDB (design chain(s); no antigen)
    design_chain: Any               # str (VHH "A") or list (Fab ["A", "B"])
    design_regions: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    antigen_structure: str = ""     # SEPARATE antigen PDB (chain T); no shared frame with framework
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def cdr_spans(self) -> dict:
        return self.params.get("cdr_spans", {})

    @property
    def epitope_source(self) -> str:
        """PDB the epitope lives in — the separate antigen file when given, else ``structure``.

        Framework and antigen are always independent files (no dock / no shared coordinate
        frame; SPEC 4.5): the framework carries only the design chain(s), the antigen only
        chain T. Falling back to ``structure`` supports single-file fixtures/tests.
        """
        return self.antigen_structure or self.structure

    @property
    def epitope_residues(self) -> list:
        """PDB residue numbers (chain T seqid.num) of the intended epitope — NOT positional
        indices (unlike cdr_spans). Matches the convention read by the reward oracle."""
        return self.params.get("epitope_residues", [])


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class DesignDataset:
    def __init__(self, rows, reward_dir, structures_root="", native=None):
        self.rows = rows                      # list[dict]
        self.reward_dir = reward_dir
        self.structures_root = structures_root
        self.native = native or NativeSeqProvider()
        self._packages: dict[str, Any] = {}   # name -> OmegaConf

    # -- loading / validation ------------------------------------------------ #
    @classmethod
    def load(cls, jsonl_path, reward_dir, structures_root=""):
        """Read the JSONL manifest, then hard-validate every row's reward contract."""
        rows = _read_jsonl(jsonl_path)
        if not rows:
            raise DesignDatasetError(f"dataset {jsonl_path} is empty")
        ds = cls(rows, reward_dir, structures_root)
        for i, row in enumerate(rows):
            ds._validate_row(i, row)
        return ds

    def package(self, name: str):
        """Public accessor for a loaded reward package (plain dict, resolve=False)."""
        return self._package(name)

    def _package(self, name: str):
        if name not in self._packages:
            path = os.path.join(self.reward_dir, f"{name}.yaml")
            if not os.path.isfile(path):
                raise DesignDatasetError(f"reward package {name!r} not found at {path}")
            # resolve=False keeps our ${native.seq.T} templates as literal strings rather
            # than letting OmegaConf try to resolve them as its own interpolations.
            self._packages[name] = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        return self._packages[name]

    def _structure_path(self, row: dict) -> str:
        s = row["structure"]
        return s if os.path.isabs(s) or not self.structures_root else os.path.join(self.structures_root, s)

    def _antigen_path(self, row: dict) -> str:
        """Resolve the separate antigen structure path, or "" when the row has none."""
        s = row.get("antigen_structure")
        if not s:
            return ""
        return s if os.path.isabs(s) or not self.structures_root else os.path.join(self.structures_root, s)

    def _validate_row(self, i: int, row: dict):
        for f in _REQUIRED_FIELDS:
            if f not in row:
                raise DesignDatasetError(f"row {i} (id={row.get('id')!r}) missing required field {f!r}")

        path = self._structure_path(row)
        if not os.path.isfile(path):
            raise DesignDatasetError(f"[{row['id']}] structure not found: {path}")

        present = scan_chain_ids(path)
        design_chains = as_chain_list(row["design_chain"])
        for c in design_chains:
            if c not in present:
                raise DesignDatasetError(
                    f"[{row['id']}] design_chain {c!r} absent from structure (present: {sorted(present)})"
                )

        # Antigen lives in a SEPARATE file (no shared frame); validate it holds chain T when given.
        antigen_path = self._antigen_path(row)
        if antigen_path:
            if not os.path.isfile(antigen_path):
                raise DesignDatasetError(f"[{row['id']}] antigen_structure not found: {antigen_path}")
            antigen_present = scan_chain_ids(antigen_path)
            if "T" not in antigen_present:
                raise DesignDatasetError(
                    f"[{row['id']}] antigen chain 'T' absent from antigen_structure "
                    f"(present: {sorted(antigen_present)})"
                )

        # Resolve the reward contract with a LIGHT native resolver: chain-presence only,
        # no sequence parse. Requires + templates must all be satisfiable.
        def light_resolver(chain: str) -> str:
            if chain not in present:
                raise BindingError(
                    f"[{row['id']}] ${{native.seq.{chain}}} references chain absent from "
                    f"structure (present: {sorted(present)})"
                )
            return ""  # non-None: contract is satisfiable; real seq parsed lazily at bind time

        package = self._package(row["reward"])
        resolve_inputs(package, row, native_seq_fn(row, light_resolver))

    # -- runtime ------------------------------------------------------------- #
    def _to_example(self, row: dict) -> Example:
        return Example(
            id=row["id"],
            task=row["task"],
            reward=row["reward"],
            structure=self._structure_path(row),
            design_chain=row["design_chain"],
            design_regions=row.get("design_regions", []),
            params=row.get("params", {}),
            antigen_structure=self._antigen_path(row),
            raw=row,
        )

    def sample(self, k=None):
        """Uniform-random draw. ``sample()`` -> one Example; ``sample(K)`` -> list of K."""
        n = 1 if k is None else k
        picks = random.choices(self.rows, k=n)
        exs = [self._to_example(r) for r in picks]
        return exs[0] if k is None else exs

    def __len__(self):
        return len(self.rows)

    def bind_inputs(self, example: Example) -> dict[str, object]:
        """Resolve an example's reward inputs to concrete values (real native sequences).

        Returns ``{column: value}`` for seeding into ``get_input_df`` (SPEC 6.4).
        ``DESIGN_SEQ``-valued columns are left for the caller to alias to ``sequence``.
        """
        package = self._package(example.reward)
        resolver = lambda chain: self.native.seq(example.structure, chain)  # noqa: E731
        return resolve_inputs(package, example.raw, native_seq_fn(example.raw, resolver))

    def fixed_residues(self, example: Example, chain_letters, R_idx, icodes) -> str:
        """``fixed_residues`` token string for an example, given a parsed enumeration."""
        return resolve_fixed_residues(
            chain_letters, R_idx, icodes, example.design_chain, example.cdr_spans
        )


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DesignDatasetError(f"{path}:{ln + 1}: invalid JSON ({e})") from e
    return rows
