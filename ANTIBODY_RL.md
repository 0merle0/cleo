# Epitope-conditioned antibody design (branch `ab-roulette`)

RL fine-tuning of ProteinMPNN to design **epitope-conditioned nanobodies (VHH) and paired Fvs**
against PINDER-derived antigen targets, scored by a Protenix v2 interface oracle. This branch adds a
dual-encoder epitope-conditioning channel and a dataset-driven, on-the-fly training harness on top of
CLEO's single-target GRPO loop.

> Full design doc: [`docs/SPEC.md`](docs/SPEC.md). Section numbers below (§4.1, §6.9, …)
> refer to it. This file is the digestible map of *what changed and where*.

## The idea in one paragraph

Each training step composes a task on the fly — **one antigen target × one framework scaffold ×
one CDR-length draw** — gaps the scaffold's CDRs out (they start **masked**, no native identity),
decodes new CDRs conditioned on the target epitope, folds the (designed antibody + antigen) with
Protenix, and rewards interface quality + epitope precision/recall + **batch CDR diversity**. The
framework and antigen are always **separate files** with a canonical chain namespace **H / L / T**
(heavy / light / antigen); **there is no pose/dock** — the dual encoder conditions blind. This is
masked-CDR *backbone* generation: Phase 1 (here) trains a **diverse proposal prior** over CDR loops
given an epitope; Phase 2 (later) finetunes that prior online toward one antigen. Everything is gated
so `conditioning.enabled=false` is a byte-identical stock-MPNN run (the M1 baseline).

## Run it

```bash
cleo-design-train --config-path config/design --config-name antibody_composed \
    run_name=<exp> output_dir=<path>
```

Key knobs in `config/design/antibody_composed.yaml`:

| Field | Meaning |
|---|---|
| `dataset.composer.vhh_fraction` | P(sample a VHH scaffold); `1 - p` = Fv. **Default 0.5.** |
| `dataset.composer.split` | `train` (18,385 targets) or `val` (140 held-out). |
| `dataset.composer.cdr_length_ranges` | e.g. `{H3: [8, 20]}` to sample CDR lengths per step; `null` = native span width. |
| `dataset.composer.reward` | Reward package name under `dataset.reward_dir` (default `antibody_interface_composed`). |
| `conditioning.enabled` | Master switch; `false` ⇒ stock MPNN, every hook a no-op. |
| `conditioning.*` | Per-mechanism ablation toggles (§4.1 / §8.1). |

## What changed on this branch

### Data pipeline (in `../data/`, not shipped in git)
- **Epitope precompute** — antigen residues within 5.0 Å of the partner chain; each antigen split to
  a single chain relabelled **T**. Manifest `pinder_epitopes.csv` (+ epitope size, net charge).
- **Trainable set** `pinder_epitopes_trainable.csv` = band **[5, 35] residues ∩ has-MSA** = 18,525
  targets (train 18,385 / val 140).
- **MSAs** — uniref-only pipeline; 12,026 antigen MSAs + 3,498 scaffold-framework MSAs.
- **Scaffold library** — 1,952 clustered reps (395 VHH + 1,557 Fv); VD backbones materialized to
  `scaffolds/vd_structures/{id}.pdb` (chains H / H+L). Pool manifest `scaffold_pool.csv`.

### Reward oracle — generalized to N designed chains (§4.2)
`src/cleo/design/utils/protenix_oracle.py`. `protenix_from_df` went from 1 designed chain to **N**
(VHH = 1, Fv = 2) on one code path. An Fv supplies a `design_chains` df column
`[{length, framework_msa_dir, cdr_spans}, ...]`; the oracle splits the decoded sequence per chain
(`_split_seq`), builds **one CDR-gapped framework MSA per chain** (H gapped at H1–H3, L at L1–L3),
aggregates interface metrics over designed chains (max iptm / min PAE), and counts epitope overlap
near *any* designed chain. A single-chain design falls back to the scalar columns (byte-identical to
the old path). Tests: `tests/unit/test_protenix_oracle.py`.

### Composing dataset — on-the-fly target × scaffold × CDR-lengths (§6.9)
`src/cleo/design/data/composer.py`. `ComposingDataset` (subclasses `DesignDataset`) holds the two
pools and composes one `Example` per `sample()` — no materialized cross-product JSONL. It:
- draws modality (VHH/Fv by `vhh_fraction`) → scaffold → target, independently;
- pins one CDR-length draw into `params.cdr_lengths` so the featurizer's gapping and the reward's
  per-chain split agree;
- emits a **uniform `design_chains`** record (1 entry VHH / 2 Fv) whose per-chain `length =
  native VD length − Σ native span widths + Σ sampled lengths` — exactly the decoded segment length
  the oracle expects.

Reward package `config/design/reward/antibody_interface.yaml` (the single package — the old scalar
variant was retired) requires `design_chains` (instead of the scalar single-chain inputs). It also
adds a **batch CDR-diversity** step (`cdr_diversity.py`): per design, how different its CDR loops are
from the batch's same-type CDRs — **structural** (pairwise CA-RMSD over the folded loops) and
**sequence** (normalized string distance). `DesignDataset.bind_inputs` now routes `${native.seq.T}`
to the separate antigen file. Per rollout the loop logs `is_vhh` into the train-metrics CSV and full
provenance (`scaffold_id`, `kind`, `target_id`, `cdr_lengths`) to `{run_name}_provenance.csv`.
Tests: `tests/unit/test_composer.py`.

### Epitope conditioning (§4.1, slice-2 steps 1–6)
`src/cleo/design/data/epitope.py` + hooks in `src/cleo/design/utils/policy.py`: a second
ProteinMPNN encodes the epitope; CDR nodes get node-init signals + encoder/decoder cross-attn;
coordinate-free CDR graph edges let the gapped (coord-less) CDR nodes participate. Master-switch and
per-mechanism toggles for ablations.
- **Step 5** node signals (each its own toggle, default off): a **CDR-identity** embedding
  (H1/H2/H3, L1–L3), a **per-CDR-type position table** (`node_init_relpos_per_cdr` — a unique
  positional embedding per CDR type), **stem-gap geometry** (the flanking-stem span distance and its
  ratio to the sampled CDR length), and an **attention-pool** of the epitope (learned query)
  replacing the masked mean. The epitope mask is **required**; whole-antigen conditioning is opt-in
  via `allow_whole_epitope` (no silent fallback), and unroutable CDRs (heavy/light mismatch) error.
- **Step 5b** `cdr_epitope_coupler`: an iterated **CDR self-attention ↔ CDR–epitope cross-attention**
  block (`coupler_rounds`), stacked *after* the one-shot encoder cross-attn. It gathers all CDR nodes
  (unioning H+L for an Fv → cross-chain paratope organization), lets them attend among themselves,
  then exchange messages with the epitope, and repeats — writing only CDR positions. Its own toggle,
  default off, ablatable against the one-shot cross-attn.
- **Step 6** `train_framework_encoder`: unfreezes the framework + epitope encoders (all-trainable
  optimizer) and re-encodes the initial state grad-tracked every PPO update instead of caching a
  detached leaf — so node-init / encoder cross-attn / epitope-encoder params train, not just the
  decoder. `false` keeps the cheaper decoder-only regime (byte-identical step-4 path).

### GRPO trainer — no KL penalty
`grpo.py`: the KL to a frozen reference is **no longer subtracted from the loss**. Drift is controlled
by learning rate + the clipped surrogate; the reference KL and the global **grad-norm** are logged as
diagnostics only (grad-norm is measured, not clipped). `use_ref_kl=true` still logs the KL.

Tests: `tests/unit/test_epitope_conditioning.py`, `test_policy_conditioning.py`,
`test_coord_free_edges.py`, `test_gapping.py`, `test_cdr_diversity.py`.

## Status

Built + tested (239 unit tests): data pipeline, N-chain oracle, composing dataset, conditioner
hooks (steps 1–6 + step-5 per-CDR position table / explicit epitope mask + step-5b CDR–epitope
coupler), CDR-diversity reward, no-KL GRPO, and the training config. Phase-2 docked refinement
(`mode="complex"`) is speced (SPEC §4.7) but not built. **Not yet run end-to-end** — the only remaining piece is
the slice-2 step-7 e2e smoke: run `antibody_composed.yaml` a few steps with `conditioning.enabled=true`
(shapes / grad-flow / Protenix reward) plus an `enabled=false` baseline-equivalence check. See §11 of
[`docs/SPEC.md`](docs/SPEC.md).

## Tests

```bash
uv run --extra dev pytest tests/unit --ignore=tests/unit/test_mutation_diversity.py
```
(`test_mutation_diversity.py` is excluded — a pre-existing unrelated import break.)
