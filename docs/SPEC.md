# Antigen-Conditioned Nanobody Design — Spec & Plan

**Status:** draft · **Owner:** jgershon · **Created:** 2026-06-27
**Branch:** new branch off CLEO; expected to diverge from the standard CLEO training protocol.

---

## 1. Motivation

CLEO fine-tunes ProteinMPNN with GRPO **per target** (enzyme optimization): one run sharpens
the CDR/sequence proposal distribution for a single binding site. That overfits to one site and
must be re-run for every new target.

This project generalizes CLEO in two steps:

1. **Multi-target GRPO** — train one policy across *many* antigen targets at once, forcing the
   model to learn general CDR↔antigen interaction principles rather than memorizing one site.
2. **Epitope conditioning** — make sequence generation a function of the antigen/epitope so that,
   at inference, a *new* target yields CDR proposals biased toward binding it — ideally
   **zero-shot**, with no per-target fine-tuning.

The novel crux is (2). (1) is both a prerequisite and an independently testable hypothesis
("does multi-task training alone improve the proposal distribution?").

## 2. Goals / Non-goals

**Goals**
- A single ProteinMPNN-derived policy trained across many nanobody–antigen complexes via GRPO.
- An epitope-conditioning mechanism (complex-graph nodes) that generalizes to held-out targets.
- A pluggable reward oracle, primary = **Protenix v2**, with a cheap proxy slot for the loop.
- Quantified comparison vs. (a) base ProteinMPNN and (b) per-target CLEO on held-out targets.

**Non-goals (for now)**
- Wet-lab validation. Success is measured against the in-silico oracle + retrospective recovery.
- Framework / full-Fv design. We design **CDR positions only**; framework + antigen are fixed.
- Docking against a given complex. **No posed antibody–antigen complex is available or assumed.**
  The CDRs start **masked** and are updated from epitope conditioning; the policy proposes CDR
  loops, and the structure oracle (Protenix) folds them — this is masked-CDR *backbone* generation,
  not refinement of a known pose. See §2.1 for the two-phase framing.
- De novo epitope discovery. The epitope is an input, not something we search over.

### 2.1 What we are actually training (framing)

Two phases:
- **Phase 1 — pretrain a proposal prior (this repo's current target).** Train the policy across many
  antigens so that, given an epitope, it proposes a *diverse* set of CDR-loop backbones that fold
  (via Protenix) into good, epitope-precise interfaces. The goal is a strong prior over "what CDR
  loops fit this ab–ag interaction," not a single optimized binder. Batch **CDR diversity** is
  rewarded (§8) so the prior spreads over backbones rather than collapsing.
- **Phase 2 — online per-target finetuning (later).** Take the pretrained prior and optimize toward
  one specific antigen in an online loop (beam-style: keep the best folded backbones each round and
  keep updating the policy). ProteinMPNN is cheap enough to run this GRPO loop on CPU while folding
  the trajectories' best designs.

## 3. Locked design decisions

| Decision | Choice | Notes |
|---|---|---|
| First milestone | Multi-target GRPO baseline (no conditioning) | Validates the multi-task hypothesis before adding architecture |
| Second milestone | Epitope-conditioning prototype | Dual-encoder + cross-attention (below) |
| Training data | **Any PINDER complex as an epitope source** | Online RL: discard the native binder, keep one partner as "antigen" + its interface as "epitope", task the policy with designing CDRs to bind it. Sidesteps nanobody-data scarcity. |
| Reward oracle | **Protenix v2** (AF3-style) | Interface confidence (ipTM / interface-PAE) **+ epitope-overlap term** (predicted interface must hit the *intended* epitope); pluggable so proxies can substitute |
| Conditioning architecture | **Dual-encoder + cross-attention** | Separate ProteinMPNN encoder for the epitope; CDR positions cross-attend to per-epitope-residue embeddings. **No posed complex required.** Complex-graph nodes kept as documented alternative (§4.4). |
| Codebase relationship | New branch off CLEO | Reuse GRPO core + MPNN policy wrapper; diverge on data loader, multi-target batching, conditioning, reward |

## 4. System architecture

```
                ┌─────────────────────────────────────────────┐
                │  Multi-target GRPO trainer                   │
                │                                              │
  epitopes  ───►│  for each target t (epitope patch) in batch:│
  (PINDER)      │    sample G CDR seqs ~ π_θ(· | fw, epitope_t)│──► reward via oracle
                │    r_i = Oracle(nanobody_i, antigen_t)       │◄── Protenix v2
                │    A_i = (r_i - mean_G) / std_G              │    (or proxy)
                │    grpo_loss += -A_i · logπ_θ(seq_i)         │  (clipped surrogate; no KL penalty)
                │  θ ← θ - η ∇loss   (drift: lr + grad-norm)   │
                └─────────────────────────────────────────────┘
                              │
                              ▼
   Policy π_θ = dual-encoder ProteinMPNN
   ┌── framework encoder ──┐      ┌── epitope encoder ──┐
   │ nanobody fw backbone  │      │ epitope backbone +  │
   │ + masked CDR (UNK seq,│      │ KNOWN epitope seq   │
   │   scaffold backbone)  │      │ (seq injected)      │
   └──────────┬────────────┘      └──────────┬──────────┘
              │  per-fw-res emb              │ per-epitope-res emb
              └──────► CDR positions cross-attend ◄──────┘
                              │
                       framework decoder → CDR sequence (autoregressive)
```

### 4.1 Policy — dual-encoder ProteinMPNN
- Backbone: ProteinMPNN encoder/decoder. Autoregressive over residues ⇒ exact per-sequence
  log-probs, which GRPO needs.
- **Three modules, ALL trainable for this task (LOCKED 2026-07-03):** framework encoder, epitope
  encoder, and decoder are each **initialized from the pretrained ProteinMPNN weights** but **all
  fine-tuned** — no frozen encoder, unlike stock CLEO (which freezes the encoder and trains only the
  decoder). The new modules (CDR node-init mixing, relative-position embedding, cross-attention)
  train from scratch. Rationale: this is a new regime (masked-CDR + epitope conditioning) — the
  framework encoder must learn to produce anchor embeddings useful for the interpolation/cross-attn,
  and the epitope encoder must learn a useful epitope representation; freezing would cap adaptation.
  **Trainer consequence:** the framework encoder's output is no longer cacheable across PPO epochs —
  re-encode every update (§6.5).
- **Two encoders, both initialized from the same pretrained ProteinMPNN encoder weights:**
  1. **Framework encoder** — nanobody framework backbone with CDR positions masked (UNK
     sequence). Structure-only, as in vanilla MPNN (we are designing this sequence).
  2. **Epitope encoder** — epitope backbone with its **known sequence injected. RESOLVED
     2026-07-04: path (b).** Vanilla MPNN's encoder is sequence-agnostic (sequence only enters the
     decoder), so to get the epitope's identity into its embedding we add the sequence embedding
     (`W_s(S)`) into the encoder **node features before the message-passing layers**, then run the
     full encoder so structural message passing mixes sequence + structure and every per-residue
     epitope embedding carries both. (Path (a) — teacher-force the decoder and read hidden states —
     was the alternative; path (b) is simpler, keeps a single forward pass, and gives the per-residue
     keys/values the cross-attention needs.) Built in `data/epitope.py::encode_epitope`.
- **Epitope → CDR conditioning (encoder-side node init, LOCKED 2026-07-03):** each masked CDR
  position's node embedding is built from two sources combined:
  1. **Interpolated anchor init** — a directional N→C sweep that mixes the two flanking framework
     stem residues' **post-encoder** embeddings with weight `i/(L+1)` (so a position near the
     N-stem is mostly N-stem context, near the C-stem mostly C-stem). Gives each CDR node honest,
     framework-anchored structural context **without inventing loop geometry**, and is
     length-agnostic. **Concatenate an explicit relative-position embedding of `(i, L)`** so
     long-loop (e.g. H3, ~24 res) apex positions stay individuated — pure interpolation makes all
     mid-loop positions ≈0.5 blend, and the apex is exactly the contact region. (Use *post-encoder*
     stem embeddings: raw MPNN node features are ~empty; structure only appears after the encoder
     layers. It's an init/prior refined downstream, so exact linearity of the space isn't required.)
  2. **Pooled epitope embedding** — pooled over the epitope encoder, added to every CDR node as a
     global "here is your target" signal. **Attention-pooled (2026-07-04)** — a learned query over
     the per-residue epitope embeddings, not a plain masked mean: since (no pose, §4.5) this is the
     always-on target-specific signal to every CDR node, a richer fingerprint earns its keep.
  **Per-residue cross-attention at BOTH encoder and decoder (LOCKED 2026-07-03).** On top of the
  pooled node-init, each CDR position attends to *individual* epitope residues (keys/values = the
  epitope encoder's per-residue embeddings) at two points:
  - **(a) Encoder** — queried by the CDR node embedding, so CDR representations are epitope-aware
    *before* decoding begins (interacts with the pooled init + rel-pos above).
  - **(b) Decoder** — queried by the per-step decoding hidden state, so the paratope refines its
    contacts as autoregressive sequence context accrues.
  Per-residue (unpooled) lets the H3 apex attend to the epitope hotspot while framework-proximal
  positions attend elsewhere. **Both sites are independently toggle-able** so we can ablate
  encoder-only vs. decoder-only vs. both — `epitope_overlap` is the readout.
- **Decoder:** framework decoder (init from pretrained MPNN decoder) generates CDR sequence
  autoregressively, conditioned on framework structure + cross-attended epitope context.
- **No posed complex required** — epitope and framework are encoded in independent frames; this
  is the reason the dual-encoder is preferred over complex-graph nodes (§4.4).
- **CDR handling (LOCKED 2026-07-03): mask the CDR structure, don't build it.** We do **not**
  generate CDR-loop backbones. Take a native antibody framework backbone (from PINDER), **mask out
  the CDR positions' structure**, and let the **epitope embedding stand in for the missing CDR
  structural context** (the CDR node representations come from the epitope cross-attention rather
  than from loop coordinates). The **number of residues per CDR is sampled from native length
  distributions** at dataset-assembly time (§6.x preprocess), which sets how many masked positions
  each CDR has. This removes the backbone-generation dependency entirely and, because the epitope is
  encoded in an independent frame, makes framework × antigen pairing **freely combinatorial** (any
  PINDER framework backbone with any antigen+epitope — no co-pose, no docking).
- **CDR graph connectivity (RESOLVED 2026-07-04 — coordinate-free, "gapped" CDRs):** the input
  framework PDB is treated as if **the CDRs are fully gapped — masked positions carry no positional
  information at all**, by construction: the model must *design* CDRs that bind the epitope, so it
  cannot be given (and must not leak) any native CDR geometry. Coordinates enter `ProteinFeatures`
  (`cleo/src/cleo/design/protein_mpnn_utils/model_utils.py:799`) via two independent channels and the
  surgery closes both for CDR-incident edges:
  1. **Connectivity — stem-borrowed kNN.** The Cα-kNN needs *some* point to pick each CDR node's
     neighbors. Each masked CDR position borrows the **real Cα of its nearest flanking framework
     stem** *for neighbor-selection only*, so it lands in a sensible real neighborhood (framework
     residues) instead of being orphaned. No fabricated coordinate becomes a feature — the borrowed
     point only chooses *which real residues* are neighbors.
  2. **Edge features — learned "unknown-distance" embedding.** For any edge with a CDR endpoint
     (**both directions**: framework→CDR and CDR→framework/CDR), the 25×`num_rbf` RBF distance block
     is **not computed from coordinates**; it is replaced by a single learned `nn.Parameter` mixed in
     by a CDR-edge gate. The sequence-offset / chain positional edge features (coordinate-free, carry
     CDR length + position) stay active. The pretrained `edge_embedding` therefore never sees an
     out-of-distribution distance — it sees an honest trainable token on exactly the edges where
     geometry is unknown.
  **Rejected: pseudo-coordinate re-anchoring ("B-anchor", 2026-07-04).** Interpolating flanking-stem
  *coordinates* to fill masked CDR atoms was considered and rejected: colinear pseudo-Cα's and the
  `torch.cross`-derived N/C/O/Cb produce inter-atomic distances that never occur in real backbones,
  feeding the pretrained RBF/`edge_embedding` pure OOD input. (The interpolation trick is safe on the
  *node embedding* side — it mixes learned embeddings already in-distribution for the encoder — but
  not on raw coordinates.) Combined with the interpolated + pooled-epitope node-init above, CDR nodes
  become first-class in the decoder without any loop backbone and without fabricated geometry.
- **What the "blank" CDR node is fed (pose-free menu, 2026-07-04).** Since there is **no dock/pose**
  (§4.5) — framework and epitope live in independent frames — no geometric signal is available; the
  epitope is a *learned identity* only. A masked CDR node receives, all leak-free: (1) the
  interpolated flanking-stem anchor + (2) rel-pos `(i, L)` [above]; (3) a **CDR-identity embedding**
  (H1/H2/H3, L1–L3) — strong, cheap, epitope-independent length/composition prior; (4) **stem-gap
  geometry** — the N-stem↔C-stem distance and slack ratio `L / gap` (framework-internal real
  geometry, a *plausibility* prior on whether a length-`L` loop can bridge the gap; not a specificity
  signal); (5) the **attention-pooled epitope** embedding (a learned query over the epitope encoder
  output, richer than a masked mean) + per-residue epitope cross-attention — the **only**
  target-specific channel, so it carries all discriminative load.

### 4.2 Reward oracle — Protenix v2 (primary)
- Interface: `RewardOracle.score(designs: list[Design]) -> list[float]`.
- Protenix v2 implementation: fold the (designed nanobody + antigen) complex; reward from
  interface confidence — candidate signals: `ipTM`, interface-PAE (mean PAE across the
  binder↔antigen interface), pTM. Likely `reward = w1·ipTM − w2·iPAE + w3·epitope_overlap`.
- **Epitope-overlap term is essential.** Without it the policy can score high by binding the
  antigen at some *other*, easier surface and ignore the specified epitope — silently defeating
  the conditioning signal. `epitope_overlap` = fraction of the predicted binder↔antigen
  interface that lands on the intended epitope residues (**precision**). Weights TBD (§9).
- **Two more general interface rewards (added 2026-07-04):**
  - **`epitope_cdr_coverage` (maximize, recall — complements `epitope_overlap`).** Fraction of the
    intended `epitope_residues` that are **contacted by a designed CDR residue** (CDR residues =
    `design_chain` at `params.cdr_spans`; contact = heavy-atom distance < ~4.5Å in the predicted
    `.cif`). `epitope_overlap` asks "of what the binder touches, how much is on-epitope?"; coverage
    asks "of the epitope, how much did the CDRs actually engage?" A binder can be all-on-epitope yet
    grip only one hotspot (high overlap, low coverage), or blanket the epitope while also gripping
    off-site (high coverage, low overlap) — both terms together pin the paratope to the epitope.
    **Pure geometry from the predicted `.cif` → no extra Protenix flag, cheap.**
  - **`cdr_interface_pae` (minimize).** Mean predicted PAE over the **CDR↔epitope submatrix** — rows =
    designed CDR residues (`design_chain`, `cdr_spans`), cols = `epitope_residues` (and symmetric) —
    i.e. how confidently the model places the *paratope* against the *specified epitope*, sharper than
    the whole-interface `chain_pair_pae_mean`. **Cost:** the summary JSON only has chain-pair PAE
    aggregates, so this requires running Protenix with **`--need_atom_confidence true`** to dump the
    residue-level PAE matrix (`_full_data_sample_*.json`) — bigger output + slightly slower fold; gate
    it behind the reward package that requests it.
- **Batch CDR diversity (maximize, added 2026-07-10).** A batch-relative term that rewards each design
  for how different its CDR loops are from the *other designs of the same CDR type in the batch*, so
  the Phase-1 prior spreads over backbones instead of collapsing to one loop per epitope. Two signals
  (`cleo.design.utils.cdr_diversity`): **structural** = mean pairwise CA-RMSD (Kabsch-superposed)
  between same-type CDR loops read from the predicted `.cif` (over equal-length pairs); **sequence** =
  mean normalized string distance between same-type CDR subsequences (always available). Reference-
  free like `mutation_diversity`; weight either/both in `reward_aggregation`. Runs as a step *after*
  the oracle so it can read the folded structures.
- **MSA strategy in the loop (REVISED — see §5.4, calibrated 2026-07-02):** the nanobody must be
  folded **with its scaffold framework MSA (CDR columns gapped)**, NOT single-sequence. This was
  measured, not assumed: on the 1mel known binder, single-sequence nanobody gives interface
  ipTM 0.31 / iPAE 24.8Å (reward nearly blind, and more sampling does not fix it), whereas the
  gapped framework MSA gives ipTM **0.94** / iPAE **4.3Å**. The framework MSA is fixed per
  scaffold; only the CDR query-row residues change per design (§5.4). The **antigen is fixed per
  target → precompute & cache its MSA once**. Both MSAs are amortized (framework per scaffold,
  antigen per target); neither is recomputed per rollout.
- **Cost is the central risk** (AF3-scale model inside an RL loop). Mitigations, in the spec as
  knobs not commitments:
  - Cheap proxy as a pre-filter / dense reward (ESM-IF logP, or a distilled surrogate), Protenix
    as sparse/periodic ground-truth reward.
  - Reduced recycles / no MSA (single-sequence) / batched inference.
  - Cache by (epitope, CDR-seq) hash.
- Pluggable so M1 can start on a proxy and swap in Protenix without trainer changes.

### 4.3 GRPO trainer (multi-target)
- Group = G samples for a single complex/target; advantage normalized **within** the group
  (no critic/value net).
- A training batch spans **many targets**; each contributes its own normalized group.
- **No KL penalty.** Drift is controlled by the learning rate + the clipped surrogate objective.
  The KL to a frozen reference and the global **grad-norm** are *logged as diagnostics only* (KL is
  not subtracted from the loss and grad-norm is not clipped), so we can watch drift without
  penalizing it (revisit only if runs actually diverge).
- Reuse CLEO's GRPO loop; new code = multi-target batching + per-group reward bookkeeping.

### 4.4 Alternative conditioning (documented, not chosen)
- **Complex-graph nodes:** antigen/epitope residues as nodes in a single MPNN graph, edges
  encode CDR↔epitope contacts. Geometrically precise but **requires a posed complex** at both
  train and inference time (the unresolved pose problem). Revisit if dual-encoder conditioning
  proves too geometrically coarse.
- **ESM cross-attention:** encode antigen with a protein LM, cross-attend during decoding. Most
  flexible, least structurally grounded.

### 4.5 Two-stage design strategy — pose-free prior + per-antigen specialization (2026-07-04)
Framework and epitope are in **independent frames — there is no dock or pose** at proposal time (the
model must design a binder blind). This splits the work into two stages that share the conditioner
and the reward oracle:

- **Stage 1 — the broad proposal prior (build now).** Train **across many antigens** (the multi-target
  data layer, §6) to learn a general `p(CDR | epitope_identity)`: framework scaffold in its own frame,
  gapped CDRs (§4.1), epitope as a learned seq+structure embedding, per-residue cross-attention doing
  all target discrimination. **No geometry anywhere** — the model learns the epitope→CDR mapping from
  reward alone. Stage 1's job is to be a good *starting distribution*, not a sharp optimizer: its
  success metric is "broad, epitope-responsive proposals," and over-reward-maxing it on narrow targets
  would mode-collapse the very exploration Stage 2 needs — so keep entropy/diversity preserved
  (`temperature=1.0`, KL to reference, don't over-weight the reward). This is exactly what the 3×3
  bake-off (§8.1) probes before committing broad-training compute.
- **Stage 2 — per-antigen specialization (seams now, hooks later, earned by the bake-off).** Off the
  Stage-1 checkpoint, specialize to **one antigen** — either **inference-time** conditioning (no weight
  updates) or a **short GRPO trajectory** on that single target (the one-example-per-rollout seam
  already keeps the group homogeneous, so this is just the dataset pointed at one row, resumed from the
  Stage-1 ckpt). **The pose is free:** every Stage-1 reward evaluation already folds *and docks* the
  design through Protenix, so the predicted complex — framework + designed CDR + epitope in one frame —
  is a **byproduct of the reward call**, not a separate docking step. Stage 2 *reads that structure*
  (instead of only the scalar reward) and unlocks everything geometry made impossible in Stage 1:
  anchor→epitope vectors, distance-biased cross-attention, and the predicted CDR coords as a hypothesis
  to refine around (relaxing the §4.1 surgery — trust the predicted geometry). New pose-dependent
  toggles (`pose_epitope_vectors`, `geometric_cross_attn`, `use_predicted_cdr_coords`), OFF until a
  `pose` feature dict is supplied.
- **Caveats to carry into Stage 2 (not blockers now):**
  1. **Gated on Stage-1 signal.** If the epitope embedding doesn't move convergence in the bake-off,
     Stage 2 is refining garbage — build Stage 2 only once the bake-off shows the embedding earns its
     keep.
  2. **Pose-confidence gate.** A bad Stage-1 proposal gets a low-confidence dock; refining around a
     wrong pose reinforces error. Stage 2 conditions on the predicted complex **only when Protenix
     confidence (ipTM / interface-PAE) clears a bar**; below it, fall back to pose-free.
  3. **Oracle-hacking.** Self-conditioning on the oracle's own prediction tightens the RL feedback
     loop — the model can learn sequences Protenix confidently *mis-scores* as binders. Keep a
     held-out check / second oracle in mind before Stage 2 runs hot.

- **Stage-2 inference — pose-free→docked beam search (user 2026-07-05).** The two stages compose at
  *inference* as a multi-round beam over one antigen, using the free pose (§4.5) to bridge rounds:
  1. **Seed (pose-free, Stage-1 prior).** Pick a set of frameworks + the target antigen; sample an
     initial pool of designs with masked CDRs and **no dock** (the Stage-1 conditioner).
  2. **Fold.** Run the pool through the reward oracle → predicted complexes (framework + designed CDR
     + epitope in one frame) + confidence.
  3. **Cluster + select.** Cluster the folded designs (by CDR sequence and/or docked epitope
     geometry) and keep the **best of each cluster** (reward × pose-confidence) — diversity-preserving
     beam, not top-k, so the beam doesn't collapse onto one mode.
  4. **Re-propose (docked, Stage-2).** For each survivor, condition the *next* round on its **predicted
     complex** (the dock PDB from step 2 — pose toggles ON) and sample refined designs around it.
  5. Repeat 2–4 until reward/confidence plateaus.
  This is the beam realization of Stage 1 (round-0 seeding) → Stage 2 (rounds ≥1 pose-conditioned
  refinement); the per-round pose gate (caveat 2) decides which survivors are trustworthy enough to
  condition on vs. fall back to pose-free.

### 4.6 Backlog — parked ideas (PR #27, revisit later, not built)
Captured so they aren't lost; none are on the Phase-1 critical path.
- **Variable CDR length via a stop token.** Instead of pre-specifying each CDR length, let the model
  emit a stop token (possibly a repurposed mask token) so length is generated, not fixed. Likely
  needs teaching the token — e.g. randomly injecting it early. A neat variant, not needed for the
  first iteration.
- **Beam-style online per-antigen finetuning** — already sketched in §2.1 / §4.5 (rounds ≥1); the
  docked-refinement primitive it needs is §4.7.

*(Built since first parking: the **alternating CDR self-attn ↔ CDR–epitope coupler** — now the
`cdr_epitope_coupler` toggle, §4.1 step 5b. The **un-mask/dock** idea is now speced concretely as
§4.7.)*

### 4.7 Phase 2 — docked refinement (`featurize mode="complex"`), planned
The PR #27 "un-mask the parser" suggestion is directionally right — there **is** a Phase-1 (masked,
pose-free) mode and a Phase-2 (docked) mode — but a single parser boolean is the wrong knob, because
"un-mask" silently bundles two orthogonal changes and one hidden sub-choice:
1. the CDR positions gain **coordinates** (no longer gapped / coord-free), and
2. there is now a **shared pose**, so the antigen should enter the *same* graph (docked) with real
   CDR↔epitope contact edges instead of sitting in the separate epitope encoder;
3. (hidden) you may keep the CDR **coords** but still **re-mask the sequence** to re-decode — so
   "un-mask" is not even binary.

Also, in Phase 2 the CDR coords come from the **predicted** complex (a folded Phase-1 design), **not**
the native scaffold — so it is "ingest a predicted pose," not "keep the native CDRs."

**Plan — express it as the reserved `featurize_example(mode=…)` seam + optional `pose` arg + a
pose-confidence gate, not a parser flag:**
- `mode="pose_free"` (Phase 1, built): gap CDRs (coord-free), antigen in the separate encoder — the
  current path. The parser resolving CDRs as non-existent (`test_gapping.py`) is this mode.
- `mode="complex"` (Phase 2, planned): ingest a predicted `pose` (best folded Phase-1 backbone),
  CDRs carry coords, antigen residues join the MPNN graph with real CDR↔epitope edges; conditioning
  shifts toward the geometric / complex-graph route (§4.4). Sequence may be kept or partially
  re-masked (choice 3 above) per refinement round.
- **Pose-confidence gate (§4.5 caveat 2):** only condition on a predicted pose when it is trustworthy
  (oracle confidence over threshold); otherwise fall back to `pose_free`. A naked parser flag cannot
  express this fallback — which is exactly why the seam, not the flag, is the right mechanism.

The two seams already exist (the optional `pose` arg and the `featurize_example(mode=…)` switch, see
the §6.8 "deferred to Stage 2" note), so `mode="complex"` is an additive build, not a rearchitecture.

## 5. Data

- **Primary source: PINDER as an epitope source (online-RL reframing).** Because training is
  online against a structure oracle, we do *not* need nanobody-antigen pairs. Take any PINDER
  dimer, keep one chain as the **antigen** and its interface as the target **epitope**, discard
  the native binder, and task the policy with designing CDRs to bind that epitope. This expands
  usable data from ~10³ nanobody complexes to PINDER and removes the scarcity risk.
- **Both chains are antigens.** Each heterodimer chain in [50,300) residues is an independent
  antigen-target, with its partner defining the epitope; the partner is discarded and replaced by
  the designed nanobody, so the partner's size is irrelevant. (Same antigen + different partner =
  different epitope = a valid distinct target.)

### 5.1 Materialized subset (PINDER release 2024-02, built 2026-06-28)
Filters (per system): heterodimer (`cluster_id_R != cluster_id_L`); antigen chain length in
**[50, 300)** (≥50 keeps folded domains, drops short peptides); **exclude** `contains_antibody`
(reserved for held-out nanobody eval) and the curated benchmark subsets (`pinder_s/xl/af2`).

- **Train:** `split == 'train'`, one representative per system `cluster_id` (best resolution) →
  **17,521 complexes** downloaded. Using both chains → **26,696 antigen-targets** spanning
  **7,821 unique antigen-chain clusters** (`cluster_id_R`/`cluster_id_L`).
- **Val (honest held-out):** `split == 'val'`, same filters, **PLUS enforced antigen-chain-cluster
  disjointness from train** — drop any val antigen-target whose antigen-chain cluster appears among
  *any* train antigen-chain cluster. PINDER's native splits only guarantee *interface/system-level*
  (`cluster_id`) separation; because we redefined the task to be **antigen-chain-centric**, we must
  additionally exclude antigen-chain homologs or the val antigen may have been seen in training.
  Raw val 438 antigen-targets → **236 honest antigen-targets / 231 antigen clusters / 178 systems**.
  **No in-loop validation (2026-07-10):** the training loop does *not* evaluate the val split during
  training — it only trains on `split == 'train'`. Held-out val is scored **offline** on saved
  checkpoints (§8), keeping the loop cheap and the val set untouched until a run finishes.
- **Files:** complexes in `~/pinder/pdbs/` (train, native `{id}.pdb`, both chains R/L) and
  `~/pinder/pdbs_val/`. Manifests in `~/projects/antibody_rl/data/`: `pinder_targets.csv` +
  `pinder_target_ids.txt` (train systems), `pinder_train_antigen_targets.csv` (per-chain targets),
  `pinder_val_targets_honest.csv` + `pinder_val_ids_honest.txt` (honest held-out). Bucket
  `gs://pinder/2024-02` is anonymously accessible (gcsfs); tooling in `~/git/pinder-env`.

### 5.2 MSAs — not shipped, computed locally
PINDER provides structures + metadata only (Neff stats + `sequence_database.parquet`), **no MSAs**.
We generate them ourselves (local `mmseqs` no-expand pipeline + `/net/databases/colabfold`, §4.2).
**Two MSAs per fold** (see §5.4 — the "nanobody is single-sequence" assumption was disproven):
(1) the **antigen MSA** — depends only on the antigen *sequence*, compute **one per unique antigen
sequence and cache** (~≤7,821) and reuse across every epitope/system sharing that antigen; and
(2) the **nanobody framework MSA** — depends only on the *scaffold framework*, compute once per
scaffold and reuse across every design on that scaffold, with CDR columns gapped per design.

### 5.3 Other
- **Per-target preprocessing:** epitope = antigen residues at the native interface (contact
  distance), extract antigen backbone + sequence (sequence needed for the epitope encoder, §4.1).
- **Nanobody framework:** a fixed canonical VHH scaffold; CDR positions are the design region
  (CDR backbone handling is the open caveat in §4.1/§9). The scaffold carries a precomputed
  **framework MSA** (fed to the oracle with CDRs gapped, §5.4); CDR backbone handling for the
  *policy* encoder is separate (§4.1) from CDR gapping in the *oracle* MSA (§5.4).
- **Lab assets** (reuse, don't rebuild): `~/ab_data/`, `/net/databases/antibody/`,
  `/net/databases/mpnn/` (incl. `antibody_mpnn_model_weights`, `msa_mpnn`, `fused_mpnn`,
  `ligand_mpnn` — candidate policy inits), MSA DBs + colabfold/hhsuite. **SAbDab/SNAC-DB nanobody
  complexes reserved for held-out evaluation** (real nanobody-antigen pairs), not training.

### 5.4 Per-design antibody MSA — framework MSA + CDR gapping (REQUIRED for the oracle)

Applies to **both scaffold kinds** in the library (`scaffolds.csv:kind`): VHH nanobodies (single
chain, CDRs H1/H2/H3) and **paired Fv** (heavy + light, CDRs H1/H2/H3 **and** L1/L2/L3). The
process below is per-chain and identical for each; a paired Fv simply runs it twice.

Calibration (2026-07-02, memory `protenix-oracle-setup.md`) proved the oracle reward is nearly
blind unless the nanobody is folded with its **framework MSA**, and that **gapping the CDR columns
costs nothing at the interface** (CDR3/paratope gets zero MSA signal either way, yet the interface
is confidently placed). So every fold in the loop needs a nanobody a3m that is: framework homologs
aligned to the scaffold, **CDR columns replaced by `-` in all homolog rows**, and the **query row
carrying the freshly-designed CDR residues**. The framework part is fixed per scaffold; only the
query-row CDRs (and their lengths) change per design.

**Two-stage setup — precompute once per scaffold, splice per design:**

1. **Precompute (once per scaffold framework, offline).** Generate a deep framework MSA whose
   query row is the scaffold's exact VD sequence so CDR indices map 1:1, then gap the CDR columns.
   - MSA search: `bench/gen_nb_msa.sh` (single-seq run of the uniref30 no-expand pipeline from
     `data/run_mmseqs_noexpand.sh`; antibody frameworks are abundant → depth ~3–4k).
   - Gap CDRs: `bench/gap_cdrs.py` (walks the a3m tracking the query-column index — lowercase =
     insertion, doesn't advance; replaces homolog chars at CDR columns with `-`, drops insertions
     inside CDR spans; leaves the query row intact). Produces the cached **gapped framework MSA**.
   - CDR spans come from `data/scaffolds/scaffold_chains.csv` (`cdr_H1/H2/H3` for the heavy/VHH
     chain, `cdr_L1/L2/L3` for the light chain of a paired Fv = half-open indices into that chain's
     `vd_seq`; e.g. 1mel_A → H1 24:31, H2 50:56, H3 97:121. For a paired Fv, `scaffold_chains.csv`
     has one row per chain (`ctype` H/L, `chain`), each with its own `vd_seq`/`vd_hash` and CDRs).
   - **Rep caveat:** scaffold MSAs in `data/msa/scaffold_msa_dirs/{vd_hash}/` were built only for
     the **1,952 cluster reps** (`scaffolds.csv:is_rep`), keyed by `vd_hash`. A non-rep scaffold
     (e.g. 1mel_A) has no precomputed dir — either (a) restrict training scaffolds to reps, or
     (b) generate a framework MSA per sampled scaffold's exact VD seq (preferred for column-exact
     gapping, since a rep's MSA is aligned to the rep's framework, not the sampled one).
     `data/scaffolds/vd_to_hash.csv` maps `scaffold_id → vd_hash`.

2. **Per-design splice (in the loop, cheap, no search).** The gapped framework MSA's homolog rows
   are already all-`-` across CDR columns, so per design just: for each CDR, write the designed
   residues into the **query row** and set that CDR block to `L_cdr` gaps in **every homolog row**
   (resize the gap block to the sampled CDR length — variable-length CDRs are handled here). Emit
   the per-design `non_pairing.a3m`, point the chain's `unpairedMsaPath` at it. No mmseqs call.

**Paired Fv specifics.** Run stages 1–2 independently for the heavy and light chains, each with
its own framework MSA + its own gapped CDR columns → two per-design `non_pairing.a3m` (one per
`unpairedMsaPath`). **Do not H/L-pair the MSA** (leave `pairedMsaPath` unset / query-only):
antibody heavy and light chains do not co-evolve like a natural operon, so a paired MSA carries no
real covariation signal — this matches AF3/ColabFold antibody practice and keeps `need_msa_search`
offline (every chain still has an unpaired path). So a paired-Fv complex = 3 chains to the oracle
(heavy fw-MSA + light fw-MSA + antigen MSA), each with `unpairedMsaPath`, none paired.

**Where this lives in code (VHH per-design splice: DONE):** implemented inside `protenix_from_df`
(`cleo/src/cleo/design/utils/protenix_oracle.py`). With `use_msa: true` + df/cfg `framework_msa_dir`
+ `cdr_spans`, `_build_design_fw_msa` writes each design's `non_pairing.a3m` (designed sequence as
the query row + `_gapped_fw_homolog_rows`, the `gap_cdrs.py` gapping cached per (scaffold a3m, CDR
spans) via `lru_cache`). Verified byte-identical to `bench/gap_cdrs.py` on the 1mel framework MSA
(3424 rows, 0 mismatches). **This is the fixed-length special case** (asserts designed length ==
framework query length), correct when the design keeps native CDR lengths.
**Generalization needed for sampled CDR lengths (§4.1/§6.7):** since we sample CDR lengths from
native distributions, the designed length ≠ native. Store the framework MSA **framework-columns-only**
(native CDRs stripped from query *and* homologs) and, per example, **insert `L_cdr` gap columns**
at each CDR position (homologs all-`-`, query = designed residues). CDR length then becomes a free
parameter and the a3m columns stay consistent. Still **to build:** this variable-length path; the
one-time `precompute_fw_msa(vd_seq)` wrapping `gen_nb_msa.sh` (raw framework MSA per scaffold) +
framework-columns-only stripping; and the **paired-Fv** case — the current oracle folds
nanobody+antigen (2 chains); paired Fv needs a per-chain fw-MSA list (heavy + light + antigen).
Validated bench assets: `bench/{gen_nb_msa.sh,gap_cdrs.py,run_calib_gapped.sh,in_msa_gapped.json}`.

## 6. Training harness — dataset-driven multi-example

CLEO today trains on a single backbone PDB per run (`featurize_pdb(cfg.pdb)` in the loop). We
generalize to a **dataframe of examples** — enzyme, antibody, or anything — where each row points
to a named **reward package**. The single-PDB flow becomes the degenerate 1-row case (fully
backward compatible). Three objects, each kept in the form it's good at:

- **Example** (one dataframe row) — the *what*. Owns everything target-specific: `structure`,
  mask, `task`, `reward` name, `params`.
- **Reward package** (a Hydra config group; few + reusable) — the *how*: an ordered pipeline of
  steps (prep + oracle + processing) and how to scale/combine metrics. **Target-agnostic.**
- **Dataset** — the per-example source. **The antibody runtime uses the online `ComposingDataset`
  (§6.9), not a materialized JSONL** — it composes each `Example` on the fly from two small pools.
  The JSONL-on-disk form below is the generic `DesignDataset` schema (still used by simpler tasks and
  by the offline assembler); an `Example` has the same shape however it is produced.

Locked (2026-07-03): per-example params reach the oracle as **df columns** (`protenix_from_df` reads
`antigen_sequence` / `antigen_msa_dir` / `design_chains` / `epitope_residues` from df columns);
**uniform-random, one example per rollout step** — keeps the GRPO group homogeneous so group-relative
advantage stays valid and per-example reward-scale differences wash out. (Runtime composition is
online + seedable; §6.9.)

### 6.1 Example schema (JSONL row)
```jsonl
{"id": "1mel_vs_lysozyme", "task": "nanobody_design", "reward": "antibody_interface",
 "structure": "data/scaffolds/pdb/1mel.pdb",
 "design_chain": "A",
 "design_regions": ["cdr_H1", "cdr_H2", "cdr_H3"],
 "params": {
   "antigen_sequence": "${native.seq.T}",
   "antigen_msa_dir": "data/msa/msa_dirs/b3909cb75837",
   "framework_msa_dir": "data/msa/scaffold_msa_dirs/1b473c2a20ce",
   "cdr_spans": {"H1": [24,31], "H2": [50,56], "H3": [97,121]},
   "epitope_residues": [62,63,73,101,102]
 }}
```
- **Chain convention (LOCKED 2026-07-04):** the **design chain is `A`** for a VHH, **`A`/`B`** for a
  paired Fab, and the **antigen/target chain is always `T`**. `design_chain` is an explicit field
  drawn from this convention (`"A"` or `["A","B"]`); binding refers to the antigen as `${native.seq.T}`.
  Structures are relabeled to this convention by the offline assembler (§6.7) so the loop never has to
  guess chain roles.
- `task` selects the featurizer / mask-resolver.
- **`params.cdr_spans` is the authoritative masked region (LOCKED 2026-07-04).** `design_regions`
  (`cdr_H1`…) are **human-readable labels only**; the numeric spans in `params.cdr_spans` define what
  `mask.py` masks, because CDR lengths are **sampled offline** (§6.7) so they no longer match the
  scaffold's native spans in `scaffold_chains.csv`. `scaffold_chains.csv` is consumed **only by the
  offline assembler**, never by the runtime mask resolver. `mask.py` emits `fixed_residues` = the
  **complement** of the union of `cdr_spans` on `design_chain` (for `task: monomer`, design_regions =
  whole chain → `fixed_residues` = ∅).
- A single-seq single-objective example is the **same schema**: `task: monomer`, `design_regions`
  = whole chain, `reward: monomer_stability`, thin `params`. Same loop, no special-casing.

### 6.2 Reward package (config group)
```yaml
# config/design/reward/antibody_interface.yaml
requires: [antigen_sequence, antigen_msa_dir, framework_msa_dir, cdr_spans]   # validated per-row
steps:
  - name: protenix                       # ORACLE step (builds the per-design MSA internally)
    target_fn: cleo.design.utils.protenix_oracle.protenix_from_df
    cfg: {cycle: 4, n_diffusion_step: 20, n_sample: 1, use_msa: true,
          need_atom_confidence: true}    # STATIC; atom_confidence on -> residue-level PAE for cdr_interface_pae
    inputs:
      antigen_sequence: ${row.params.antigen_sequence}
      antigen_msa_dir:  ${row.params.antigen_msa_dir}
      framework_msa_dir: ${row.params.framework_msa_dir}   # scaffold's CDR-gapped framework MSA
      cdr_spans:        ${row.params.cdr_spans}            # -> per-design gapped a3m + CDR-residue masks
      epitope_residues: ${row.params.epitope_residues}     # for epitope_cdr_coverage + cdr_interface_pae
reward_aggregation:   # REAL UniversalReward API: reward_aggregation + lower_bound/upper_bound
  - {metric: protenix_interface_iptm,       mode: max, lower_bound: 0.0, upper_bound: 0.8,  weight: 1.0}
  - {metric: protenix_epitope_overlap,      mode: max, lower_bound: 0.0, upper_bound: 1.0,  weight: 1.0}  # precision
  - {metric: protenix_epitope_cdr_coverage, mode: max, lower_bound: 0.0, upper_bound: 1.0,  weight: 1.0}  # recall
  - {metric: protenix_cdr_interface_pae,    mode: min, lower_bound: 0.0, upper_bound: 30.0, weight: 1.0}  # paratope PAE
```
`requires` + `steps.*.inputs` are the package's declared contract. The framework-MSA gapping is
**not a separate step** — with `use_msa: true` and `framework_msa_dir` + `cdr_spans` bound,
`protenix_from_df` assembles each design's framework MSA itself (`_build_design_fw_msa`: designed
sequence as the query row + the scaffold's CDR-gapped homolog rows, cached per scaffold via
`_gapped_fw_homolog_rows`). Verified byte-identical to `bench/gap_cdrs.py`. Enzyme / monomer
packages are the same shape with different steps (e.g. `boltz_from_df` + a catalytic-geometry step
+ `compute_dist_to_ref_seqs_from_df`).

### 6.3 Binding + validation (explicit, no implicit fallback)
Inputs are **templates** resolved per-example against named sources. There is **no `??` fallback** —
you name the source explicitly, and any unsatisfied argument is a **hard error at dataset build**:

| source | resolves to | example |
|---|---|---|
| `${row.params.X}` | a field of the dataframe row | `${row.params.epitope_residues}` |
| `${native.seq}` | native seq of the **designed chain** (`design_chain`, default `A`), parsed from `structure` | drift-to-wildtype reference |
| `${native.seq.<chain>}` | native seq of a named chain | `${native.seq.T}` = co-crystal antigen |
| `${design.seq}` | the sampled sequence under score (usually implicit — the df `sequence` col) | |
| literal | plain YAML value | `5.0` |

- **Reference sequence is always explicit.** A Hamming-to-reference reward writes either
  `${row.params.reference_seq}` (a specific engineered target) **or** `${native.seq}` (the
  protein's own sequence) — whichever the example intends. Never a silent substitution.
- `${native.seq.<chain>}` also removes duplication for co-complex scaffolds: 1mel's antigen *is*
  chain L of the input PDB, so the row points at it instead of pasting 129 residues (and it can't
  drift out of sync with the structure).
- **Validation is part of making the dataset.** `DesignDataset.load()` iterates every row, looks up
  its reward package, and resolves every binding in `requires` + `steps.*.inputs`. If any required
  input is unbound, references a missing `row.params` key, or names a `native.<chain>` absent from
  the structure → **raise immediately**, naming the row `id`, the package, and the offending
  argument. A dataset that loads is guaranteed that every example fully satisfies its reward's contract.
- **Validation is cheap, not a 26k-PDB parse (LOCKED 2026-07-04).** At build we check the *contract*,
  not the residues: `structure` file exists, the referenced chains (`design_chain`, every
  `native.<chain>`) are **present** (a light chain-ID header scan, no full atom parse), and every
  binding resolves. Actual per-chain **sequences are parsed lazily at featurize time and cached** — so
  a `${native.seq.T}` reference is validated for *chain presence* at build but its sequence string is
  only materialized when the example is first sampled. Contract is still guaranteed before training.

### 6.4 The loop seam (3 changes, backward compatible)
```python
# policy.py train()
example      = self.dataset.sample()                 # uniform random, one per step
feature_dict = self.featurize_example(example)       # was featurize_pdb; design_regions -> chain_mask
reward_fn    = self.reward_registry.bind(example)    # resolve package bindings -> df columns
to_log       = self.train_step(step, init_state, feature_dict, reward_fn)
```
- `featurize_example` loads `example.structure` (`parse_PDB` already yields per-chain sequence →
  populates the `native.*` context for free) and resolves `design_regions` into the
  `chain_mask` / `fixed_positions` it already builds. A run with `cfg.pdb` set is an implicit 1-row dataset.
- `reward_registry.bind` returns a `UniversalReward` whose `get_input_df` seeds resolved bindings as
  df columns — the exact mechanism the oracle already consumes. No change to scoring functions.

### 6.5 Training step — per-example conditioning + gradient accumulation

**Conditioning invariant.** Each sampled sequence's policy-gradient term
`∇θ log π_θ(seq | backbone, epitope)` **must** be evaluated under *that example's own*
conditioning — its backbone graph **and** its epitope encoding. Sequences from one example are
never scored under another's conditioning. GRPO advantages are group-normalized **strictly within
an example** (this is what cancels per-target reward-scale differences); only those *detached
scalars* are ever shared across examples.

**Multi-example updates use gradient accumulation, not input mixing.** Run a separate,
properly-conditioned forward/backward per example and **sum the gradients into the shared trainable
params** before one `optimizer.step()`. Exact because `∇θ Σ_k L_k = Σ_k ∇θ L_k`, each `L_k`
correctly conditioned. This gives the shared params an *averaged multi-target* gradient instead of
the single-target zig-zag you get stepping after each structure — essential once the conditioning
encoder is shared across targets.

**All three modules are trainable (§4.1) — so nothing structural is cached across PPO epochs.**
θ = framework encoder + epitope encoder + decoder + the new conditioning modules (CDR node-init
mixing, relative-position embedding, per-residue cross-attention at **encoder + decoder**). Because the framework
encoder is in θ, its output `h_V/h_E/E_idx` changes every gradient step, so each PPO inner update
must **re-encode both the framework and the epitope** for every example, then decode — a full policy
forward under the current θ. (The rollout-phase encode is `no_grad`, used only to sample; it is not
reused for the gradient.) This is heavier than stock CLEO's frozen-encoder path, but both encoders
are small (3 layers, 128-dim) and cheap next to the reward folds; the real cost to watch is
**backprop memory** (K examples × full encoder+decoder) — reduce K or `N_updates` if memory-bound.

So a per-example conditioned forward (every inner update) = framework-encode(masked backbone) +
epitope-encode(epitope) + CDR node-init (interpolated anchors + pooled epitope) + encoder-side
per-residue epitope cross-attn + decoder with decoder-side per-residue epitope cross-attn.

```python
# --- rollout phase (no_grad): per example, sample + group-normalized advantages ---
groups = []
with torch.no_grad():
    for ex in dataset.sample(K):                   # K examples this step (task-batch)
        feat = featurize_example(ex)               # masked backbone graph + CDR mask
        epi  = load_epitope(ex)                    # epitope backbone + sequence
        out  = policy_forward(feat, epi)           # encode(fw)+encode(epi)+node-init+decode -> sample
        R, _ = reward_registry.bind(ex)(step, out, feat)   # folds -> per-example reward
        A    = (R - R.mean()) / (R.std() + 1e-3)   # group-normalized WITHIN ex (pure GRPO)
        groups.append((feat, epi, out["S"], out["log_probs"].detach(), A))

# --- update phase: N_updates PPO epochs, accumulate grads across examples ---
for _ in range(N_updates):
    optimizer.zero_grad()
    for (feat, epi, S, old_logp, A) in groups:
        out   = policy_forward(feat, epi, decoding_order=..., sampled_actions=S)  # RE-encode fw+epi, decode
        ratio = exp(sum(out["log_probs"] * onehot(S)) - old_logp)
        L     = -ppo_clip(ratio, A)                # (+ optional KL to frozen ref, per-example)
        (L / K).backward()                         # ACCUMULATE into shared θ (all 3 modules)
    optimizer.step()
```

**Knobs.** Task-batch **K** (examples/step) × group size **B** under a fixed fold budget `K·B`
(fold count is unchanged vs. single-structure stepping — only the step cadence changes). Use **pure
group-relative** advantages, *not* the running-mean baseline (`use_avg_reward`) — a reward history
pooled across heterogeneous targets is meaningless. Bigger K → lower cross-target gradient variance
(matters most for the *shared* epitope encoder); bigger B → better within-target advantage
estimates. Start K≈4–8, B≈8–16.

**Real-code refactor.** `grpo.PolicyMPNNvGRPO.train_step` currently takes a single
`(init_state, feature_dict)`; generalize to a **list of example-groups**. `policy.py train()`
replaces the single `featurize_pdb(cfg.pdb)` with `dataset.sample(K)` + building K groups.
**Unfreeze the encoder** — stock CLEO calls `freeze_encoder` and puts only the decoder in Adam;
here the optimizer must span **framework encoder + epitope encoder + decoder + the conditioning
modules** (§4.1), and the cached-`init_state` shortcut is removed (re-encode every update).

### 6.6 Module layout
```
cleo/src/cleo/design/data/          # DATA LAYER — BUILT + unit-tested 2026-07-04
  dataset.py     # DesignDataset.load(jsonl, reward_dir) -> validate every row's reward
                 #   contract (hard error, names row id); .sample(k), .bind_inputs(ex),
                 #   .fixed_residues(ex, ...); NativeSeqProvider (lazy gemmi seq parse,
                 #   cached); scan_chain_ids (light ATOM-record chain-presence check); Example
  mask.py        # resolve_fixed_residues(chain_letters,R_idx,icodes, design_chain, cdr_spans)
                 #   -> COMPLEMENT token string ({chain}{R_idx}{icode}, matches parse_PDB);
                 #   authoritative region = params.cdr_spans; H*->chain A, L*->chain B
  binding.py     # resolve_one/resolve_inputs: ${row|native|design|literal}; DESIGN_SEQ
                 #   sentinel; hard error on unsatisfied requires / missing chain / missing key
  epitope.py     # BUILT + tested 2026-07-04 (moved into data/ 2026-07-10). ConditioningConfig (master `enabled` +
                 #   one flag per mechanism = ablation surface); EpitopeConditioner with 3
                 #   hooks — encode_epitope (path-b seq-injected message passing), condition_nodes
                 #   (CDR node-init: interp anchors + rel-pos + pooled epitope, then encoder
                 #   cross-attn; CDR-only), decoder_cross_attn (per-step, caller-gated). All hooks
                 #   exact no-ops when disabled (byte-identical M1 baseline). tests/unit/
                 #   test_epitope_conditioning.py (11). NOT yet wired into policy — see §6.8 (slice 2).
data/preprocess/            # OFFLINE dataset assembly (produces the JSONL the data layer consumes)
  cdr_length_dists.py       # empirical per-CDR native length distributions (sampling source)
  assemble_antibody.py      # sample antigen+epitope × framework, sample CDR lengths -> JSONL rows
config/design/reward/
  antibody_interface.yaml   # BUILT (requires + steps.inputs contract + reward_aggregation)
  _base_reward.yaml  monomer_stability.yaml  enzyme_catalytic.yaml   # TODO
```
**Gotcha (resolved):** reward packages are loaded via `OmegaConf.to_container(..., resolve=False)`
so our `${native.seq.T}` templates survive as literal strings — OmegaConf would otherwise try to
resolve them as *its own* interpolations and raise `InterpolationKeyError`. The data layer therefore
operates on plain dicts; the reward **registry** (policy track, §6.4) re-loads `steps` +
`reward_aggregation` into a `UniversalReward` and overrides `get_input_df` to seed the values from
`.bind_inputs(ex)` as df columns. `requires`/`inputs` are registry metadata, never passed to
`UniversalReward.__init__`.
**Registry + loop seam — BUILT + unit-tested 2026-07-04.** `data/registry.py`: `RewardRegistry.bind(
example)` -> `BoundReward` (a `UniversalReward` subclass whose `get_input_df` seeds the resolved
per-example inputs as df columns; `inputs` stripped from steps, `${design.seq}` aliases the `sequence`
column). `policy.py`: `_featurize(pdb, fixed_residues_fn)` refactor + `featurize_pdb` (cfg.fixed_residues)
and `featurize_example` (CDR-mask complement via `dataset.fixed_residues`); `__init__` loads a
`DesignDataset` + `RewardRegistry` when `cfg.dataset` is set (else the single-PDB path, `cfg.reward`
now optional); `train()` samples one example/step, featurizes+masks it, binds its reward, and passes it
to `train_step(..., reward_fn)`. Both `PolicyMPNN.train_step` and `grpo.PolicyMPNNvGRPO.train_step` take
an optional per-step `reward_fn` (fall back to the shared `self.reward_fn`) — fully backward compatible.
Tests: `tests/unit/test_reward_registry.py` (bind strips inputs / keeps aggregation; `get_input_df`
injection incl. dict/list values + `${design.seq}`; full pipeline via a monkeypatched step, offline).
**Still TODO on this track (bigger changes):** the K-example gradient-accumulation update (§6.5 — the
current seam is one example/step, which already keeps the GRPO group homogeneous); a `cfg.dataset`
training config + an end-to-end smoke run; unfreezing the encoder / all-trainable optimizer (§4.1),
which arrives with the conditioned policy module.

### 6.7 Dataset assembly — antibody preprocessing (`data/preprocess/`)
Offline, run once to build a training JSONL; distinct from the runtime data layer (§6.6), which
only *consumes* it. Antibody examples are assembled combinatorially from PINDER (no backbone
generation — CDR structure is masked, §4.1):
```
pools:  antigens {seq, structure (chain T only), epitope residues, cached MSA}   (PINDER targets, §5.1)
        frameworks {native chothia backbone (CDRs intact), framework seq, MSA, native cdr_spans}
  for each example:
    sample (antigen, framework)                      # freely combinatorial (independent frames, §4.1)
    two SEPARATE files: framework pdb (design chain -> A / A,B), antigen pdb (chain -> T)  # §6.1, step 3.5
    carry the framework's NATIVE cdr_spans (positional into the design chain) as-is  # region to gap
    attach cdr_length_ranges (from native CDR length distributions)   # cdr_length_dists.py
    remap epitope_residues to the relabeled antigen chain T (PDB seqid.num, §4.1)
    emit JSONL row: {id, task, reward: antibody_interface, structure: <framework pdb>,
                     antigen_structure: <antigen pdb>, design_chain: "A" (Fab: ["A","B"]),
                     params: {antigen_sequence, antigen_msa_dir, framework_msa_dir,
                     cdr_spans, cdr_length_ranges, epitope_residues}}
```
**Length sampling + CDR gapping moved to rollout** (step 3.5, user 2026-07-05): the offline job no
longer masks backbones or shifts spans — it emits the *native* framework PDB plus native `cdr_spans`
(the region to excise) and `cdr_length_ranges`. At featurize time `sample_cdr_lengths` +
`apply_cdr_gaps` excise the native CDRs and splice sampled-length gap nodes, so each rollout sees a
freshly-lengthed gapped scaffold (length diversity is *sampled*, not enumerated as extra rows).
Framework and antigen MSAs are amortized (per framework / per antigen); only the per-design a3m
assembly (§5.4) is per-rollout. The same folder holds an `assemble_enzyme.py` / generic assembler
later — the JSONL schema (§6.1) is problem-agnostic.

**Precomputed metadata status (user 2026-07-05):**
- **Framework CDR indices — DONE.** `data/scaffolds/scaffold_chains.csv` carries `cdr_H1/H2/H3`
  (+`L1/2/3`) as half-open ranges into `vd_seq` for all 12,133 scaffold chains ([[msa-and-scaffolds]]).
  Assembly just maps these vd_seq ranges → framework-PDB design-chain positions to fill `cdr_spans`.
- **Target epitope residues — DONE (2026-07-05).** `data/precompute_epitopes.py` (gemmi 0.6.5, run
  with `cleo/.venv/bin/python`): for each antigen-target the epitope = antigen residues with a
  heavy-atom contact `< 5.0 Å` (`--cutoff`) to the partner chain, stored as **PDB `seqid.num`**
  (the convention `featurize_epitope` matches). Same pass **splits each antigen into a single-chain
  file relabelled chain `T`** (numbering preserved) at `~/pinder/antigens/{id}__ag{R|L}.pdb`, so the
  antigen loads on its own (two-file loading, §6.7 / step 3.5 — no shared frame). Manifest
  `data/pinder_epitopes.csv` (id, antigen_chain, partner_chain, split, antigen_file, n_antigen_res,
  n_epitope, epitope_residues, seq_hash, msa_dir) **joins the already-computed antigen MSAs** via
  `data/target_to_msa.csv` (`data/msa/msa_dirs/{seq_hash}/`, parallel to the scaffold framework MSAs).
  Resumable; round-trip validated (split file → `featurize_epitope`, epitope_mask sum == n_epitope,
  numbering matches). Remaining: thread scaffold `cdr_*` + these epitopes into the training JSONL.

### 6.8 Slice 2 — wiring the conditioner into the policy (Stage-1, pose-free)
The conditioner module (`data/epitope.py`) is built + unit-tested (§6.6). Slice 2 wires it into
the live policy for the **Stage-1 pose-free** build (§4.5). Every step below is gated so
`ConditioningConfig.enabled=False` remains a byte-identical stock-MPNN run (the M1 baseline). Ordered
by risk, each independently testable.

**What exists now (2026-07-10).** The pose-free Stage-1 policy is fully built + unit-tested (**235
passing**), gated so `conditioning.enabled=False` is byte-identical stock MPNN:
- **Dual-encoder conditioner** (`data/epitope.py`): separate epitope encoder (seq-injected message
  passing), CDR node-init (interp anchors, rel-pos, pooled epitope, CDR-identity, per-CDR position
  table, stem-gap geometry), encoder + decoder per-residue cross-attn, an iterated **CDR self-attn ↔
  CDR–epitope coupler** (`cdr_epitope_coupler`, step 5b — cross-chain paratope organization, stacked
  after the one-shot encoder cross-attn), coord-free CDR edges, learned-query attention-pool — each
  an independent ablation toggle. An epitope mask is **required**
  (whole-antigen conditioning is opt-in via `allow_whole_epitope`); unroutable CDRs error loudly.
- **Policy wiring** (`policy.py`): `attach_epitope` / `encode_initial_state` / `rollout` apply the
  three hooks; `train_framework_encoder` unfreezes the framework + epitope encoders (grad-tracked
  re-encode every PPO update). Per-item assert that the antigen chain is `T`.
- **Composing dataset** (§6.9), **N-chain oracle** (VHH + Fv, §4.2), **batch CDR-diversity reward**
  (`cdr_diversity.py`, §4.2), and **no-KL GRPO** (KL + grad-norm logged as diagnostics). Config:
  `config/design/antibody_composed.yaml`.

**Remaining: the step-7 e2e smoke** — run `antibody_composed.yaml` a few steps with
`conditioning.enabled=True` (shapes / grad-flow / Protenix reward) and an `enabled=False`
baseline-equivalence check. Kept for a human read-through of the code first.

<details><summary>Historical build log (slice-2 steps, ordered by risk) — kept for provenance.</summary>

4. **Wire the 3 conditioning hooks into encode + rollout (`policy.py`).** (a) `attach_epitope(example,
   feature_dict)` — encode the epitope once per example (frozen, `no_grad`) and stash `epi_per_res`
   `[1,M,H]` + `epi_mask` `[1,M]` into feature_dict for the hooks; no-op when disabled / no epitope.
   (b) `encode_initial_state` applies `condition_nodes` (node-init + encoder cross-attn) at CDR nodes
   after `model.encode`; identity when disabled (byte-identical M1). (c) `rollout` applies
   `decoder_cross_attn` per decode step, gated to CDR steps (`chain_mask_t==1`) via `torch.where`.
   Conditioner built **before** `get_optimizer` and its params added to the optimizer, so the decoder
   cross-attn (inside the grad-tracked rollout) trains now; the framework/epitope encoders stay frozen
   and the cached init_state is reused until step 6, so node-init / encoder cross-attn sit upstream of
   the detached leaf and get grads only once step 6 drops the cache. `PolicyMPNNvGRPO` inherits all
   (only overrides `train_step`).

3.5. **Gapped-framework data path (`data/gapping.py`, `data/mask.py`, `dataset.py`, `policy.py`).**
   Length is an explicit design variable (user 2026-07-05: sample per-step from ranges; gap at
   featurize time). (a) `Example.antigen_structure` — framework and antigen are **separate files**
   (no shared frame); the framework file carries no antigen chain, so chain T stays out of the main
   graph *for free* (subsumes step-4's design-chain-only parsing). (b) `mask.sample_cdr_lengths` —
   per-CDR length each rollout (`cdr_lengths` override > `cdr_length_ranges` sample > native span
   width). (c) `gapping.apply_cdr_gaps` — excise native CDR residues, splice N gap nodes
   (`chain_mask=1`), stem-copied backbone for finite tensors (never read under the surgery),
   contiguous renumber. (d) `featurize_example` builds the gap spec when conditioning +
   `coord_free_cdr_edges` are on and the example defines `cdr_spans`. **Supersedes the old §6.7
   "offline length sampling / masked backbones" plan** — gapping is now at rollout from native PDBs.

1. **Epitope encoder as a second ProteinMPNN (`policy.py.__init__`).** Instantiate a second MPNN,
   init from the same pretrained weights, hold it on the policy (`self.epi_encoder`). Pass it into
   `EpitopeConditioner(cfg.conditioning, epi_encoder=self.epi_encoder)`. Only when
   `cfg.conditioning.enabled`.
2. **`featurize_epitope` + a `mode` seam on `featurize_example`.** New helper builds the epitope
   feature dict (structure of chain `T` + its known sequence in `S`, for path-b injection).
   `featurize_example(example, mode="scaffold")` — `"scaffold"` = today's gapped-framework path;
   reserve `"complex"` for Stage 2 (predicted pose). **Forward-compat seam #2.**
3. **`ProteinFeatures` surgery (`model_utils.py:799`), the riskiest bit — behind a flag.** Add
   `coord_free_cdr_edges` support to `ProteinFeatures.forward`: (a) stem-borrowed Cα for the CDR
   rows before `_dist` (neighbor-selection only); (b) a learned `unknown_edge` param spliced into the
   RBF block for edges with a CDR endpoint, both directions, via a CDR-edge gate built from
   `chain_mask`. Keep the positional/chain edge features. Guard so the stock path is untouched when
   the flag is off. Unit-test in isolation (mask-gated edges change; framework-only edges identical
   to stock).
4. **Call the 3 hooks.** In `encode`/`train`: `encode_epitope(epi_fd)` once per example →
   `(epi_per_res, epi_mask)`; after the framework `encode`, `condition_nodes(h_V, chain_mask,
   epi_per_res, epi_mask)`; in `rollout` (~`model_utils.py:402`, before `W_out`),
   `decoder_cross_attn(h_V_t, epi_per_res, epi_mask)` gated by `chain_mask_t` (CDR steps only). All
   methods take an optional `pose=None` now (ignored) — **forward-compat seam #1**.
5. **New pose-free node signals (§4.1 menu).** Extend `CDRNodeInit`: CDR-identity embedding
   (H1/H2/H3, L1–L3 — from `design_regions`), stem-gap geometry `(dist, L/gap)` from the real stem
   Cα's, and swap pooled-mean → attention-pool in `EpitopeConditioner`.
6. **Optimizer / trainer (§4.1, §6.5).** Add conditioner + epitope-encoder params to the optimizer;
   **unfreeze the framework encoder** (all-trainable); drop the cached-`init_state` shortcut
   (re-encode every PPO update, since the encoder now trains).
7. **Config + smoke.** A `cfg.conditioning` group + a `cfg.dataset` training config; end-to-end
   smoke on the mini fixture with `enabled=True` (shapes/grad-flow) and a baseline-equivalence check
   with `enabled=False`.

**Deferred to Stage 2 (§4.5), not built here:** the `pose`-dependent toggles
(`pose_epitope_vectors`, `geometric_cross_attn`, `use_predicted_cdr_coords`), the `"complex"`
featurize mode body, the pose-confidence gate, and the predicted-structure ingest. The two seams
(#1 optional `pose` arg, #2 `featurize_example(mode=...)`) exist so these are additive, not a
rearchitecture.

</details>

### 6.9 Composing dataset — target × scaffold × CDR-lengths on the fly (built 2026-07-05)
A materialized JSONL manifest (§6.1) cannot hold the antibody training distribution: it is the full
**cross-product** of every antigen target × every framework scaffold × every CDR-length draw
(18,385 × 1,952 ≈ 3.6e7 rows before length diversity), static across a run. `ComposingDataset`
(`data/composer.py`, subclasses `DesignDataset`) instead holds two small pools and **composes one
`Example` per `sample()` call**, drawing the three axes independently:

- **target** — `data/pinder_epitopes_trainable.csv` (18,525 = band [5,35] ∩ has-MSA; train 18,385 /
  val 140), filtered by split. Carries the antigen file (chain `T`), epitope residues (chain-T
  seqid numbers), antigen MSA dir, and precomputed epitope net charge.
- **scaffold** — `data/scaffold_pool.csv` (1,952 reps = 395 VHH + 1,557 Fv), split into VHH/Fv lists;
  reps missing a required per-chain framework MSA are dropped at load. The **VHH/Fv mix is a tunable
  knob** `vhh_fraction` (user default 0.5), with fallback to the non-empty pool.
- **CDR lengths** — one `sample_cdr_lengths` draw per call, **pinned** into `params.cdr_lengths` so the
  featurizer's gapping (`sample_cdr_lengths` honors the fixed override) and the reward's per-chain
  sequence split use the *same* lengths.

The composed row is byte-compatible with the existing featurize/reward path and emits a **uniform
`params.design_chains`** = `[{length, framework_msa_dir, cdr_spans}, ...]` — one entry for VHH, two
for Fv (H-then-L; `H*` spans → H record via `msa_dir_H`, `L*` → L via `msa_dir_L`). Per-chain
`length = len(native VD) − Σ(native span widths) + Σ(sampled lengths)` is exactly the decoded segment
length the oracle's `_split_seq` (§4.2) expects, so VHH (1 chain) and Fv (2 chains) go through the same
generalized oracle path. Reward package = `config/design/reward/antibody_interface.yaml` (the single
package, requires `design_chains` instead of the scalar single-chain `framework_msa_dir`/`cdr_spans`).
`DesignDataset.bind_inputs` routes `${native.seq.T}` to the separate antigen file (`epitope_source`),
backward-compatible via fallback. Per rollout the loop logs `is_vhh` into the train-metrics CSV and
full provenance (`scaffold_id`, `kind`, `target_id`, `cdr_lengths`) to `{run_name}_provenance.csv`.

**Online + reproducible.** This is the runtime dataloading path — composition is fully online (no
materialized cross-product on disk). All three draws go through the composer's own `rng`, so seeding
it makes a run **deterministically replayable** while staying online; the provenance CSV records the
exact (scaffold, target, CDR-length) draw per step for post-hoc audit.
Tests: `test_composer.py` (11). Config: `config/design/antibody_composed.yaml`.

## 7. Milestones

| ID | Goal | Exit criteria |
|---|---|---|
| **M0** | Scaffolding | Repo skeleton, PINDER subset in `~/pinder` + epitope-extraction loader, `RewardOracle` interface + a cheap proxy impl, MPNN policy wrapper producing per-seq log-probs, single-target GRPO smoke test reproduces CLEO behavior. |
| **M1** | Multi-target GRPO baseline (no conditioning) | One policy trained across N targets; on held-out targets it beats base ProteinMPNN on the eval metrics (§8). Answers: does multi-task alone help? |
| **M2** | Epitope-conditioning prototype | Dual-encoder + cross-attention wired in (epitope encoder w/ sequence injection); overfit check on few targets, then ablation vs M1 (conditioning on vs off) on held-out targets. |
| **M3** | Zero-shot evaluation | Held-out-target proposals scored by Protenix v2; compare zero-shot conditioned model vs per-target CLEO vs base MPNN. |
| **M4** (stretch) | Protenix-in-the-loop | Replace/augment proxy reward with Protenix v2 (with cost mitigations) and re-run M1/M2. |

## 8. Evaluation & metrics

- **Held-out set:** the honest PINDER val set from §5.1 (236 antigen-targets / 231 antigen
  clusters, antigen-chain-cluster disjoint from train — not merely interface-disjoint). Use this
  for zero-shot generalization claims; SAbDab/SNAC-DB nanobody complexes for real-pair eval.
- **Oracle metrics:** Protenix ipTM / interface-PAE of proposed designs (held-out targets).
- **Native-sequence recovery** of CDRs on held-out complexes (sanity, not the real goal).
- **Proposal-distribution quality:** reward percentiles / pass@k under the oracle vs baselines.
- **Baselines:** (a) base ProteinMPNN, (b) per-target CLEO (the thing we want to match zero-shot),
  (c) M1 model (to isolate the value of conditioning in M2).
- **Generalization gap:** train-target vs held-out-target reward.

### 8.1 Validating the conditioning channel (does the epitope encoder actually steer MPNN?)
Two failure modes to separate, because they need different tests: **(i) dead channel** — the epitope
embedding doesn't affect the CDR logits (wiring/gradient bug, or RL zeroed the cross-attn); **(ii)
live but unused** — output changes with epitope but not in a way that helps the *intended* target (a
target-agnostic prior). Validate on a ladder: *is the wire connected → does it carry
epitope-specific info → does using it improve the right target → does it generalize.*

**Headline test — matched-vs-mismatched reward matrix (causal, uses the existing oracle).** Generate
CDRs conditioned on each of N epitopes, then fold every design against every epitope (N×N). A working
conditioner has a **dominant diagonal** — designs score best (ipTM / `epitope_overlap`) on the
epitope they were conditioned on. Quantify as **retrieval@1** (matched epitope = argmax over the row)
and the diagonal−off-diagonal reward gap. `epitope_overlap` is the sharpest channel (it catches
"binds the antigen at the wrong surface" — exactly the unconditioned failure).

**Strategy bake-off — overfit-speed on a 3×3 matrix (added 2026-07-04, runs before the full ladder).**
Before the expensive full ablation, rank the §4.1 conditioning strategies (pooled-only node-init /
+interpolated anchors / +encoder cross-attn / +decoder cross-attn / both / **+CDR–epitope coupler**
(step 5b, stacked on encoder cross-attn) / seq-injection path a-vs-b) by **how fast each overfits a
tiny conditioning-discriminative set** — convergence speed is a cheap, high-signal discriminator.
- **Design matrix = full cross-product (the crux).** 3 frameworks × 3 antigens/epitopes = **9 cells**,
  each framework paired with *every* antigen. This is deliberate: if framework identity alone (or
  antigen alone) predicts the answer, every strategy overfits without reading the epitope and the test
  is blind. The cross-product makes **framework + epitope jointly necessary** to fit all 9 — the same
  structure as the matched-vs-mismatched matrix, shrunk for speed.
- **Discriminator:** steps-to-threshold per strategy (+ the matched−mismatched gap curve, already the
  online guardrail), same seed/budget across strategies.
- **Run on the real Protenix oracle (decided 2026-07-04).** No surrogate tier — the overfit target
  is the actual interface reward, so a strategy's speed edge is measured against the objective we
  care about. Kept affordable by the small scale: only **9 cells**, a handful of strategies compared,
  and RL-fast sampling (c4/p20/e1, small K·B) — the same per-fold cost already benchmarked (~12s).
  Depends on the conditioned policy module (`data/epitope.py`) existing.

**The ladder:**
- **Rung 0 — plumbing (pre-RL unit test).** *Forward-pass sensitivity:* perturb/swap the epitope
  input and measure the change in CDR logits (per-position AA-distribution KL); must be **nonzero
  even at init** or the channel isn't connected. *Gradient norms* on the epitope-encoder + cross-attn
  params: non-zero, non-vanishing.
- **Rung 1 — information.** Conditioned CDR node embeddings vary far more across epitopes than across
  seeds for one epitope; a frozen **linear probe** can recover epitope identity / true contact
  residues from them (info is present, decoupled from whether the decoder uses it).
- **Rung 2 — causality/specificity.** The matched-vs-mismatched matrix above.
- **Rung 3 — ablation (necessity).** *Zero/shuffle* the epitope at inference → reward must drop.
  *Toggle the cross-attn sites* (encoder-only / decoder-only / both / pooled-only / none, §4.1) and
  compare `epitope_overlap`; the none/pooled-only run is the control for what per-residue cross-attn
  buys. Add a **coupler on-vs-off** pair on top of encoder cross-attn (step 5b) — isolates what the
  iterated cross-chain CDR self-attn ↔ epitope organization buys over the one-shot pass.
- **Rung 4 — interpretability.** Cross-attn weights: does the H3 apex put mass on the *true* epitope
  interface residues (from the native complex) vs. diffuse/uniform?
- **Rung 5 — generalization (the real goal).** Matched > mismatched on **held-out** epitopes, and
  conditioned-M2 must **beat the M1 unconditioned baseline** on epitope-specific reward. If
  M2 ≈ M1, conditioning adds nothing regardless of the internal probes.

**Sequencing.** Rung 0 runs the moment the module exists (no training). Rungs 1–2 come from the M2
**overfit-a-few-epitopes** check — if the model can't overfit *distinct* high-reward CDRs to distinct
epitopes, the channel is broken and we stop before scaling. Rungs 3–5 come with the M2-vs-M1 ablation.
**Online guardrail (wire in now):** log **`matched_reward − mismatched_reward`** (and matched vs.
mismatched `epitope_overlap`) every training step — one extra fold per design against a shuffled
epitope (we're folding anyway), turning "is the conditioner alive?" into a live curve. This gap
going and staying **> 0** is the **M2 exit criterion**.

### Note — inference-time pose (resolved by the dual-encoder)
The dual-encoder (§4.1) encodes epitope and framework in **independent frames**, so we no longer
need a posed framework↔epitope complex at inference — this was the main reason for choosing it
over complex-graph nodes. The remaining **CDR-loop backbone** concern is also resolved (§4.1,
2026-07-03): CDR structure is masked and supplied by the epitope embedding + coordinate-free anchor
edges, so no CDR backbone is needed at all.

## 9. Open questions / parameters to pin down

- [x] **Epitope sequence injection** — RESOLVED (2026-07-04, §4.1): **path (b)** — `W_s(S)` into the
      encoder node features before the message-passing layers (not decoder teacher-forcing).
- [x] **CDR-loop backbone** — RESOLVED (2026-07-03, §4.1): don't build it. Mask the CDR structure
      on a native PINDER framework and let the epitope embedding supply CDR node context; sample CDR
      lengths from native distributions.
- [x] **Masked-CDR k-NN graph entry** — RESOLVED (2026-07-04, §4.1): **stem-borrowed k-NN** for
      connectivity + **learned "unknown-distance" edge embedding** on CDR-incident edges (both
      directions); pseudo-coordinate re-anchoring rejected as OOD.
- [ ] Reward shaping: weights for `ipTM`, `iPAE`, `epitope_overlap`; dense-proxy + sparse-Protenix schedule.
- [ ] Proxy oracle for the loop (ESM-IF logP vs distilled surrogate vs Rosetta ddG).
- [ ] Policy init: vanilla ProteinMPNN vs `antibody_mpnn` vs `msa_mpnn`/`fused_mpnn` weights.
- [ ] PINDER subset definition: redundancy/interface-quality filters; how many targets for M1/M2.
- [ ] Is multi-target GRPO *alone* (M1) enough to beat per-target CLEO, or is conditioning required?
- [ ] GRPO hyperparams: group size G, KL β, learning rate, batch composition across targets.
- [ ] Epitope definition: contact-distance cutoff; whole antigen vs epitope patch as conditioning.
- [ ] Locate/install Protenix v2 + weights (not confirmed on the system).
- [ ] **Oracle serving — persistent workers vs. per-call subprocess.** `protenix_from_df` currently
      spawns **one subprocess per GPU per reward call** (env init + model load + kernel JIT on top of
      the actual fold; warm kernels are cached but weights reload each call). Since the oracle is the
      RL loop's cost center (called every step), consider hosting Protenix (and Boltz) as
      **persistent warm model workers** — e.g. a **Ray Serve / Ray actor** pool: weights loaded once
      per GPU, the trainer submits fold requests over RPC, and requests **batch across the rollout**
      (K·B designs/step) instead of one JSON-per-GPU. Maps cleanly onto the existing boundary — the
      subprocess call *is* already a cross-env RPC into Protenix's isolated venv, so a Ray worker just
      makes that boundary persistent (the worker runs *in* the Protenix env; the trainer stays in
      cleo's). **Adopt only if it beats current throughput** — benchmark warm-worker latency +
      cross-process serialization vs. the measured ~12–19s/fold subprocess path before committing.
      Also unlocks a shared oracle pool across concurrent runs.

## 10. Repo layout (follows CLEO project conventions)

We follow CLEO's project pattern: a **copy of the cleo repo** with project-specific modules added
under `src/cleo/design/utils/`, plus `configs/`, `structures/`, run/log dirs. We work on the
`antibody-rl` branch of the cleo clone. Key extension points map onto existing CLEO machinery:

- **Reward** = CLEO's `UniversalReward` (config-driven, normalized weighted sum of metric *steps*).
  Our `ipTM / iPAE / epitope_overlap` are just reward-aggregation metrics; the oracle and
  epitope-overlap are added as **reward steps** (like `boltz_from_df` / `af3_from_df`).
- **Oracle** = a new `protenix_from_df` step alongside the existing `boltz_from_df` (Boltz is
  already a dep → usable as the cheaper interim AF3-style oracle) and `af3_from_df`.
- **Policy/GRPO** = `cleo.design.utils.grpo.PolicyMPNNvGRPO`; this is where the deepest divergence
  lives — CLEO trains on a **single backbone PDB** per run, we need **multi-target batching** and
  the **dual-encoder + cross-attention** conditioning.

```
~/projects/antibody_rl/
  SPEC.md
  cleo/                         # clone of github.com/0merle0/cleo @ branch antibody-rl
    src/cleo/design/utils/
      grpo.py                   # extend: multi-target batching; KL + grad-norm logged (not penalized)
      reward.py                 # UniversalReward (reuse)
      protenix_oracle.py        # NEW: protenix_from_df (N-chain: VHH + Fv) + epitope_overlap metric
      cdr_diversity.py          # NEW: batch CDR-diversity reward step (structural + sequence)
    src/cleo/design/data/
      epitope.py                # NEW: dual-encoder conditioner (epitope encoder+seq injection, cross-attn)
      composer.py               # NEW: on-the-fly target × scaffold × CDR-length composing dataset
  configs/                      # M1/M2 training, sample, evaluate configs (Hydra)
  structures/                   # canonical VHH framework scaffold(s)
  cleo_runs/  logs/  train.submit   # outputs + SLURM (gpu-train, l40)
~/pinder/                       # filtered PINDER subset (staging)
```

## 11. Immediate next steps
(M0 scaffolding + the data pipeline are done: PINDER subset in `~/pinder`, epitope precompute,
uniref MSAs, VHH/Fv scaffold library + VD backbones, Protenix v2 oracle (N-chain), dataset-driven
harness, epitope conditioner wired (steps 1–6), composing dataset (§6.9), and the training config.)

1. **Slice-2 step 7 e2e smoke** — run `antibody_composed.yaml` on a few steps with `conditioning.
   enabled=True` (shapes / grad-flow / Protenix reward), plus an `enabled=False` baseline-equivalence
   check (byte-identical stock MPNN, M1 baseline). *(Held for a human read-through of steps 5–6 first.)*
2. **M1 → M2 runs** — multi-target GRPO baseline (conditioning off) vs conditioned, ablations per §8.1.
