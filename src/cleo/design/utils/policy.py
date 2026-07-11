"""
Vanilla REINFORCE (policy gradient) fine-tuning of ProteinMPNN.

Provides :class:`PolicyMPNN`, which wraps a ProteinMPNN model with an
autoregressive rollout that keeps gradients through the decoder so that
the REINFORCE loss can flow back. A running-mean baseline is subtracted
from the reward to reduce variance.

Amino-acid encoding constants used across the codebase (``alphabet``,
``restype_STRtoINT``, etc.) are defined here and should be imported from
this module rather than duplicated elsewhere.
"""

import contextlib
import json
import os
import torch
import numpy as np
from tqdm import tqdm
import time
import hydra
from omegaconf import OmegaConf

from cleo.design.protein_mpnn_utils.data_utils import featurize, parse_PDB
from cleo.design.protein_mpnn_utils.model_utils import ProteinMPNN
from cleo.design.data.epitope import ConditioningConfig, EpitopeConditioner, ordered_cdr_type_ids
from cleo.design.data.gapping import apply_cdr_gaps
from cleo.design.data.mask import sample_cdr_lengths
from cleo.design.data.dataset import scan_chain_ids


PROTEIN_MPNN_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "protein_mpnn_utils")
PROTEIN_MPNN_CKPT_PATH = os.path.join(PROTEIN_MPNN_UTILS_DIR, "vanilla_protein_mpnn_weights.pt")
LIGAND_MPNN_CKPT_PATH = os.path.join(PROTEIN_MPNN_UTILS_DIR, "ligand_protein_mpnn_weights.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ProteinMPNN amino-acid encoding tables
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
alphabet = list(restype_STRtoINT)


class PolicyMPNN:
    """ProteinMPNN wrapper for RL fine-tuning with vanilla REINFORCE."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = DEVICE
        # Parsed early so load_mpnn_model can thread coord_free_cdr_edges into the feature extractor.
        self.conditioning_cfg = ConditioningConfig.from_dict(cfg.get("conditioning"))
        self.run_name = cfg.run_name
        self.output_dir = os.path.join(cfg.output_dir, cfg.run_name)

        os.makedirs(self.output_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(self.output_dir, f"{self.run_name}_config.yaml"))

        self.reward_history = [0]

        self.model = self.load_mpnn_model()
        self.model.to(self.device)

        self.ligand_mpnn_use_atom_context = 1
        self.ligand_mpnn_cutoff_for_score = 8.0

        # Epitope conditioning (SPEC 4.1 / 6.8). Master `enabled=False` => the conditioner is an
        # exact no-op with zero params (byte-identical M1 baseline), so this is safe to always
        # construct. When enabled, build a SECOND ProteinMPNN whose encoder embeds the epitope
        # (init from the pretrained base weights, independently trainable) and hand it to the
        # conditioner. Built BEFORE the optimizer so its trainable hook params (step 4) are
        # picked up by get_optimizer. (self.conditioning_cfg was parsed at the top of __init__.)
        if self.conditioning_cfg.enabled:
            self.epi_encoder = self._build_epitope_encoder()
            self.conditioner = EpitopeConditioner(
                self.conditioning_cfg, epi_encoder=self.epi_encoder
            ).to(self.device)
            print(f"Epitope conditioning ENABLED: {self.conditioning_cfg}")
        else:
            self.epi_encoder = None
            self.conditioner = EpitopeConditioner(self.conditioning_cfg)  # no-op, no params

        # Step 6 (SPEC 6.8): when set, the framework encoder + epitope encoder are unfrozen and the
        # initial state is re-encoded (grad-tracked) every PPO update instead of being cached as a
        # detached leaf — so node-init / encoder cross-attn / epitope-encoder params get gradients.
        # Only meaningful with conditioning on; the default (False) keeps the decoder-only step-4 path.
        self.train_framework_encoder = self.conditioning_cfg.enabled and bool(
            cfg.get("train_framework_encoder", False)
        )

        self.optimizer = self.get_optimizer()

        # Single-target reward (cfg.reward). Optional in dataset-driven mode, where the
        # reward is built per-example by the registry (SPEC 6.4).
        self.reward_fn = hydra.utils.instantiate(cfg.reward) if cfg.get("reward") else None

        # Dataset-driven multi-example harness (SPEC 6). When cfg.dataset is set, sample
        # one example per step and bind its reward package; otherwise use cfg.pdb + reward_fn.
        self.dataset = None
        self.reward_registry = None
        if cfg.get("dataset"):
            from cleo.design.data.registry import RewardRegistry

            comp = cfg.dataset.get("composer")
            if comp is not None:
                # On-the-fly composer: sample target x scaffold x CDR-lengths per step
                # (no materialized cross-product JSONL; SPEC 6.9).
                import random as _random

                from cleo.design.data.composer import ComposingDataset

                ranges = comp.get("cdr_length_ranges", None)
                # Data seed (SPEC 6.9): seeding the composer's own rng makes the target/scaffold/
                # CDR-length draws deterministically replayable while staying online. None => unseeded.
                data_seed = comp.get("seed", None)
                self.dataset = ComposingDataset.load(
                    comp.targets_csv,
                    comp.scaffold_pool_csv,
                    comp.reward,
                    cfg.dataset.reward_dir,
                    split=comp.get("split", None),
                    vhh_fraction=comp.get("vhh_fraction", 0.5),
                    cdr_length_ranges=OmegaConf.to_container(ranges) if ranges is not None else None,
                    rng=_random.Random(data_seed) if data_seed is not None else None,
                )
            else:
                from cleo.design.data.dataset import DesignDataset

                self.dataset = DesignDataset.load(
                    cfg.dataset.jsonl,
                    cfg.dataset.reward_dir,
                    structures_root=cfg.dataset.get("structures_root", ""),
                )
                print(f"Loaded dataset with {len(self.dataset)} examples from {cfg.dataset.jsonl}")
            self.reward_registry = RewardRegistry(self.dataset, cfg.output_dir, cfg.run_name)

        self.checkpoint_every_n_steps = self.cfg.checkpoint_every_n_steps
        self.best_seen_reward = 0
        self.step_at_best_seen_reward = 0


    def load_mpnn_model(self):
        """Load and return a ProteinMPNN model initialised from checkpoint weights."""

        model_type = self.cfg.model_type

        if model_type == "protein_mpnn":
            self.atom_context_num = 1
            k_neighbors = 48
            self.ligand_mpnn_use_side_chain_context = 0
            ckpt_path = PROTEIN_MPNN_CKPT_PATH
        elif model_type == "ligand_mpnn":
            self.atom_context_num = 25
            k_neighbors = 32
            self.ligand_mpnn_use_side_chain_context = 0
            ckpt_path = LIGAND_MPNN_CKPT_PATH
        else:
            raise ValueError("Invalid model type specified. Choose 'ligand_mpnn' or 'protein_mpnn'.")

        if self.cfg.get('checkpoint_path'):
            print(f"Loading training checkpoint from {self.cfg.checkpoint_path}")
            ckpt_path = self.cfg.checkpoint_path
        else:
            print(f"Using default MPNN checkpoint: {ckpt_path}")

        model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=self.atom_context_num,
            model_type=model_type,
            ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context,
            coord_free_cdr_edges=(
                self.conditioning_cfg.enabled and self.conditioning_cfg.coord_free_cdr_edges
            ),
        )

        is_default_checkpoint = ckpt_path in [PROTEIN_MPNN_CKPT_PATH, LIGAND_MPNN_CKPT_PATH]
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=is_default_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(self.device)

    def _build_epitope_encoder(self):
        """A second ProteinMPNN whose encoder embeds the epitope (SPEC 4.1).

        Initialised from the **pretrained base** ProteinMPNN weights (not the policy's training
        checkpoint — the epitope encoder is an independent module, not a fine-tuned decoder), and
        independently trainable. Only its encoder side is used (``encode_epitope``, path-b
        sequence injection); the decoder weights load but stay unused.
        """
        model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            device=self.device,
            atom_context_num=1,
            model_type="protein_mpnn",
            ligand_mpnn_use_side_chain_context=0,
        )
        ckpt = torch.load(PROTEIN_MPNN_CKPT_PATH, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(self.device)

    def get_optimizer(self):
        """Return an Adam optimizer over the trainable parameters.

        Two regimes (SPEC 6.8):

        - **Step 4 (default)** — freeze the framework encoder; train the decoder (``decoder_layers``
          + ``W_out``) plus the epitope-conditioning hook params. The cached init-state is a detached
          leaf, so node-init / encoder cross-attn sit upstream of it and get no gradient yet; only the
          decoder cross-attn (inside the grad-tracked rollout) trains.
        - **Step 6 (``train_framework_encoder``)** — everything trainable: the whole framework MPNN
          (encoder + decoder), the epitope encoder, and every conditioner hook. The init-state is
          re-encoded grad-tracked each update (see ``train_step``), so the upstream hooks train too.
        """
        if self.train_framework_encoder:
            for param in self.model.parameters():
                param.requires_grad = True
            trainable = list(self.model.parameters())
            if self.conditioning_cfg.enabled:
                trainable += list(self.conditioner.parameters())  # incl. the epitope encoder submodule
        else:
            for name, param in self.model.named_parameters():
                if "decoder_layers" not in name and "W_out" not in name:
                    param.requires_grad = False
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if self.conditioning_cfg.enabled:
                trainable += list(self.conditioner.parameters())

        optimizer = torch.optim.Adam(trainable, lr=self.cfg.lr)
        return optimizer

    def _featurize(self, pdb, fixed_residues_fn, gap=None):
        """Parse a structure and build the feature dict consumed by :meth:`rollout`.

        ``fixed_residues_fn(chain_letters, R_idx, icodes) -> list[str]`` returns the
        residue tokens to hold fixed (NOT designed). ``featurize_pdb`` supplies the
        config-level ``fixed_residues``; ``featurize_example`` computes the CDR-mask
        complement from the dataset example.

        ``gap`` (SPEC 6.8 step 3.5): when given — ``{"design_chain", "cdr_spans", "cdr_lengths"}``
        — the native CDRs are excised and sampled-length gap nodes are spliced in
        (:func:`cleo.design.data.gapping.apply_cdr_gaps`), which sets ``chain_mask`` itself
        (gap nodes designable); ``fixed_residues_fn`` is then unused. Requires
        ``coord_free_cdr_edges`` so the gap nodes' placeholder coordinates are never read.
        """

        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
            pdb,
            device=self.device,
            atom_context_num=self.atom_context_num,
            chains="",
            parse_all_atoms=self.ligand_mpnn_use_side_chain_context,
        )

        if gap is not None:
            protein_dict = apply_cdr_gaps(
                protein_dict, gap["design_chain"], gap["cdr_spans"], gap["cdr_lengths"]
            )  # sets chain_mask (gap = designable) + side_chain_mask
        else:
            R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
            chain_letters_list = list(protein_dict["chain_letters"])
            encoded_residues = [
                str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
                for i in range(len(R_idx_list))
            ]

            chain_mask = torch.tensor(
                np.array([True for _ in protein_dict["chain_letters"]], dtype=np.int32),
                device=self.device,
            )
            protein_dict["chain_mask"] = chain_mask

            fixed_residues = fixed_residues_fn(chain_letters_list, R_idx_list, icodes)
            fixed_positions = torch.tensor(
                [int(item not in fixed_residues) for item in encoded_residues],
                device=self.device,
            )
            if fixed_residues:
                protein_dict["chain_mask"] = protein_dict["chain_mask"] * fixed_positions

            protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        omit_AA_list = self.cfg.omit_AA if self.cfg.omit_AA is not None else []
        omit_AA = torch.tensor(
            np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
            device=self.device,
        )

        n_res = protein_dict["chain_mask"].shape[0]            # post-gapping length (gap != native)
        bias_AA_per_residue = torch.zeros([n_res, 21], device=self.device, dtype=torch.float32)
        omit_AA_per_residue = torch.zeros([n_res, 21], device=self.device, dtype=torch.float32)

        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=self.ligand_mpnn_cutoff_for_score,
            use_atom_context=self.ligand_mpnn_use_atom_context,
            number_of_ligand_atoms=self.atom_context_num,
            model_type=self.cfg.model_type,
        )
        feature_dict["batch_size"] = self.cfg.batch_size
        B, L, _, _ = feature_dict["X"].shape

        feature_dict["temperature"] = self.cfg.temperature
        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        feature_dict["bias"] = (
            (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )

        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["randn"] = torch.randn(
            [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
            device=self.device,
        )

        return feature_dict

    def featurize_pdb(self, pdb):
        """Featurize a single PDB, holding ``cfg.fixed_residues`` fixed (single-target path)."""
        return self._featurize(pdb, lambda cl, ri, ic: self.cfg.fixed_residues.split())

    def featurize_example(self, example, mode="scaffold"):
        """Featurize a dataset example, masking to its ``params.cdr_spans`` (SPEC 6.4).

        ``fixed_residues`` = the CDR-mask complement computed by the data layer from
        the example's ``design_chain`` + ``cdr_spans`` against this exact enumeration,
        so the mask lines up 1:1 with the tokens ``_featurize`` builds.

        ``mode`` (SPEC 6.8 forward-compat seam #2): ``"scaffold"`` = the Stage-1 gapped-framework
        path (design chain only, CDRs masked); ``"complex"`` is reserved for Stage-2 pose
        conditioning (framework + predicted CDR + epitope in one frame) and is not built yet.
        """
        if mode != "scaffold":
            raise NotImplementedError(
                f"featurize_example(mode={mode!r}) is a Stage-2 seam (SPEC 4.5/6.8); only "
                f"'scaffold' is implemented."
            )
        # Stage-1 gapped-framework path (SPEC 6.8 step 3.5): excise native CDRs and splice in
        # sampled-length gap nodes. Only when the coord-free surgery is on (so the gap nodes'
        # placeholder coordinates are never read) and the example actually defines CDR spans.
        gap = None
        if (
            self.conditioning_cfg.enabled
            and self.conditioning_cfg.coord_free_cdr_edges
            and example.cdr_spans
        ):
            gap = {
                "design_chain": example.design_chain,
                "cdr_spans": example.cdr_spans,
                "cdr_lengths": sample_cdr_lengths(example.params),  # fresh per step (length diversity)
            }
        feature_dict = self._featurize(
            example.structure,
            lambda cl, ri, ic: self.dataset.fixed_residues(example, cl, ri, ic).split(),
            gap=gap,
        )
        # CDR-type ids per segment (position order), for the step-5 CDR-identity node signal.
        # Ignored downstream unless conditioning.node_init_cdr_identity is on.
        if self.conditioning_cfg.enabled and example.cdr_spans:
            feature_dict["cdr_ids"] = ordered_cdr_type_ids(example.design_chain, example.cdr_spans)
        return feature_dict

    def featurize_epitope(self, example):
        """Build the epitope feature dict consumed by ``EpitopeConditioner.encode_epitope`` (SPEC 4.1).

        Parses the **whole antigen chain T** (full structural context for message passing) with its
        **known sequence** in ``S`` (path-b injection). Emits two masks:
        - ``mask`` — all valid antigen residues; the encoder message-passes over this (whole fold).
        - ``epitope_mask`` — the ``epitope_residues`` patch (matched by PDB residue number, the
          reward's convention); ``encode_epitope`` returns THIS for ``pool_epitope`` + cross-attn,
          so the CDR conditions on the epitope specifically, not the whole antigen (§4.1, 2026-07-04).
        """
        # Invariant (SPEC §6.1, LOCKED 2026-07-04): the antigen/target chain is always 'T'. Assert
        # it per item rather than trusting the manifest, so a mislabeled antigen fails loudly here.
        antigen_chains = scan_chain_ids(example.epitope_source)
        if "T" not in antigen_chains:
            raise ValueError(
                f"example {example.id}: antigen source {example.epitope_source} has no chain 'T' "
                f"(found {sorted(antigen_chains)}); the antigen/target chain must be 'T'")

        protein_dict, _bb, _oa, icodes, _wa = parse_PDB(
            example.epitope_source,
            device=self.device,
            atom_context_num=self.atom_context_num,
            chains=["T"],
            parse_all_atoms=self.ligand_mpnn_use_side_chain_context,
        )
        if protein_dict["R_idx"].shape[0] == 0:
            raise ValueError(f"example {example.id}: no antigen chain 'T' found in {example.epitope_source}")

        # The epitope is fixed context (never designed); featurize() still requires these keys.
        n_res = protein_dict["R_idx"].shape[0]
        protein_dict["chain_mask"] = torch.ones(n_res, device=self.device)
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=self.ligand_mpnn_cutoff_for_score,
            use_atom_context=self.ligand_mpnn_use_atom_context,
            number_of_ligand_atoms=self.atom_context_num,
            model_type="protein_mpnn",
        )

        R_idx_list = [int(r) for r in protein_dict["R_idx"].cpu().numpy()]
        epi_set = {int(r) for r in example.epitope_residues}
        epitope_mask = torch.tensor(
            [[1.0 if r in epi_set else 0.0 for r in R_idx_list]],
            device=self.device,
            dtype=torch.float32,
        )
        if float(epitope_mask.sum()) == 0.0:
            raise ValueError(
                f"example {example.id}: none of epitope_residues={sorted(epi_set)} matched chain-T "
                f"residue numbers (are they PDB seqid.num, not positional indices?)"
            )
        feature_dict["epitope_mask"] = epitope_mask
        return feature_dict

    def attach_epitope(self, example, feature_dict):
        """Encode the epitope once and stash its per-residue embeddings + patch mask into
        ``feature_dict`` for the conditioning hooks (SPEC 4.1, slice-2 step 4).

        No-op (returns ``feature_dict`` unchanged) when conditioning is disabled or the
        example carries no ``epitope_residues``.

        Always stashes the parsed epitope feature dict ``epi_fd`` so ``encode_initial_state``
        can (re-)encode it. In the **step-4** regime (framework frozen) the embeddings are
        precomputed here once under ``no_grad`` and cached; the downstream ``decoder_cross_attn``
        still trains because ``epi_per_res`` enters it as a constant inside the grad-tracked
        rollout. In the **step-6** regime (``train_framework_encoder``) they are left for
        ``encode_initial_state`` to recompute grad-tracked each update, so the epitope encoder trains.

        Keys written: ``epi_fd`` (parsed antigen feature dict), and — step-4 only — ``epi_per_res``
        ``[1, M, H]`` + ``epi_mask`` ``[1, M]`` (the epitope patch from ``encode_epitope``).
        """
        if not (self.conditioning_cfg.enabled and example.epitope_residues):
            return feature_dict
        feature_dict["epi_fd"] = self.featurize_epitope(example)
        if not self.train_framework_encoder:
            with torch.no_grad():
                epi_per_res, epi_mask = self.conditioner.encode_epitope(feature_dict["epi_fd"])
            feature_dict["epi_per_res"] = epi_per_res
            feature_dict["epi_mask"] = epi_mask
        return feature_dict

    def encode_initial_state(self, feature_dict, grad=False):
        """Encode the framework structure and apply the post-encode framework-node conditioning
        (node-init + encoder cross-attn) at CDR positions (SPEC 4.1 hook 1+2).

        ``condition_nodes`` is an identity when conditioning is disabled, so the M1 baseline
        path is byte-identical. ``grad=False`` (default) runs under ``no_grad`` — the step-4
        cached-init-state path and every experience-collection rollout. ``grad=True`` (step 6)
        keeps the graph so the framework + epitope encoders train; it (re)computes the epitope
        embeddings grad-tracked from ``epi_fd``. When enabled but no embeddings are cached yet
        (step-6 experience collection), they are computed here under the active context so the
        experience and update rollouts see the *same* conditioning.
        """
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            if self.conditioning_cfg.enabled and feature_dict.get("epi_fd") is not None:
                if grad or feature_dict.get("epi_per_res") is None:
                    epi_per_res, epi_mask = self.conditioner.encode_epitope(feature_dict["epi_fd"])
                    feature_dict["epi_per_res"] = epi_per_res
                    feature_dict["epi_mask"] = epi_mask
            h_V, h_E, E_idx = self.model.encode(feature_dict)
            if self.conditioning_cfg.enabled:
                h_V = self.conditioner.condition_nodes(
                    h_V,
                    feature_dict["chain_mask"],
                    feature_dict.get("epi_per_res"),
                    feature_dict.get("epi_mask"),
                    cdr_ids=feature_dict.get("cdr_ids"),
                    X=feature_dict.get("X"),
                )
        return h_V, h_E, E_idx

    def gather_nodes(self, nodes, neighbor_idx):
        """
        Copy from proteinMPNN Utils
        """
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
        return neighbor_features

    def cat_neighbors_nodes(self, h_nodes, h_neighbors, E_idx):
        """
        Copy from proteinMPNN Utils
        """
        h_nodes = self.gather_nodes(h_nodes, E_idx)
        h_nn = torch.cat([h_neighbors, h_nodes], -1)
        return h_nn

    def rollout(
        self,
        feature_dict,
        h_V,
        h_E,
        E_idx,
        decoding_order=None,
        sampled_actions=None,
        model=None,
    ):
        """Autoregressive decoding with gradients through the decoder.

        Adapted from the fused ProteinMPNN decoding loop. Gradients flow
        through the decoder layers so the RL loss can update decoder weights.
        ``scatter`` is used instead of in-place indexing to keep the
        computation graph intact.

        Args:
            feature_dict: Feature dictionary from :meth:`featurize_pdb`.
            h_V: Node features from encoder [1, L, C].
            h_E: Edge features from encoder [1, L, K, C].
            E_idx: Edge indices [1, L, K].
            decoding_order: Optional pre-defined decoding order [B, L].
            sampled_actions: Optional pre-sampled sequence integers [B, L].
                When provided, log-probs are computed for these actions
                instead of sampling new ones (used by GRPO for importance
                ratio computation).
            model: Model to decode with (defaults to ``self.model``).

        Returns:
            dict with keys ``S``, ``sampling_probs``, ``log_probs``,
            ``decoding_order``, ``state_features``, ``chain_mask``.
        """

        if model is None:
            model = self.model

        if sampled_actions is None:
            B_decoder = feature_dict["batch_size"]
            S_true = feature_dict["S"]
            mask = feature_dict["mask"]
            chain_mask = feature_dict["chain_mask"]
            bias = feature_dict["bias"]
            randn = feature_dict["randn"]
            temperature = feature_dict["temperature"]
        else:
            B_decoder = sampled_actions.shape[0]
            S_true = sampled_actions
            mask = feature_dict["mask"][:1].repeat(B_decoder, 1)
            chain_mask = feature_dict["chain_mask"][:1].repeat(B_decoder, 1)
            bias = feature_dict["bias"][:1].repeat(B_decoder, 1, 1)
            randn = feature_dict["randn"][:1].repeat(B_decoder, 1)
            temperature = feature_dict["temperature"]

        B, L = S_true.shape

        if decoding_order is None:
            decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))

        E_idx = E_idx.repeat(B_decoder, 1, 1)
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(L, L, device=self.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([B, L, 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        chain_mask = chain_mask.repeat(B_decoder, 1)
        mask = mask.repeat(B_decoder, 1)
        bias = bias.repeat(B_decoder, 1, 1)

        all_probs = torch.zeros((B_decoder, L, 20), device=self.device, dtype=torch.float32)
        all_log_probs = torch.zeros((B_decoder, L, 21), device=self.device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=self.device)
        S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=self.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=self.device) for _ in range(len(model.decoder_layers))]

        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(L):
            t = decoding_order[:, t_]
            chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]
            mask_t = torch.gather(mask, 1, t[:, None])[:, 0]
            bias_t = torch.gather(bias, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = self.cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
            mask_bw_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1]))

            for l, layer in enumerate(model.decoder_layers):
                h_ESV_decoder_t = self.cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t

                # Uses scatter (not in-place indexing) so gradients flow through the decoder
                h_V_stack[l+1] = h_V_stack[l+1].scatter(
                    1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=None),
                )

            h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]

            # Hook 3 (SPEC 4.1): per-step decoder cross-attn to epitope residues, gated to the
            # CDR decode steps (chain_mask_t == 1). Identity elsewhere and when conditioning /
            # decoder_cross_attn is off. The delta's params sit in the grad-tracked path, so they
            # train even while the encoder stays frozen (step 4).
            if self.conditioning_cfg.enabled and feature_dict.get("epi_per_res") is not None:
                h_V_t_cond = self.conditioner.decoder_cross_attn(
                    h_V_t, feature_dict["epi_per_res"], feature_dict.get("epi_mask"),
                )
                h_V_t = torch.where(chain_mask_t[:, None] > 0.5, h_V_t_cond, h_V_t)

            logits = model.W_out(h_V_t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            probs = torch.nn.functional.softmax((logits.detach() + bias_t) / temperature, dim=-1)
            probs_sample = probs[:, :20] / torch.sum(probs[:, :20], dim=-1, keepdim=True)

            if sampled_actions is None:
                S_t = torch.multinomial(probs_sample, 1)[:, 0]
            else:
                S_t = sampled_actions[torch.arange(B), t]

            all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 20), (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float())
            all_log_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21), (chain_mask_t[:, None, None] * log_probs[:, None, :]).float())

            with torch.no_grad():
                S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
                S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
                h_S.scatter_(1, t[:, None, None].repeat(1, 1, h_S.shape[-1]), model.W_s(S_t)[:, None, :])
                S.scatter_(1, t[:, None], S_t[:, None])

        output_dict = {
            "S": S,
            "sampling_probs": all_probs,
            "log_probs": all_log_probs,
            "decoding_order": decoding_order,
            "state_features": h_V_stack[-1].detach(),
            "chain_mask": chain_mask,
        }

        return output_dict
    
    def get_sequences(self, policy_output, chain_mask=None):
        """Decode integer sequences from a rollout output into amino-acid strings."""
        sampled_sequences = policy_output["S"]
        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        sequences = []
        for i in range(sampled_sequences.shape[0]):
            seq_str = "".join([alphabet[int(s)] for s in sampled_sequences[i]])
            sequences.append(seq_str)
        return sequences

    def train_step(self, step, init_state, feature_dict, reward_fn=None):
        """Run one REINFORCE update: rollout -> reward -> policy gradient."""

        reward_fn = reward_fn if reward_fn is not None else self.reward_fn
        to_log = {}
        self.optimizer.zero_grad()

        # init_state=None (step 6): re-encode grad-tracked so the encoders train. Else use the
        # cached detached leaf and flip on grad (decoder-only path).
        if init_state is None:
            h_V_in, h_E_in, E_idx_in = self.encode_initial_state(feature_dict, grad=True)
        else:
            h_V_in, h_E_in, E_idx_in = init_state
            h_V_in.requires_grad = True
            h_E_in.requires_grad = True

        out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in)

        seq_mask = torch.nn.functional.one_hot(out["S"], num_classes=len(alphabet)).float()
        batched_log_probs = (out["log_probs"] * seq_mask).sum(dim=(-1))

        batched_reward, metrics = reward_fn(step, out, feature_dict, self.device)
        to_log.update(metrics)

        baseline = torch.tensor(self.reward_history, dtype=torch.float32, device=self.device).mean()
        self.reward_history.append(batched_reward.mean().item())
        baseline_subtracted_reward = batched_reward - baseline

        if baseline_subtracted_reward.shape != batched_log_probs.shape:
            loss = -(batched_log_probs.sum(dim=-1) * baseline_subtracted_reward).mean()
        else:
            loss = -(batched_log_probs * baseline_subtracted_reward).sum(dim=-1).mean()

        loss.backward()
        self.optimizer.step()

        to_log["loss"] = loss.detach().cpu().item()
        to_log["reward"] = batched_reward.mean().cpu().item()

        return to_log

    def train(self):
        """Run the main training loop for ``N_steps`` iterations."""

        start_time = time.time()
        for step in tqdm(range(self.cfg.N_steps), desc="Training"):
            self.model.train()

            # Dataset-driven: sample one example, featurize+mask it, bind its reward
            # package. Single-PDB path (cfg.pdb + shared reward_fn) is the fallback.
            if self.reward_registry is not None:
                example = self.dataset.sample()
                feature_dict = self.featurize_example(example)
                feature_dict = self.attach_epitope(example, feature_dict)  # epitope hooks (step 4)
                reward_fn = self.reward_registry.bind(example)
            else:
                feature_dict = self.featurize_pdb(self.cfg.pdb)
                reward_fn = self.reward_fn

            # Step 6: re-encode grad-tracked each update inside train_step (init_state=None).
            # Otherwise cache a detached init-state leaf once (step-4 / stock decoder-only path).
            if self.train_framework_encoder:
                init_state = None
            else:
                h_V, h_E, E_idx = self.encode_initial_state(feature_dict)
                init_state = (h_V.clone(), h_E.clone(), E_idx.clone())

            to_log = self.train_step(step, init_state, feature_dict, reward_fn)
            if self.reward_registry is not None:
                to_log.update(self._epitope_log_fields(example))
                self._log_provenance(step, example)

            runtime = time.time() - start_time
            self.log_metrics(step, runtime, to_log)

            if step > 0 and step % self.checkpoint_every_n_steps == 0:
                self.checkpoint_model(step, to_log)

        print("Training complete.")
        print(f"Best reward seen: {self.best_seen_reward:.4f} at step {self.step_at_best_seen_reward}")

    def _epitope_log_fields(self, example):
        """Per-rollout epitope metadata for the training CSV: epitope size and net charge,
        so post-hoc analysis can surface any reward/size or reward/charge trends (SPEC 6.7).
        Both keys are emitted on every dataset-driven step (0.0 / NaN when an example carries
        no epitope) so the CSV columns stay stable across the run."""
        nc = example.epitope_net_charge
        fields = {
            "epitope_size": float(example.n_epitope),
            "epitope_net_charge": float(nc) if nc is not None else float("nan"),
        }
        # Composer rollouts also carry the scaffold modality (VHH vs Fv). Emit it as a float
        # so post-hoc analysis can split reward by modality; absent (=> no column) for the
        # JSONL path, keeping CSV columns stable within any single run.
        kind = example.params.get("kind")
        if kind is not None:
            fields["is_vhh"] = 1.0 if kind == "vhh" else 0.0
        return fields

    def _log_provenance(self, step, example):
        """Append this rollout's composed provenance (scaffold, target, sampled CDR lengths)
        to ``{run_name}_provenance.csv``. Only the composer stamps ``scaffold_id`` into params;
        the JSONL / single-PDB paths are no-ops. String-valued, so it lives outside the
        float-only training-metrics CSV."""
        p = example.params
        if "scaffold_id" not in p:
            return
        log_path = os.path.join(self.output_dir, f"{self.run_name}_provenance.csv")
        cdr_lengths = json.dumps(p.get("cdr_lengths", {}), separators=(",", ":"))
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("step,scaffold_id,kind,target_id,cdr_lengths\n")
        with open(log_path, "a") as f:
            f.write(f'{step},{p["scaffold_id"]},{p["kind"]},{p["target_id"]},"{cdr_lengths}"\n')

    def log_metrics(self, step, runtime, to_log):
        """Append one row to the CSV training log."""
        metrics_to_log = [k for k, v in to_log.items() if isinstance(v, float)]
        log_path = os.path.join(self.output_dir, f"{self.run_name}_train_metrics.csv")
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("step,runtime," + ",".join(metrics_to_log) + '\n')
        with open(log_path, 'a') as f:
            f.write(f"{step},{runtime:.4f}," + ",".join([f"{to_log[m]:.4f}" for m in metrics_to_log]) + '\n')
    
    def checkpoint_model(self, step, to_log):
        """Save ``_last``, ``_step_NNNN``, and (if improved) ``_best`` checkpoints."""
        curr_reward = to_log["reward"]
        ckpt = {
            "config": dict(self.cfg),
            "step": step,
            "reward": curr_reward,
            "model_state_dict": self.model.state_dict(),
        }
        # The conditioner carries trainable params (hooks + epitope encoder); persist them so a
        # conditioned run is recoverable. Absent/no-op for the M1 baseline.
        if self.conditioning_cfg.enabled:
            ckpt["conditioner_state_dict"] = self.conditioner.state_dict()

        torch.save(ckpt, os.path.join(self.output_dir, f"{self.run_name}_last.pt"))
        torch.save(ckpt, os.path.join(self.output_dir, f"{self.run_name}_step_{step:04}.pt"))

        if curr_reward > self.best_seen_reward:
            self.best_seen_reward = curr_reward
            self.step_at_best_seen_reward = step
            torch.save(ckpt, os.path.join(self.output_dir, f"{self.run_name}_best.pt"))

    def sample_from_policy(self, num_batches):
        """Sample sequences from the current policy without gradients."""

        with torch.no_grad():
            output_list = []
            for _ in tqdm(range(num_batches)):
                feature_dict = self.featurize_pdb(self.cfg.pdb)
                h_V, h_E, E_idx = self.encode_initial_state(feature_dict)
                out = self.rollout(feature_dict, h_V, h_E, E_idx)
                output_list.append(out)

        chain_mask = feature_dict["chain_labels"] == 0
        chain_mask = chain_mask[0]

        sequences = []
        for out in output_list:
            sequences.extend(self.get_sequences(out, chain_mask))

        return sequences