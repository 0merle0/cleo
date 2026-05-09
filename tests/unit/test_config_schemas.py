"""Schema tests for shipped Hydra configs.

Loads every YAML under `config/` and checks that required keys exist and
that referenced `target_fn` strings resolve to importable callables. This
catches typos and stale references at test-time instead of crashing minutes
into a real run.
"""
import importlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load(path):
    # Don't resolve interpolations — many use ${hydra:runtime.cwd} which is
    # only defined inside a hydra run. We just want the raw structure.
    return OmegaConf.load(path)


def _resolve_target(target_str):
    module_path, _, attr = target_str.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


class TestDesignTrainingConfig:
    cfg = _load(CONFIG_DIR / "design" / "denovo_petase.yaml")

    @pytest.mark.parametrize(
        "key",
        [
            "run_name",
            "output_dir",
            "pdb",
            "checkpoint_path",
            "algorithm",
            "batch_size",
            "N_steps",
            "lr",
            "model_type",
            "temperature",
            "fixed_residues",
            "reward",
        ],
    )
    def test_top_level_keys_present(self, key):
        assert key in self.cfg

    def test_algorithm_value(self):
        assert self.cfg.algorithm in {"grpo", "vanillapg"}

    def test_model_type_value(self):
        assert self.cfg.model_type in {"protein_mpnn", "ligand_mpnn"}

    def test_reward_has_target(self):
        assert "_target_" in self.cfg.reward
        assert _resolve_target(self.cfg.reward._target_) is not None

    def test_reward_steps_have_required_keys(self):
        for step in self.cfg.reward.steps:
            assert "name" in step
            assert "target_fn" in step
            assert "cfg" in step

    def test_reward_step_target_fns_importable(self):
        for step in self.cfg.reward.steps:
            assert _resolve_target(step.target_fn) is not None, step.target_fn

    def test_reward_aggregation_entries_well_formed(self):
        for entry in self.cfg.reward.reward_aggregation:
            for key in ("metric", "lower_bound", "upper_bound", "weight", "mode"):
                assert key in entry, f"missing {key} in {entry}"
            assert entry.mode in {"max", "min"}


class TestDesignSampleConfig:
    cfg = _load(CONFIG_DIR / "design" / "sample.yaml")

    @pytest.mark.parametrize(
        "key", ["output_dir", "output_name", "checkpoints", "num_batches", "fragment_bounds"]
    )
    def test_keys_present(self, key):
        assert key in self.cfg

    def test_fragment_bounds_are_pairs(self):
        for fb in self.cfg.fragment_bounds:
            assert len(fb) == 2
            assert fb[0] <= fb[1]


class TestDesignResampleConfig:
    cfg = _load(CONFIG_DIR / "design" / "resample_fragments.yaml")

    @pytest.mark.parametrize(
        "key",
        ["fragment_dict_path", "num_sequences", "connector", "output_dir", "output_name"],
    )
    def test_keys_present(self, key):
        assert key in self.cfg


class TestDesignEvaluateConfig:
    cfg = _load(CONFIG_DIR / "design" / "evaluate.yaml")

    @pytest.mark.parametrize("key", ["input_fasta", "output_dir", "output_name", "steps"])
    def test_keys_present(self, key):
        assert key in self.cfg

    def test_step_target_fns_importable(self):
        for step in self.cfg.steps:
            assert _resolve_target(step.target_fn) is not None, step.target_fn


class TestDesignDnaConfig:
    cfg = _load(CONFIG_DIR / "design" / "dna_fragment_design.yaml")

    @pytest.mark.parametrize("key", ["vector_json_path", "vector"])
    def test_keys_present(self, key):
        assert key in self.cfg


class TestOptimizeBaseSurrogateConfig:
    cfg = _load(CONFIG_DIR / "optimize" / "base_surrogate.yaml")

    @pytest.mark.parametrize("key", ["run_name", "trainer", "data", "checkpointer"])
    def test_keys_present(self, key):
        assert key in self.cfg

    def test_data_has_dataset_cfg(self):
        assert "dataset_cfg" in self.cfg.data
        assert "label_col" in self.cfg.data.dataset_cfg
        assert "seq_col" in self.cfg.data.dataset_cfg


class TestOptimizeAcqfConfig:
    cfg = _load(CONFIG_DIR / "optimize" / "momi_acqf_opt.yaml")

    @pytest.mark.parametrize("key", ["run_name", "outdir", "opt_loop", "acqf"])
    def test_keys_present(self, key):
        assert key in self.cfg


class TestOptimizePredFastaConfig:
    cfg = _load(CONFIG_DIR / "optimize" / "pred_fasta.yaml")

    @pytest.mark.parametrize(
        "key", ["run_name", "model_base_path", "outdir", "fasta_path", "batch_size", "ckpt_name"]
    )
    def test_keys_present(self, key):
        assert key in self.cfg


class TestAllConfigsLoad:
    """Sanity check: every YAML under config/ parses without error."""

    @pytest.mark.parametrize(
        "yaml_path",
        sorted(CONFIG_DIR.rglob("*.yaml")),
        ids=lambda p: str(p.relative_to(CONFIG_DIR)),
    )
    def test_loads(self, yaml_path):
        cfg = _load(yaml_path)
        assert cfg is not None
