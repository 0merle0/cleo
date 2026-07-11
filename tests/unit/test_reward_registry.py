"""Unit tests for the reward registry seam (SPEC 6.4).

Verifies that RewardRegistry.bind(example) produces a BoundReward that (a) strips
the `inputs` metadata from steps, (b) seeds resolved per-example inputs as df
columns via get_input_df, and (c) drives the full UniversalReward pipeline so a
step function receives those columns — all offline (the oracle step is
monkeypatched, no Protenix).
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cleo.design.data import DESIGN_SEQ, DesignDataset
from cleo.design.data.registry import BoundReward, RewardRegistry

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_complex.pdb"
CDR = {"H1": [2, 5], "H2": [8, 11]}


def _echo_reward_dir(tmp_path):
    d = tmp_path / "reward"
    d.mkdir(exist_ok=True)
    (d / "echo.yaml").write_text(
        "requires: [antigen_sequence, cdr_spans, epitope_residues]\n"
        "steps:\n"
        "  - name: echo\n"
        "    target_fn: cleo.design.utils.protenix_oracle.protenix_from_df\n"
        "    cfg: {some_static: 1}\n"
        "    inputs:\n"
        "      antigen_sequence: ${native.seq.T}\n"
        "      cdr_spans: ${row.params.cdr_spans}\n"
        "      epitope_residues: ${row.params.epitope_residues}\n"
        "      design_echo: ${design.seq}\n"
        "reward_aggregation:\n"
        "  - {metric: score, mode: max, lower_bound: 0.0, upper_bound: 1.0, weight: 1.0}\n"
    )
    return d


def _dataset(tmp_path):
    row = {
        "id": "t0", "task": "nanobody_design", "reward": "echo",
        "structure": str(FIXTURE), "design_chain": "A",
        "design_regions": ["cdr_H1", "cdr_H2"],
        "params": {"cdr_spans": CDR, "epitope_residues": [3, 4, 5]},
    }
    p = tmp_path / "ds.jsonl"
    p.write_text(json.dumps(row) + "\n")
    return DesignDataset.load(str(p), str(_echo_reward_dir(tmp_path)))


def test_bind_strips_inputs_and_keeps_aggregation(tmp_path):
    ds = _dataset(tmp_path)
    reg = RewardRegistry(ds, str(tmp_path / "out"), "run")
    reward = reg.bind(ds.sample())
    assert isinstance(reward, BoundReward)
    assert "inputs" not in reward.steps[0]            # inputs -> df columns, not step cfg
    assert reward.steps[0].name == "echo"
    assert reward.reward_aggregation[0].metric == "score"


def test_get_input_df_injects_resolved_columns(tmp_path):
    ds = _dataset(tmp_path)
    reward = RewardRegistry(ds, str(tmp_path / "out"), "run").bind(ds.sample())
    df = reward.get_input_df(["ACDE", "AGDE", "AAAA"])
    assert list(df["antigen_sequence"]) == ["NLAFALSELD"] * 3     # native.seq.T broadcast
    assert df["cdr_spans"].tolist() == [CDR, CDR, CDR]            # dict intact per row
    assert df["epitope_residues"].tolist() == [[3, 4, 5]] * 3
    assert list(df["design_echo"]) == ["ACDE", "AGDE", "AAAA"]    # ${design.seq} -> sequence


def test_full_reward_pipeline_step_sees_injected_columns(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)
    reward = RewardRegistry(ds, str(tmp_path / "out"), "run").bind(ds.sample())

    seen = {}

    def dummy_step(df, cfg, step_name="step"):
        seen["cols"] = set(df.columns)
        seen["antigen"] = list(df["antigen_sequence"])
        seen["static"] = cfg.some_static           # static step cfg survives
        seen["rundir"] = cfg.rundir                # reserved keys set by UniversalReward
        df["score"] = [0.5] * len(df)              # metric consumed by reward_aggregation
        return df

    # UniversalReward resolves each step's target_fn via get_method; swap in the dummy.
    monkeypatch.setattr("cleo.design.utils.reward.get_method", lambda path: dummy_step)

    B, L = 3, 4
    policy_output = {"S": torch.randint(0, 20, (B, L))}
    # real feature dicts carry both; chain_mask marks the designable positions (all, here — single chain)
    feature_dict = {
        "chain_labels": torch.zeros(1, L, dtype=torch.long),
        "chain_mask": torch.ones(1, L, dtype=torch.long),
    }
    rewards, to_log = reward(step=0, policy_output=policy_output, feature_dict=feature_dict, device="cpu")

    assert rewards.shape == (B,)
    assert torch.allclose(rewards, torch.full((B,), 0.5), atol=1e-3)
    assert {"antigen_sequence", "cdr_spans", "epitope_residues", "design_echo"} <= seen["cols"]
    assert seen["antigen"] == ["NLAFALSELD"] * B
    assert seen["static"] == 1
    assert "step_0000" in seen["rundir"]
