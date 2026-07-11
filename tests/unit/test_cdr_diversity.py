"""Unit tests for the batch CDR-diversity reward (SPEC §8)."""
import json

import numpy as np
import pandas as pd
import pytest

from cleo.design.utils.cdr_diversity import (
    _as_design_chains,
    _iter_cdrs,
    _kabsch_rmsd,
    _seq_distance,
    cdr_diversity_from_df,
)


def test_seq_distance_bounds():
    assert _seq_distance("ABCDE", "ABCDE") == 0.0
    assert _seq_distance("ABCDE", "VWXYZ") == 1.0
    assert _seq_distance("ABCDE", "ABCDZ") == pytest.approx(0.2)
    # length gap counts as difference over the longer length
    assert _seq_distance("ABC", "ABCDE") == pytest.approx(2 / 5)
    assert _seq_distance("", "") == 0.0


def test_kabsch_rmsd_invariant_to_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(8, 3))
    # rotate + translate P -> same shape, RMSD ~ 0
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Q = P @ R.T + np.array([5.0, -2.0, 1.0])
    assert _kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-6)
    # genuinely different shapes -> positive
    assert _kabsch_rmsd(P, rng.normal(size=(8, 3))) > 0.1


def test_as_design_chains_vhh_and_fv():
    vhh = _as_design_chains({"design_chains": [{"length": 120, "cdr_spans": {"H1": [25, 32]}}]}, "A" * 120)
    assert vhh == [{"length": 120, "cdr_spans": {"H1": [25, 32]}}]
    # JSON string form + fallback to scalar cdr_spans
    fv = _as_design_chains(
        {"design_chains": json.dumps([{"length": 5, "cdr_spans": {"H1": [1, 3]}},
                                      {"length": 4, "cdr_spans": {"L1": [0, 2]}}])},
        "AAAAA" + "CCCC",
    )
    assert [c["length"] for c in fv] == [5, 4]
    legacy = _as_design_chains({"cdr_spans": {"H3": [2, 4]}}, "ABCDEF")
    assert legacy == [{"length": 6, "cdr_spans": {"H3": [2, 4]}}]


def test_iter_cdrs_chain_local_slicing():
    chains = [{"length": 5, "cdr_spans": {"H1": [1, 3]}}, {"length": 4, "cdr_spans": {"L1": [0, 2]}}]
    got = list(_iter_cdrs("HHHHH" + "LLLL", chains))
    assert ("A", "H1", 1, 3, "HH") in got
    assert ("B", "L1", 0, 2, "LL") in got


def _row(name, seq, spans):
    return {"name": name, "sequence": seq, "design_chains": [{"length": len(seq), "cdr_spans": spans}]}


def test_sequence_diversity_rewards_difference():
    # three designs; design c has a distinct H1, so it should score highest, identical a/b lowest.
    df = pd.DataFrame([
        _row("a", "XXAAAXX", {"H1": [2, 5]}),
        _row("b", "XXAAAXX", {"H1": [2, 5]}),
        _row("c", "XXVWYXX", {"H1": [2, 5]}),
    ])
    out = cdr_diversity_from_df(df, {}, step_name="div")
    s = out.set_index("name")["div_cdr_seq_diversity"]
    assert s["a"] == pytest.approx(s["b"])
    assert s["c"] > s["a"]
    # no structure column -> structural diversity is 0.0 (finite, never NaN — aggregator fails fast on NaN)
    assert (out["div_cdr_struct_diversity"] == 0.0).all()
    assert np.isfinite(out["div_cdr_struct_diversity"]).all()


def test_singleton_type_has_zero_diversity():
    df = pd.DataFrame([_row("solo", "XXAAAXX", {"H1": [2, 5]})])
    out = cdr_diversity_from_df(df, {}, step_name="div")
    assert out["div_cdr_seq_diversity"].iloc[0] == 0.0
