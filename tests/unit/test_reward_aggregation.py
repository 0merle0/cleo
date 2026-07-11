"""Unit tests for UniversalReward metric aggregation — esp. fail-fast on non-finite metrics."""
import types

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cleo.design.utils.reward import UniversalReward


def _metric(name, mode="max", lb=0.0, ub=1.0, w=1.0):
    return types.SimpleNamespace(metric=name, mode=mode, lower_bound=lb, upper_bound=ub, weight=w)


def test_aggregate_rewards_weighted_sum():
    r = UniversalReward(reward_aggregation=[_metric("a"), _metric("b", mode="min", ub=10.0)])
    df = pd.DataFrame({"name": ["s0", "s1"], "a": [1.0, 0.0], "b": [0.0, 10.0]})
    out = r._aggregate_rewards(df)
    assert out.shape == (2,)
    assert out[0] > out[1]                          # s0 wins both metrics


def test_aggregate_rewards_raises_on_nan_naming_offender():
    r = UniversalReward(reward_aggregation=[_metric("a")])
    df = pd.DataFrame({"name": ["good", "bad"], "a": [0.5, float("nan")]})
    with pytest.raises(ValueError, match="non-finite") as e:
        r._aggregate_rewards(df)
    assert "bad" in str(e.value)                     # names the offending design
    assert "'a'" in str(e.value)                     # names the metric


def test_aggregate_rewards_raises_on_inf():
    r = UniversalReward(reward_aggregation=[_metric("a")])
    df = pd.DataFrame({"name": ["x"], "a": [float("inf")]})
    with pytest.raises(ValueError, match="non-finite"):
        r._aggregate_rewards(df)
