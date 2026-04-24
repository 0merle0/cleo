"""Shared fixtures for the CLEO test suite."""
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def tiny_sequence_df():
    """A minimal DataFrame mirroring what the reward pipeline passes to step_fns."""
    return pd.DataFrame(
        {
            "name": ["s0", "s1", "s2"],
            "sequence": ["ACDE", "ACDF", "AGDE"],
        }
    )


@pytest.fixture
def cfg_ns():
    """Factory producing an attribute-access config object (step_fns expect cfg.field)."""
    return SimpleNamespace


@pytest.fixture
def tiny_fragment_dictionary():
    """A 3-region fragment dictionary with 2 entries per region."""
    return {
        1: [("1.a", "AA"), ("1.b", "AB")],
        2: [("2.a", "CC"), ("2.b", "CD")],
        3: [("3.a", "EE"), ("3.b", "EF")],
    }
