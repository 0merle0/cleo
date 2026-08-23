"""
Entrypoint for fine-tuning ProteinMPNN with reinforcement learning.

Supports two algorithms selected via the ``algorithm`` config key:
  - ``vanillapg``: Vanilla REINFORCE with a running-mean baseline.
  - ``grpo``: Group Relative Policy Optimization (clipped surrogate
    objective with optional KL penalty to a frozen reference model).

Usage:
    python -m cleo.design.train_policy --config-name denovo_petase
"""

import random
from pathlib import Path

import hydra
import numpy as np
import torch

_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/design")


@hydra.main(version_base=None, config_path=_CONFIG_DIR)
def train_policy(cfg):
    """Launch a training run using the algorithm specified in *cfg*."""

    # Optional explicit seed. Default is None, which leaves the process
    # unseeded and reproduces the historical behaviour exactly -- every run
    # reported before E21 was drawn this way.
    #
    # Seeding matters more than it looks. E19 re-ran an E18 config in a fresh
    # process and the *control* moved 21 points of pass rate, which is larger
    # than most effects the selection experiments were built to detect. An
    # unseeded run therefore cannot be replicated even in principle, and a
    # seed-variance arm cannot be labelled without this. cudnn is left in its
    # default nondeterministic mode: full determinism costs throughput and is
    # not what is wanted here, which is *labelled* draws from the run-to-run
    # distribution rather than one frozen trajectory.
    seed = cfg.get("seed", None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seeded run: seed={seed}")

    algorithm = cfg.get("algorithm", "").lower()

    if algorithm == "vanillapg":
        print("Using vanilla REINFORCE algorithm for training")
        from cleo.design.utils.policy import PolicyMPNN
        policy = PolicyMPNN(cfg)

    elif algorithm == "grpo":
        print("Using GRPO algorithm for training")
        from cleo.design.utils.grpo import PolicyMPNNvGRPO
        policy = PolicyMPNNvGRPO(cfg)

    else:
        raise ValueError(
            f"Unsupported algorithm: {cfg.get('algorithm')}. "
            "Supported algorithms are 'grpo' and 'vanillaPG'."
        )

    policy.train()


if __name__ == "__main__":
    train_policy()