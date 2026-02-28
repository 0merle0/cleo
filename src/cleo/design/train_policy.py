"""
Entrypoint for fine-tuning ProteinMPNN with reinforcement learning.

Supports two algorithms selected via the ``algorithm`` config key:
  - ``vanillapg``: Vanilla REINFORCE with a running-mean baseline.
  - ``grpo``: Group Relative Policy Optimization (clipped surrogate
    objective with optional KL penalty to a frozen reference model).

Usage:
    python -m cleo.design.train_policy --config-name denovo_petase
"""

import hydra


@hydra.main(version_base=None, config_path="../../../config/design")
def train_policy(cfg):
    """Launch a training run using the algorithm specified in *cfg*."""

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