import sys, os
from omegaconf import OmegaConf
import hydra
from policy_utils import PolicyMPNN


@hydra.main(version_base=None, config_path="../config")
def train_policy(cfg):
    """Train surrogate model."""

    policy_mpnn = PolicyMPNN(cfg)

    policy_mpnn.train()


if __name__ == "__main__":
    train_policy()