import sys, os
from omegaconf import OmegaConf
import hydra
from policy_utils import PolicyMPNN
from PPO import PPOPolicy
from GRPO import GRPO_singleprompt


@hydra.main(version_base=None, config_path="../config")
def train_policy(cfg):
    """
    Train a policy network using either vanilla REINFORCE or PPO algorithm.
    The algorithm is determined by the 'algorithm' setting in the config.
    
    Args:
        cfg: Configuration object from hydra
    """
    # Initialize the appropriate policy based on the algorithm specified in config
    if cfg.get('algorithm').lower() == 'ppo':
        print(f"Using PPO algorithm for training")
        policy = PPOPolicy(cfg)
    elif cfg.get('algorithm').lower() == 'grpo':
        print(f"Using GRPO algorithm for training")
        policy = GRPO_singleprompt(cfg)
    else:
        print(f"Using REINFORCE algorithm for training")
        policy = PolicyMPNN(cfg)

    # Train the policy
    policy.train()


if __name__ == "__main__":
    train_policy()