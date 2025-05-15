import sys, os
from omegaconf import OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./config")
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
        from PPO import PPOPolicy
        policy = PPOPolicy(cfg)

    elif cfg.get('algorithm').lower() == 'grpo':
        print(f"Using GRPO algorithm for training")
        from GRPO import GRPO_singleprompt
        policy = GRPO_singleprompt(cfg)
    
    elif cfg.get('algorithm').lower() == 'vanillapg':
        print(f"Using vanilla REINFORCE algorithm for training")
        from policy_utils import PolicyMPNN
        policy = PolicyMPNN(cfg)
    
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.get('algorithm')}. Supported algorithms are 'ppo', 'grpo', and 'vanillaPG'.")

    # Train the policy
    policy.train()


if __name__ == "__main__":
    train_policy()