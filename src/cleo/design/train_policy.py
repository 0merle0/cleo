import sys, os
from omegaconf import OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../../../config/design")
def train_policy(cfg):
    """
    Train a policy network using either vanilla REINFORCE or PPO algorithm.
    The algorithm is determined by the 'algorithm' setting in the config.
    
    Args:
        cfg: Configuration object from hydra
    """
    
    if cfg.get('algorithm').lower() == 'vanillapg':
        print(f"Using vanilla REINFORCE algorithm for training")
        from cleo.design.utils.policy import PolicyMPNN
        policy = PolicyMPNN(cfg)

    elif cfg.get('algorithm').lower() == 'grpo':
        print(f"Using GRPO algorithm for training")
        from cleo.design.utils.grpo import PolicyMPNNvGRPO
        policy = PolicyMPNNvGRPO(cfg)
    
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.get('algorithm')}. Supported algorithms are 'grpo' and 'vanillaPG'.")

    # Train the policy
    policy.train()
    
    # Close wandb
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    train_policy()