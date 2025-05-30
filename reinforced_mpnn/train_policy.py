import sys, os
from omegaconf import OmegaConf
import hydra
import wandb
from policy_utils import PolicyMPNN
from PPO import PPOPolicy
from GRPO import GRPO_singleprompt


@hydra.main(version_base=None, config_path="./config")
def train_policy(cfg):
    """
    Train a policy network using either vanilla REINFORCE or PPO algorithm.
    The algorithm is determined by the 'algorithm' setting in the config.
    
    Args:
        cfg: Configuration object from hydra
    """
    # Initialize wandb
    wandb.init(
        project="policy_mpnn",
        entity="bakerlab",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.run_name,
        dir=cfg.output_dir,
        # Mode can be set through config (online, offline, disabled)
        mode="online" if not hasattr(cfg, "wandb_mode") else cfg.wandb_mode
    )
    
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
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    train_policy()