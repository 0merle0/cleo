import sys, os
from omegaconf import OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./config")
def evaluate_policy(cfg):
    """
    Evaluate a trained policy network.
    Uses PolicyMPNN for inference regardless of training algorithm.
    
    Args:
        cfg: Configuration object from hydra
    """
    
    # Use PolicyMPNN for evaluation - only need inference, not training logic
    from policy_utils import PolicyMPNN
    policy = PolicyMPNN(cfg, eval_mode=True)

    # Evaluate the policy
    _, _ = policy.evaluate()


if __name__ == "__main__":
    evaluate_policy() 