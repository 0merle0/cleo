import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb as pdb_lib
from policy_utils import PolicyMPNN, alphabet
import wandb

class ExperienceBuffer:
    """Storage for PPO experience collection"""
    def __init__(self, device):
        self.device = device
        self.clear()
        
    def clear(self):
        self.log_probs = []
        self.rewards = []
        self.sequences = []
        self.decoding_order = []
        self.states = []
        self.dones = []  # Added to track episode terminations
        
    def add(self, log_prob, reward, sequence, decoding_order, state, done=False):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.sequences.append(sequence)
        self.decoding_order.append(decoding_order)
        self.states.append(state)
        self.dones.append(torch.tensor(float(done), device=self.device))
        
    def get_batch(self):
        return {
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.stack(self.rewards),
            'sequences': self.sequences,
            'decoding_order': self.decoding_order,
            'states': self.states,
            'dones': torch.stack(self.dones) if self.dones else None
        }


class GRPO_singleprompt(PolicyMPNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # PPO specific parameters
        self.clip_param = cfg.get('clip_param', 0.2)
        self.value_coef = cfg.get('value_coef', 0.5)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.max_grad_norm = cfg.get('max_grad_norm', 0.5)
        self.ppo_epochs = cfg.get('ppo_epochs', 4)
        self.mini_batch_size = cfg.get('mini_batch_size', 4)
        self.target_kl = cfg.get('target_kl', 0.015)
        self.gamma = cfg.get('gamma', 0.99)
        self.gae_lambda = cfg.get('gae_lambda', 0.95)
        self.norm_adv = cfg.get('norm_adv', True)
        self.clip_vloss = cfg.get('clip_vloss', True)
        # Experience buffer
        self.buffer = ExperienceBuffer(self.device)
        
    def get_optimizer(self):
        """Define optimizer over decoder params with gradient clipping"""
        # Freeze encoder weights
        for name, param in self.model.named_parameters():
            if "decoder_layers" in name or "W_out" in name:
                continue
            else:
                param.requires_grad = False

        # Only provide optimizer with unfrozen params
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr
        )
        return optimizer
    
    def collect_experience(self, step, init_state, feature_dict):
        """Collect experience for PPO using the current policy
        
        Processes a single batch of sequences and stores each sequence
        as a separate experience in the buffer.
        """
        self.buffer.clear()
        
        h_V_in, h_E_in, E_idx_in = init_state
        
        # Make fresh copies before passing to rollout
        h_V = h_V_in.clone().detach().requires_grad_(True)
        h_E = h_E_in.clone().detach().requires_grad_(True)
        
        # Get sequence and probabilities
        out = self.rollout(feature_dict, h_V, h_E, E_idx_in)
        
        # Get batched outputs
        S = out["S"]  # Shape: [batch_size, L]
        log_probs = out["log_probs"]  # Shape: [batch_size, L, n_tokens]
        decoding_order = out["decoding_order"] # Shape: [batch_size, L]
        
        # Calculate log probs for each sequence
        seq_mask = torch.nn.functional.one_hot(S, num_classes=len(alphabet)).float()
        batch_log_probs = (log_probs * seq_mask).sum(dim=(-1,-2))  # Shape: [batch_size]
        
        # Normalize by sequence length
        seq_length = S.shape[1]  # L (sequence length)
        normalized_log_probs = batch_log_probs 
        
        # Get rewards - no gradient tracking needed
        with torch.no_grad():
            batch_rewards, reward_metrics = self.reward_fn(step, out, feature_dict, self.device)
            
            # Log reward metrics to wandb
            if wandb.run and reward_metrics:
                self.aux_metrics = {f"reward_metrics/{k}": v for k, v in reward_metrics.items()}

        
        # Store each sequence in the batch as a separate experience
        batch_size = S.shape[0]

        # pdb_lib.set_trace()
        for i in range(batch_size):
            self.buffer.add(
                normalized_log_probs[i].detach(),
                batch_rewards[i].detach(),
                S[i].detach() if isinstance(S, torch.Tensor) else S[i],
                decoding_order[i].detach(),
                (h_V.detach(), h_E.detach()),  # Keep h_V sliced to maintain batch dim
                done=True  # Each sequence is a complete "episode"
            )
        
        # Compute returns and advantages
        # pdb_lib.set_trace()
        returns, advantages = self.compute_returns_and_advantages()
        
        return self.buffer.get_batch(), returns, advantages
        
    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages for protein sequences
        """
        # Get rewards and values from buffer
        rewards = torch.stack(self.buffer.rewards) # Shape: [batch_size]
        returns = rewards  # For single-step environments, return = reward
        
        # Calculate advantages as normalized rewards
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Log reward distribution to wandb
        if wandb.run:
            # Note: Removed the wandb.log call here as it will be handled in train_step
            # This ensures consistent step counting
            self.reward_stats = {
                "rewards/mean": rewards.mean().item(),
                "rewards/std": rewards.std().item(),
                "rewards/min": rewards.min().item(),
                "rewards/max": rewards.max().item(),
            }

        return returns, advantages

    def update_policy(self, batch, returns, advantages, init_state, feature_dict):
        """Update policy and value networks using PPO"""
        # Get data from batch
        old_log_probs = batch['log_probs']
        rewards = batch['rewards']
        sequences = batch['sequences']
        states = batch['states']
        decoding_order = batch['decoding_order']
        
        # Metrics to track
        clipfracs = []
        approx_kls = []
        pg_losses = []
        vf_losses = []
        entropy_losses = []
        total_losses = []
        
        # Mini-batch updates
        batch_size = len(old_log_probs)
        # pdb_lib.set_trace()
        
        _, _, E_idx_in = init_state
        
        # Run multiple epochs over the collected data
        early_stop = False
        for epoch in range(self.ppo_epochs):
            # Shuffle the indices for each epoch
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                if end <= start:
                    continue
                    
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_sequences = [sequences[i] for i in mb_indices]
                mb_states = [states[i] for i in mb_indices]
                mb_decoding_order = [decoding_order[i] for i in mb_indices]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                mb_loss = 0
                mb_approx_kls = []
                
                for i in range(len(mb_indices)):
                    # Get state and sequence for this sample
                    h_V, h_E = mb_states[i]
                    h_V = h_V.requires_grad_(True)
                    h_E = h_E.requires_grad_(True)
                    S = mb_sequences[i]
                    
                    # Run forward pass with current policy
                    # pdb_lib.set_trace()
                    feature_dict_mb = feature_dict.copy()
                    feature_dict_mb["batch_size"] = 1
                    # feature_dict_mb["randn"] = torch.randn([1, feature_dict_mb["mask"].shape[1]], device=self.device)
                    decoding_order_sampled = mb_decoding_order[i].unsqueeze(0)
                    sampled_actions = S.unsqueeze(0)
                    # pdb_lib.set_trace()
                    out = self.rollout(feature_dict_mb, h_V, h_E, E_idx_in, decoding_order_sampled, sampled_actions)
                    
                    # Calculate log probs for the sequence
                    log_probs = out["log_probs"]
                    seq_mask = torch.nn.functional.one_hot(S, num_classes=len(alphabet)).float()
                    new_log_prob = (log_probs * seq_mask).sum(dim=(-1,-2))
                    seq_length = S.shape[0] 
                    new_log_prob = new_log_prob 
                    
                    # Calculate entropy
                    probs = torch.exp(log_probs)
                    # entropy = -(probs * log_probs).sum(dim=(-1,-2)).mean()
                    entropy_per_pos = -torch.sum(probs * log_probs, dim=-1)  # Sum over amino acids only
                    entropy = entropy_per_pos.mean()  # Mean over positions

                    
                    # Calculate ratio for PPO (using logratio for numerical stability)
                    logratio = new_log_prob - mb_old_log_probs[i]
                    ratio = logratio.exp()
                    
                    # Calculate clipfrac (useful for monitoring)
                    with torch.no_grad():
                        clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                        clipfracs.append(clipfrac)
                    
                    # Policy loss - IMPORTANT: Using max like CleanRL
                    pg_loss1 = -mb_advantages[i] * ratio
                    pg_loss2 = -mb_advantages[i] * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # KL divergence - using CleanRL's improved approximation
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        mb_approx_kls.append(approx_kl)
                    
                    # Combined loss
                    loss = pg_loss - self.entropy_coef * entropy 
                    
                    # Scale the loss by minibatch size for better stability
                    loss = loss / len(mb_indices)
                    
                    # Backward
                    loss.backward()
                    
                    # Track metrics
                    pg_losses.append(pg_loss.item())
                    entropy_losses.append(entropy.item())
                    total_losses.append(loss.item() * len(mb_indices))  # Unscale for logging
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.max_grad_norm
                )
                
                # Perform optimization step
                self.optimizer.step()
                
                # Early stopping based on KL divergence (per minibatch)
                if self.target_kl is not None:
                    avg_kl = sum(mb_approx_kls) / len(mb_approx_kls)
                    approx_kls.append(avg_kl)
                    if avg_kl > self.target_kl:
                        early_stop = True
                        print(f"Early stopping at KL {avg_kl:.4f} > {self.target_kl:.4f}")
                        break
            
            if early_stop:
                break
        
        # Calculate mean metrics
        metrics = {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(pg_losses),
            'entropy': np.mean(entropy_losses),
            'kl': np.mean(approx_kls) if approx_kls else 0,
            'clipfrac': np.mean(clipfracs)
        }
        
        # Store policy metrics for logging in train_step
        self.policy_metrics = {
            'policy/policy_loss': metrics['policy_loss'],
            'policy/entropy': metrics['entropy'],
            'policy/kl_divergence': metrics['kl'],
            'policy/clip_fraction': metrics['clipfrac'],
            'policy/total_loss': metrics['total_loss']
        }
        
        return metrics

    def train_step(self, step, init_state, feature_dict):
        """Single PPO training step"""
        to_log = {}
        
        # 1. Collect experience with fresh copies of the initial state
        h_V_in, h_E_in, E_idx_in = init_state
        fresh_init_state = (
            h_V_in.clone().detach(),
            h_E_in.clone().detach(),
            E_idx_in.clone().detach()
        )
        
        # 2. Collect experience
        batch, returns, advantages = self.collect_experience(
            step=step,
            init_state=fresh_init_state,
            feature_dict=feature_dict
        )
        # pdb_lib.set_trace()
        # 3. Update policy and value networks using PPO
        metrics = self.update_policy(batch, returns, advantages, fresh_init_state, feature_dict)
        
        # 4. Track average rewards for logging
        rewards = batch['rewards']
        avg_reward = rewards.mean().item()
        self.reward_history.append(torch.tensor(avg_reward, device=self.device))
        
        to_log.update(metrics)
        to_log['reward'] = avg_reward
        
        # 5. Log all metrics to wandb with the same step value
        if wandb.run:
            # Combine all metrics
            wandb_log = {}
            # Add reward stats (previously logged in compute_returns_and_advantages)
            if hasattr(self, 'aux_metrics'):
                wandb_log.update(self.aux_metrics)
            if hasattr(self, 'reward_stats'):
                wandb_log.update(self.reward_stats)
            # Add policy metrics (previously logged in update_policy)
            if hasattr(self, 'policy_metrics'):
                wandb_log.update(self.policy_metrics)
            # Add current metrics
            wandb_log['reward'] = avg_reward
            # Log everything at once with the same step
            wandb.log(wandb_log, step=step)
        
        return to_log