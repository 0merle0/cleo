import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from policy_utils import PolicyMPNN, alphabet


class ValueNetwork(nn.Module):
    """Value function estimator for PPO"""
    def __init__(self, input_dim=128, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ExperienceBuffer:
    """Storage for PPO experience collection"""
    def __init__(self, device):
        self.device = device
        self.clear()
        
    def clear(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.sequences = []
        self.states = []
        self.dones = []  # Added to track episode terminations
        
    def add(self, log_prob, value, reward, sequence, state, done=False):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.sequences.append(sequence)
        self.states.append(state)
        self.dones.append(torch.tensor(float(done), device=self.device))
        
    def get_batch(self):
        return {
            'log_probs': torch.stack(self.log_probs),
            'values': torch.stack(self.values),
            'rewards': torch.stack(self.rewards),
            'sequences': self.sequences,
            'states': self.states,
            'dones': torch.stack(self.dones) if self.dones else None
        }


class PPOPolicy(PolicyMPNN):
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
        
        # Initialize value network
        self.value_net = ValueNetwork(input_dim=128, hidden_dim=32).to(self.device)
        
        # Create optimizer for value network
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=cfg.get('value_lr', cfg.lr)
        )
        
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
    
    def estimate_value(self, h_V):
        """Estimate value from protein representation"""
        # h_V: [B, L, C] - use mean pooling across residues
        pooled = h_V.mean(dim=1)  # [B, C]
        return self.value_net(pooled)
    
    def collect_experience(self, step, n_rollouts, init_state, feature_dict):
        """Collect experience for PPO using the current policy"""
        self.buffer.clear()
        
        h_V_in, h_E_in, E_idx_in = init_state
        
        for _ in range(n_rollouts):
            # Make fresh copies before passing to rollout
            h_V = h_V_in.clone().detach().requires_grad_(True)
            h_E = h_E_in.clone().detach().requires_grad_(True)
            
            # Get sequence and probabilities
            out = self.rollout(feature_dict, h_V, h_E, E_idx_in)
            
            # Calculate log probs for the sampled sequence
            S = out["S"]
            log_probs = out["log_probs"]
            seq_mask = torch.nn.functional.one_hot(S, num_classes=len(alphabet)).float()
            log_prob = (log_probs * seq_mask).sum(dim=(-1,-2))
            
            # Normalize log_prob by sequence length for better numerical stability
            seq_length = S.shape[0] * S.shape[1]  # B x L
            normalized_log_prob = log_prob / seq_length
            
            # Get value estimate
            value = self.estimate_value(h_V)
            
            # Get reward - no gradient tracking needed
            with torch.no_grad():
                reward, _ = self.reward_fn(step, out, feature_dict, self.device)
            
            # Store experience - detach everything
            self.buffer.add(
                normalized_log_prob.detach(), 
                value.detach(), 
                reward.detach(), 
                S.detach() if isinstance(S, torch.Tensor) else S,
                (h_V.detach(), h_E.detach()),
                done=True  # Assuming each rollout is a complete episode
            )
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        return self.buffer.get_batch(), returns, advantages
        
    def compute_returns_and_advantages(self):
        """Compute returns and advantages using GAE"""
        # Convert lists to tensors
        rewards = torch.stack(self.buffer.rewards)
        values = torch.stack(self.buffer.values)
        dones = torch.stack(self.buffer.dones) if self.buffer.dones else torch.zeros_like(rewards)
        
        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        
        # GAE calculation
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0  # Assume last state is non-terminal for simplicity
                nextvalues = 0         # Terminal state value is 0
            else:
                nextnonterminal = 1.0 - dones[t + 1]  # Use next state's done flag
                nextvalues = values[t + 1]
                
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        # Calculate returns
        returns = advantages + values
        
        return returns, advantages

    def update_policy(self, batch, returns, advantages, init_state, feature_dict):
        """Update policy and value networks using PPO"""
        # Get data from batch
        old_log_probs = batch['log_probs']
        old_values = batch['values']
        rewards = batch['rewards']
        sequences = batch['sequences']
        states = batch['states']
        
        # Metrics to track
        clipfracs = []
        approx_kls = []
        pg_losses = []
        vf_losses = []
        entropy_losses = []
        total_losses = []
        
        # Mini-batch updates
        batch_size = len(old_log_probs)
        
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
                mb_old_values = old_values[mb_indices]
                mb_sequences = [sequences[i] for i in mb_indices]
                mb_states = [states[i] for i in mb_indices]
                
                # Normalize advantages per minibatch (like CleanRL)
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Zero gradients
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                mb_loss = 0
                mb_approx_kls = []
                
                for i in range(len(mb_indices)):
                    # Get state and sequence for this sample
                    h_V, h_E = mb_states[i]
                    h_V = h_V.requires_grad_(True)
                    h_E = h_E.requires_grad_(True)
                    S = mb_sequences[i]
                    
                    # Run forward pass with current policy
                    out = self.rollout(feature_dict, h_V, h_E, E_idx_in)
                    
                    # Calculate log probs for the sequence
                    log_probs = out["log_probs"]
                    seq_mask = torch.nn.functional.one_hot(S, num_classes=len(alphabet)).float()
                    new_log_prob = (log_probs * seq_mask).sum(dim=(-1,-2))
                    seq_length = S.shape[0] * S.shape[1]
                    new_log_prob = new_log_prob / seq_length
                    
                    # Calculate entropy
                    probs = torch.exp(log_probs)
                    # entropy = -(probs * log_probs).sum(dim=(-1,-2)).mean()
                    entropy_per_pos = -torch.sum(probs * log_probs, dim=-1)  # Sum over amino acids only
                    entropy = entropy_per_pos.mean()  # Mean over positions

                    # Get new value prediction
                    new_value = self.estimate_value(h_V)
                    
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
                    
                    # Value loss with optional clipping
                    if self.clip_vloss:
                        v_loss_unclipped = (new_value - mb_returns[i]) ** 2
                        v_clipped = mb_old_values[i] + torch.clamp(
                            new_value - mb_old_values[i],
                            -self.clip_param,
                            self.clip_param
                        )
                        v_loss_clipped = (v_clipped - mb_returns[i]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        value_loss = 0.5 * v_loss_max.mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(new_value, mb_returns[i].unsqueeze(0))
                    
                    # KL divergence - using CleanRL's improved approximation
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        mb_approx_kls.append(approx_kl)
                    
                    # Combined loss
                    loss = pg_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    
                    # Scale the loss by minibatch size for better stability
                    loss = loss / len(mb_indices)
                    
                    # Backward
                    loss.backward()
                    
                    # Track metrics
                    pg_losses.append(pg_loss.item())
                    vf_losses.append(value_loss.item())
                    entropy_losses.append(entropy.item())
                    total_losses.append(loss.item() * len(mb_indices))  # Unscale for logging
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.max_grad_norm
                )
                
                # Perform optimization step
                self.optimizer.step()
                self.value_optimizer.step()
                
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
            'value_loss': np.mean(vf_losses),
            'entropy': np.mean(entropy_losses),
            'kl': np.mean(approx_kls) if approx_kls else 0,
            'clipfrac': np.mean(clipfracs)
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
            n_rollouts=self.cfg.batch_size, 
            init_state=fresh_init_state,
            feature_dict=feature_dict
        )
        
        # 3. Update policy and value networks using PPO
        metrics = self.update_policy(batch, returns, advantages, fresh_init_state, feature_dict)
        
        # 4. Track average rewards for logging
        rewards = batch['rewards']
        avg_reward = rewards.mean().item()
        self.reward_history.append(torch.tensor(avg_reward, device=self.device))
        
        to_log.update(metrics)
        to_log['reward'] = avg_reward
        
        return to_log