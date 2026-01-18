import sys, os, copy
from abc import ABC, abstractmethod
import numpy as np
import torch
from policy_utils import alphabet
import pdb as pdb_lib
import pandas as pd
from omegaconf import OmegaConf
import subprocess

from fragment_utils import make_fragment_dict, sample_sequences, get_fragment_rewards


class Reward(ABC):
    """
        Base class for reward functions

        ! should try and ensure rewards are in the range of [0, 1]
    """

    def get_sequences(self, policy_output):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]
        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    @abstractmethod
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):
        """
            Compute rewards for each sequence in the sampled_sequences
            rewards should be returned as tensor of shape (B,), as well as any metrics you would like to track

            policy_output: Output from the policy network - dict
            feature_dict: Feature dictionary - dict
            evaluate: Flag indicating if this is evaluation mode

        """
        pass


class EnrichAminoAcidReward(Reward):
    """
        Simple reward to upweight an amino acid of interest

        Make sure final reward ends up on same device
    """
    def __init__(self, AA_to_enrich=None):
        self.AA_to_enrich = AA_to_enrich
        assert AA_to_enrich in alphabet, f"Amino acid {AA_to_enrich} not in alphabet"
        self.AA_to_enrich_idx = alphabet.index(AA_to_enrich)

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):
        sampled_seqs = policy_output["S"]
        num_correct_aas = (sampled_seqs == self.AA_to_enrich_idx).float().sum(dim=-1)
        reward = num_correct_aas / sampled_seqs.shape[1]
        
        metrics = {
            "num_correct_aas": num_correct_aas.mean().cpu().item()
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            metrics['policy_log_probs'] = policy_output.get('log_probs', None)

        return reward.to(device), metrics


class AF3RMSDPipelineReward(Reward):
    """
        Penicillin active site, using af3 RMSD as reward
    """

    def __init__(
            self, 
            pipeline_config_path, 
            output_dir, 
            run_name,
            rmsd_ub=10.0, 
            rmsd_lb=0.0, 
            frag_cfg=None, 
            max_retries=3
        ):

        # sys.path.append("/projects/ml/itopt/policy_mpnn/software/pipelines")
        sys.path.append("/home/ssalike/git/pipelines")
        # if we used an apptainer we would could install cifutils and datahub directly
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/cifutils/src")
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/datahub/src")        
        os.environ["PYTHONPATH"] = ":" # pipeline will freak if this is not set

        from pipelines.pipeline import main as run_pipeline

        self.run_pipeline = run_pipeline
        self.pipeline_config = OmegaConf.load(pipeline_config_path)
        self.output_dir = output_dir
        self.run_name = run_name
        self.rmsd_ub = rmsd_ub
        self.rmsd_lb = rmsd_lb
        self.frag_cfg = frag_cfg
        self.max_retries = max_retries

        # make subdirectory for pipeline output
        pipeline_output_dir = os.path.join(self.output_dir, self.run_name, "pipeline_output")
        os.makedirs(pipeline_output_dir, exist_ok=True)
    
    def get_sequences(self, policy_output, chain_mask=None):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]

        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    def get_input_df(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame(
            {
                "sequence": sequences,
                "name": [f"seq_{i:04}" for i in range(len(sequences))],
                "origin.path": [f"seq_{i:04}.path" for i in range(len(sequences))],
            }
        )

        return df


    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}" if not evaluate else f"pipeline_output_eval_batch_{step:04}"
        )
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)

        if self.frag_cfg is not None:
            fragment_dict = make_fragment_dict(sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)


        df_out = self.run_pipeline(df_input, config)


        # add code to delete af3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)  # Clean up outputs to retry

        # Reward shaping
        as_rmsd = torch.tensor(df_out["alignment.motif_allatom_align.motif_allatom_rmsd"].tolist())
        iptm = torch.tensor(df_out["af3_iptm"].tolist())

        # Normalize RMSD to [0,1] range
        as_rmsd_clamped = torch.clamp(as_rmsd, min=self.rmsd_lb, max=self.rmsd_ub)
        rmsd_reward =  1 - (as_rmsd_clamped / (self.rmsd_ub - self.rmsd_lb))

        # Combine rmsd and iptm rewards
        reward = rmsd_reward * iptm

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "rmsd_mean": as_rmsd.mean().cpu().item(),
            "rmsd_min": as_rmsd.min().cpu().item(),
            "rmsd_max": as_rmsd.max().cpu().item(),
            "iptm_mean": iptm.mean().cpu().item(),
            "iptm_min": iptm.min().cpu().item(),
            "iptm_max": iptm.max().cpu().item(),
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            metrics['policy_log_probs'] = policy_output.get('log_probs', None)

        return reward.to(device), metrics

class granzyme_pipeline_reward(AF3RMSDPipelineReward):
    """
        Granzyme pipeline reward, inheriting from AF3RMSDPipelineReward
    """
    def get_dist_to_ref_seq(self, sequences):
        assert self.ref_seq is not None, "Reference sequence is None"
        dist_list = []
        for seq in sequences:
            dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
            dist_list.append(float(dist))

        return torch.tensor(dist_list)
    
    def get_pairwise_diversity(self, sequences):
        dist_list = []
        for i in range(len(sequences)):
            _list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
            dist_list.append(np.mean(_list))

        # return mean pairwise distance
        return torch.tensor(dist_list)
    
    def __init__(self, 
                pipeline_config_path, 
                output_dir, 
                run_name, 
                rmsd_ub=10.0, 
                rmsd_lb=0.0, 
                pae_ub=15.0, 
                pae_lb=0.0, 
                ref_seq=None,
                max_dist_to_ref_seq=20,
                min_dist_to_ref_seq=6,
                max_pairwise_diversity=20, 
                min_pairwise_diversity=0, 
                esm_perplexity_ub=15.0, 
                esm_perplexity_lb=6.0,
                max_retries=3,
                frag_cfg=None, 
                ):
        """
            Initialize the Granzyme reward. Adds PAE-based piecewise-linear reward scaling in addition to RMSD.

            Args:
                pipeline_config_path (str): Path to the pipeline config YAML.
                output_dir (str): Directory where all outputs will be written.
                run_name (str): Name of the current run.
                rmsd_ub (float): Upper bound for RMSD scaling (same semantics as parent).
                rmsd_lb (float): Lower bound for RMSD scaling.
                pae_ub (float): Upper bound for PAE scaling (reward→0 when >= this value).
                pae_lb (float): Lower bound for PAE scaling (reward→1 when <= this value).
                frag_cfg: Optional fragment-based reward configuration.
                max_retries (int): Max retries for pipeline failures.
        """
        super().__init__(
            pipeline_config_path=pipeline_config_path,
            output_dir=output_dir,
            run_name=run_name,
            rmsd_ub=rmsd_ub,
            rmsd_lb=rmsd_lb,
            frag_cfg=frag_cfg,
            max_retries=max_retries,
        )

        # Store PAE bounds for use during reward calculation
        self.pae_ub = pae_ub
        self.pae_lb = pae_lb
        self.ref_seq = ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        
        if evaluate:
            eval_idx = 0
            while os.path.exists(os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")):
                eval_idx += 1 # find the next available eval index for supporting multiple evals during training
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")
        else:
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_iter_{step:04}")
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        orig_sequences = self.get_sequences(policy_output, chain_mask=chain_mask)
        
        if self.frag_cfg is not None:
            sample_size = self.frag_cfg.sample_size if not evaluate else 32
            fragment_dict = make_fragment_dict(orig_sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
            # remove duplicate sequences
            # sequences = list(set(sequences))
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)

        # Run the pipeline
        df_out = self.run_pipeline(df_input, config)

        # Clean up AF3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)

        # Reward shaping - granzyme specific
        as_rmsd1 = torch.tensor(df_out["af3_p1_metrics.motif_allatom_align.motif_allatom_rmsd"].tolist())
        ipae1 = torch.tensor(df_out["af3_p1.af3_chain_pair_pae_min"].tolist())

        # Piecewise linear RMSD reward using existing lb/ub parameters
        #   RMSD ≤ rmsd_lb  → reward = 1.0
        #   rmsd_lb < RMSD < rmsd_ub → linear decrease from 1.0 to 0.0
        #   RMSD ≥ rmsd_ub  → reward = 0.0
        as_rmsd_clamped = torch.clamp(as_rmsd1, min=self.rmsd_lb, max=self.rmsd_ub)
        rmsd_reward1 = torch.where(
            as_rmsd1 <= self.rmsd_lb,
            torch.ones_like(as_rmsd1),
            1 - (as_rmsd_clamped - self.rmsd_lb) / (self.rmsd_ub - self.rmsd_lb),
        )

        # Piecewise linear PAE reward using pae_lb/pae_ub parameters
        #   PAE ≤ pae_lb  → reward = 1.0
        #   pae_lb < PAE < pae_ub → linear decrease from 1.0 to 0.0
        #   PAE ≥ pae_ub  → reward = 0.0
        ipae_clamped = torch.clamp(ipae1, min=self.pae_lb, max=self.pae_ub)
        pae_reward1 = torch.where(
            ipae1 <= self.pae_lb,
            torch.ones_like(ipae1),
            1 - (ipae_clamped - self.pae_lb) / (self.pae_ub - self.pae_lb),
        )

        ipae2 = torch.tensor(df_out["af3_p2.af3_chain_pair_pae_min"].tolist())
        ipae_clamped = torch.clamp(ipae2, min=self.pae_lb, max=self.pae_ub)
        pae_reward2 = torch.where(
            ipae2 <= self.pae_lb,
            torch.ones_like(ipae2),
            1 - (ipae_clamped - self.pae_lb) / (self.pae_ub - self.pae_lb),
        )
        
        reward = torch.clamp(rmsd_reward1 * (pae_reward1 - pae_reward2) * pae_reward1, min=0.0, max=1.0)
        
        # Store recombined rewards for evaluation before fragment aggregation
        # get distance to reference sequence if provided
        if self.ref_seq is not None:
            dist_from_ref_seq = self.get_dist_to_ref_seq(df_out["sequence"].tolist()) 
            clampled_dist = torch.clamp(dist_from_ref_seq, min=self.min_dist_to_ref_seq, max=self.max_dist_to_ref_seq)
            norm_dist = (clampled_dist - self.min_dist_to_ref_seq) / (self.max_dist_to_ref_seq - self.min_dist_to_ref_seq + 1e-6)
            dist_from_ref_seq_reward = 1 - norm_dist

        # compute distance to other sequences in the batch to encourage diversity
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist())
        clamped_diversity = torch.clamp(pairwise_diversity, min=0, max=20)
        pairwise_diversity_reward = (clamped_diversity - self.min_pairwise_diversity) / (self.max_pairwise_diversity - self.min_pairwise_diversity + 1e-6)
        
        esm_perplexity = torch.tensor(df_out["esm_perplexity.perplexity"].tolist())
        esm_perplexity_clamped = torch.clamp(esm_perplexity, min=self.esm_perplexity_lb, max=self.esm_perplexity_ub)
        esm_perplexity_reward = 1 - (esm_perplexity_clamped - self.esm_perplexity_lb) / (self.esm_perplexity_ub - self.esm_perplexity_lb)

        # average the auxiliary rewards from above
        # aux_rewards = torch.stack([dist_from_ref_seq_reward, esm_perplexity_reward], dim=0)
        # aux_rewards_mean = aux_rewards.mean(dim=0)
        # aux_rewards_mean = aux_rewards_mean.to(device)
        
        # total reward is the mean of reward and auxiliary rewards
        total_reward = reward * dist_from_ref_seq_reward * pairwise_diversity_reward * esm_perplexity_reward

        # Store recombined rewards for evaluation before fragment aggregation
        recombined_rewards = total_reward.clone() if evaluate else None

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, total_reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "rmsd_mean_p1": as_rmsd1.mean().cpu().item(),
            "rmsd_min_p1": as_rmsd1.min().cpu().item(),
            "rmsd_max_p1": as_rmsd1.max().cpu().item(),
            "ipae_mean_p1": ipae1.mean().cpu().item(),
            "ipae_min_p1": ipae1.min().cpu().item(),
            "ipae_max_p1": ipae1.max().cpu().item(),
            "ipae_mean_p2": ipae2.mean().cpu().item(),
            "ipae_min_p2": ipae2.min().cpu().item(),
            "ipae_max_p2": ipae2.max().cpu().item(),
            "ipae_diff_mean": (ipae1 - ipae2).mean().cpu().item(),
            "pae_reward_mean": (pae_reward1 - pae_reward2).mean().cpu().item(),
            "pae_reward_min": (pae_reward1 - pae_reward2).min().cpu().item(),
            "pae_reward_max": (pae_reward1 - pae_reward2).max().cpu().item(),
            "esm_perplexity_mean": esm_perplexity.mean().cpu().item(),
            "esm_perplexity_min": esm_perplexity.min().cpu().item(),
            "esm_perplexity_max": esm_perplexity.max().cpu().item(),
            "esm_perplexity_reward_mean": esm_perplexity_reward.mean().cpu(),
            "esm_perplexity_reward_min": esm_perplexity_reward.min().cpu().item(),
            "esm_perplexity_reward_max": esm_perplexity_reward.max().cpu().item(),
            "dist_from_ref_seq_mean": dist_from_ref_seq.mean().cpu().item(),
            "dist_from_ref_seq_min": dist_from_ref_seq.min().cpu().item(),
            "dist_from_ref_seq_max": dist_from_ref_seq.max().cpu().item(),
            "dist_from_ref_seq_reward_mean": dist_from_ref_seq_reward.mean().cpu(),
            "dist_from_ref_seq_reward_min": dist_from_ref_seq_reward.min().cpu().item(),
            "dist_from_ref_seq_reward_max": dist_from_ref_seq_reward.max().cpu().item(),
            "pairwise_diversity_mean": pairwise_diversity.mean().cpu().item(),
            "pairwise_diversity_min": pairwise_diversity.min().cpu().item(),
            "pairwise_diversity_max": pairwise_diversity.max().cpu().item(),
            "pairwise_diversity_reward_mean": pairwise_diversity_reward.mean().cpu(),
            "pairwise_diversity_reward_min": pairwise_diversity_reward.min().cpu().item(),
            "pairwise_diversity_reward_max": pairwise_diversity_reward.max().cpu().item(),
            "total_reward_mean": total_reward.mean().cpu().item(),
            "total_reward_min": total_reward.min().cpu().item(),
            "total_reward_max": total_reward.max().cpu().item(),
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            # metrics['sampling_log_probs'] = policy_output.get('sampling_probs', None)
            # metrics['log_probs'] = policy_output.get('log_probs', None)
            metrics['sampled_sequences'] = orig_sequences
            
            if self.frag_cfg is not None:
                metrics['recombined_sequences'] = sequences  # The recombined sequence strings
                metrics['recombined_rewards'] = recombined_rewards.cpu().numpy()  # Rewards before fragment aggregation
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used


        return reward.to(device), metrics

class granzyme_chai_reward(AF3RMSDPipelineReward):
    """
        Granzyme pipeline reward, inheriting from AF3RMSDPipelineReward
    """
    def __init__(self, pipeline_config_path, output_dir, run_name, rmsd_ub=10.0, rmsd_lb=0.0, pae_ub=15.0, pae_lb=0.0, frag_cfg=None, max_retries=3):
        """
            Initialize the Granzyme reward. Adds PAE-based piecewise-linear reward scaling in addition to RMSD.

            Args:
                pipeline_config_path (str): Path to the pipeline config YAML.
                output_dir (str): Directory where all outputs will be written.
                run_name (str): Name of the current run.
                rmsd_ub (float): Upper bound for RMSD scaling (same semantics as parent).
                rmsd_lb (float): Lower bound for RMSD scaling.
                pae_ub (float): Upper bound for PAE scaling (reward→0 when >= this value).
                pae_lb (float): Lower bound for PAE scaling (reward→1 when <= this value).
                frag_cfg: Optional fragment-based reward configuration.
                max_retries (int): Max retries for pipeline failures.
        """
        super().__init__(
            pipeline_config_path=pipeline_config_path,
            output_dir=output_dir,
            run_name=run_name,
            rmsd_ub=rmsd_ub,
            rmsd_lb=rmsd_lb,
            frag_cfg=frag_cfg,
            max_retries=max_retries,
        )

        # Store PAE bounds for use during reward calculation
        self.pae_ub = pae_ub
        self.pae_lb = pae_lb

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        
        if evaluate:
            eval_idx = 0
            while os.path.exists(os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")):
                eval_idx += 1 # find the next available eval index for supporting multiple evals during training
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")
        else:
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_iter_{step:04}")
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        orig_sequences = self.get_sequences(policy_output, chain_mask=chain_mask)
        sequences = orig_sequences
        
        if self.frag_cfg is not None:
            fragment_dict = make_fragment_dict(orig_sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)

        # Run the pipeline
        df_out = self.run_pipeline(df_input, config)

        # Clean up AF3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)

        # Reward shaping - granzyme specific
        as_rmsd1 = torch.tensor(df_out["chai_p1_metrics.motif_allatom_align.motif_allatom_rmsd"].tolist())
        ipae1 = torch.tensor(df_out["chai_p1.pae"].tolist())

        # Piecewise linear RMSD reward using existing lb/ub parameters
        #   RMSD ≤ rmsd_lb  → reward = 1.0
        #   rmsd_lb < RMSD < rmsd_ub → linear decrease from 1.0 to 0.0
        #   RMSD ≥ rmsd_ub  → reward = 0.0
        as_rmsd_clamped = torch.clamp(as_rmsd1, min=self.rmsd_lb, max=self.rmsd_ub)
        rmsd_reward1 = torch.where(
            as_rmsd1 <= self.rmsd_lb,
            torch.ones_like(as_rmsd1),
            1 - (as_rmsd_clamped - self.rmsd_lb) / (self.rmsd_ub - self.rmsd_lb),
        )

        # Piecewise linear PAE reward using pae_lb/pae_ub parameters
        #   PAE ≤ pae_lb  → reward = 1.0
        #   pae_lb < PAE < pae_ub → linear decrease from 1.0 to 0.0
        #   PAE ≥ pae_ub  → reward = 0.0
        ipae_clamped = torch.clamp(ipae1, min=self.pae_lb, max=self.pae_ub)
        pae_reward1 = torch.where(
            ipae1 <= self.pae_lb,
            torch.ones_like(ipae1),
            1 - (ipae_clamped - self.pae_lb) / (self.pae_ub - self.pae_lb),
        )

        ipae2 = torch.tensor(df_out["chai_p2.pae"].tolist())
        ipae_clamped = torch.clamp(ipae2, min=self.pae_lb, max=self.pae_ub)
        pae_reward2 = torch.where(
            ipae2 <= self.pae_lb,
            torch.ones_like(ipae2),
            1 - (ipae_clamped - self.pae_lb) / (self.pae_ub - self.pae_lb),
        )
        
        reward = torch.clamp(rmsd_reward1 * (pae_reward1 - pae_reward2) * pae_reward1, min=0.0, max=1.0)
        
        # Store recombined rewards for evaluation before fragment aggregation
        recombined_rewards = reward.clone() if evaluate else None

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "rmsd_mean_p1": as_rmsd1.mean().cpu().item(),
            "rmsd_min_p1": as_rmsd1.min().cpu().item(),
            "rmsd_max_p1": as_rmsd1.max().cpu().item(),
            "ipae_mean_p1": ipae1.mean().cpu().item(),
            "ipae_min_p1": ipae1.min().cpu().item(),
            "ipae_max_p1": ipae1.max().cpu().item(),
            "ipae_mean_p2": ipae2.mean().cpu().item(),
            "ipae_min_p2": ipae2.min().cpu().item(),
            "ipae_max_p2": ipae2.max().cpu().item(),
            "ipae_diff_mean": (ipae1 - ipae2).mean().cpu().item(),
            "pae_reward_mean": (pae_reward1 - pae_reward2).mean().cpu().item(),
            "pae_reward_min": (pae_reward1 - pae_reward2).min().cpu().item(),
            "pae_reward_max": (pae_reward1 - pae_reward2).max().cpu().item(),
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            # metrics['sampling_log_probs'] = policy_output.get('sampling_probs', None)
            # metrics['log_probs'] = policy_output.get('log_probs', None)
            metrics['pipeline_output_dir'] = config.rundir
            metrics['sampled_sequences'] = orig_sequences
            
            if self.frag_cfg is not None:
                metrics['recombined_sequences'] = sequences  # The recombined sequence strings
                metrics['recombined_rewards'] = recombined_rewards.cpu().numpy()  # Rewards before fragment aggregation
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used
                metrics['recombined_metrics'] = {}
                metrics['recombined_metrics']['ipae1'] = ipae1.cpu().numpy()
                metrics['recombined_metrics']['ipae2'] = ipae2.cpu().numpy()
                metrics['recombined_metrics']['rmsd1'] = as_rmsd1.cpu().numpy()

        return reward.to(device), metrics


class granzyme_chai_reward_constrained(AF3RMSDPipelineReward):
    """
        Granzyme pipeline reward, inheriting from AF3RMSDPipelineReward
    """
    def get_dist_to_ref_seq(self, sequences):
        assert self.ref_seq is not None, "Reference sequence is None"
        dist_list = []
        for seq in sequences:
            dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
            dist_list.append(float(dist))

        return torch.tensor(dist_list)
    
    def get_pairwise_diversity(self, sequences):
        dist_list = []
        for i in range(len(sequences)):
            _list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
            dist_list.append(np.mean(_list))

        # return mean pairwise distance
        return torch.tensor(dist_list)
    
    def __init__(self, 
                pipeline_config_path, 
                output_dir, 
                run_name, 
                rmsd_ub=10.0, 
                rmsd_lb=0.0, 
                pae_ub=15.0, 
                pae_lb=0.0, 
                ref_seq=None,
                max_dist_to_ref_seq=20,
                min_dist_to_ref_seq=6,
                max_pairwise_diversity=20, 
                min_pairwise_diversity=0, 
                esm_perplexity_ub=15.0, 
                esm_perplexity_lb=6.0,
                max_retries=3,
                frag_cfg=None, 
                ):
        """
            Initialize the Granzyme reward. Adds PAE-based piecewise-linear reward scaling in addition to RMSD.

            Args:
                pipeline_config_path (str): Path to the pipeline config YAML.
                output_dir (str): Directory where all outputs will be written.
                run_name (str): Name of the current run.
                rmsd_ub (float): Upper bound for RMSD scaling (same semantics as parent).
                rmsd_lb (float): Lower bound for RMSD scaling.
                pae_ub (float): Upper bound for PAE scaling (reward→0 when >= this value).
                pae_lb (float): Lower bound for PAE scaling (reward→1 when <= this value).
                frag_cfg: Optional fragment-based reward configuration.
                max_retries (int): Max retries for pipeline failures.
        """
        super().__init__(
            pipeline_config_path=pipeline_config_path,
            output_dir=output_dir,
            run_name=run_name,
            rmsd_ub=rmsd_ub,
            rmsd_lb=rmsd_lb,
            frag_cfg=frag_cfg,
            max_retries=max_retries,
        )

        # Store PAE bounds for use during reward calculation
        self.pae_ub = pae_ub
        self.pae_lb = pae_lb
        self.ref_seq = ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        
        if evaluate:
            eval_idx = 0
            while os.path.exists(os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")):
                eval_idx += 1 # find the next available eval index for supporting multiple evals during training
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")
        else:
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_iter_{step:04}")
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        orig_sequences = self.get_sequences(policy_output, chain_mask=chain_mask)
        sequences = orig_sequences
        
        if self.frag_cfg is not None:
            fragment_dict = make_fragment_dict(orig_sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)

        # Run the pipeline
        df_out = self.run_pipeline(df_input, config)

        # Clean up AF3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)

        # Reward shaping - granzyme specific
        as_rmsd1 = torch.tensor(df_out["chai_p1_metrics.motif_allatom_align.motif_allatom_rmsd"].tolist())
        ipae1 = torch.tensor(df_out["chai_p1.pae"].tolist())

        # Piecewise linear RMSD reward using existing lb/ub parameters
        #   RMSD ≤ rmsd_lb  → reward = 1.0
        #   rmsd_lb < RMSD < rmsd_ub → linear decrease from 1.0 to 0.0
        #   RMSD ≥ rmsd_ub  → reward = 0.0
        as_rmsd_clamped = torch.clamp(as_rmsd1, min=self.rmsd_lb, max=self.rmsd_ub)
        rmsd_reward1 = torch.where(
            as_rmsd1 <= self.rmsd_lb,
            torch.ones_like(as_rmsd1),
            1 - (as_rmsd_clamped - self.rmsd_lb) / (self.rmsd_ub - self.rmsd_lb),
        )

        # Piecewise linear PAE reward using pae_lb/pae_ub parameters
        #   PAE ≤ pae_lb  → reward = 1.0
        #   pae_lb < PAE < pae_ub → linear decrease from 1.0 to 0.0
        #   PAE ≥ pae_ub  → reward = 0.0
        ipae_clamped = torch.clamp(ipae1, min=self.pae_lb, max=self.pae_ub)
        pae_reward1 = torch.where(
            ipae1 <= self.pae_lb,
            torch.ones_like(ipae1),
            1 - (ipae_clamped - self.pae_lb) / (self.pae_ub - self.pae_lb),
        )

        ipae2 = torch.tensor(df_out["chai_p2.pae"].tolist())
        ipae_clamped = torch.clamp(ipae2, min=self.pae_lb, max=self.pae_ub)
        pae_reward2 = torch.where(
            ipae2 <= self.pae_lb,
            torch.ones_like(ipae2),
            1 - (ipae_clamped - self.pae_lb) / (self.pae_ub - self.pae_lb),
        )
        
        reward = torch.clamp(rmsd_reward1 * (pae_reward1 - pae_reward2) * pae_reward1, min=0.0, max=1.0)
        
        # get distance to reference sequence if provided
        if self.ref_seq is not None:
            dist_from_ref_seq = self.get_dist_to_ref_seq(df_out["sequence"].tolist()) 
            clampled_dist = torch.clamp(dist_from_ref_seq, min=self.min_dist_to_ref_seq, max=self.max_dist_to_ref_seq)
            norm_dist = (clampled_dist - self.min_dist_to_ref_seq) / (self.max_dist_to_ref_seq - self.min_dist_to_ref_seq + 1e-6)
            dist_from_ref_seq_reward = 1 - norm_dist

        # compute distance to other sequences in the batch to encourage diversity
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist())
        clamped_diversity = torch.clamp(pairwise_diversity, min=0, max=20)
        pairwise_diversity_reward = (clamped_diversity - self.min_pairwise_diversity) / (self.max_pairwise_diversity - self.min_pairwise_diversity + 1e-6)
        
        # combine auxiliary rewards by product to mirror pipeline behavior
        total_reward = reward * pairwise_diversity_reward 
        if self.ref_seq is not None:
            total_reward = total_reward * dist_from_ref_seq_reward

        # Store recombined rewards for evaluation before fragment aggregation
        recombined_rewards = total_reward.clone() if evaluate else None

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, total_reward, fragment_dict, self.frag_cfg.fragment_bounds)
        else:
            reward = total_reward

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "rmsd_mean_p1": as_rmsd1.mean().cpu().item(),
            "rmsd_min_p1": as_rmsd1.min().cpu().item(),
            "rmsd_max_p1": as_rmsd1.max().cpu().item(),
            "ipae_mean_p1": ipae1.mean().cpu().item(),
            "ipae_min_p1": ipae1.min().cpu().item(),
            "ipae_max_p1": ipae1.max().cpu().item(),
            "ipae_mean_p2": ipae2.mean().cpu().item(),
            "ipae_min_p2": ipae2.min().cpu().item(),
            "ipae_max_p2": ipae2.max().cpu().item(),
            "ipae_diff_mean": (ipae1 - ipae2).mean().cpu().item(),
            "pae_reward_mean": (pae_reward1 - pae_reward2).mean().cpu().item(),
            "pae_reward_min": (pae_reward1 - pae_reward2).min().cpu().item(),
            "pae_reward_max": (pae_reward1 - pae_reward2).max().cpu().item(),
            "pairwise_diversity_mean": pairwise_diversity.mean().cpu().item(),
            "pairwise_diversity_min": pairwise_diversity.min().cpu().item(),
            "pairwise_diversity_max": pairwise_diversity.max().cpu().item(),
            "pairwise_diversity_reward_mean": pairwise_diversity_reward.mean().cpu(),
            "pairwise_diversity_reward_min": pairwise_diversity_reward.min().cpu().item(),
            "pairwise_diversity_reward_max": pairwise_diversity_reward.max().cpu().item(),
            "total_reward_mean": total_reward.mean().cpu().item(),
            "total_reward_min": total_reward.min().cpu().item(),
            "total_reward_max": total_reward.max().cpu().item(),
        }
        if self.ref_seq is not None:
            metrics["dist_from_ref_seq_mean"] = dist_from_ref_seq.mean().cpu().item()
            metrics["dist_from_ref_seq_min"] = dist_from_ref_seq.min().cpu().item()
            metrics["dist_from_ref_seq_max"] = dist_from_ref_seq.max().cpu().item()
            metrics["dist_from_ref_seq_reward_mean"] = dist_from_ref_seq_reward.mean().cpu()
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            # metrics['sampling_log_probs'] = policy_output.get('sampling_probs', None)
            # metrics['log_probs'] = policy_output.get('log_probs', None)
            metrics['pipeline_output_dir'] = config.rundir
            metrics['sampled_sequences'] = orig_sequences
            
            if self.frag_cfg is not None:
                metrics['recombined_sequences'] = sequences  # The recombined sequence strings
                metrics['recombined_rewards'] = recombined_rewards.cpu().numpy()  # Rewards before fragment aggregation
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used
                metrics['recombined_metrics'] = {}
                metrics['recombined_metrics']['ipae1'] = ipae1.cpu().numpy()
                metrics['recombined_metrics']['ipae2'] = ipae2.cpu().numpy()
                metrics['recombined_metrics']['rmsd1'] = as_rmsd1.cpu().numpy()

        return reward.to(device), metrics





class AF3ClickEnzymeReward(Reward):
    """
        Penicillin active site, using af3 RMSD as reward
    """

    def __init__(
            self, 
            pipeline_config_path, 
            output_dir, 
            run_name,
            ptm_lb=0.0,
            ptm_ub=0.85,
            iptm_lb=0.0,
            iptm_ub=0.85,
            rmsd_3AH_lb=0.0,
            rmsd_3AH_ub=10.0,
            rmsd_DEP_lb=0.0,
            rmsd_DEP_ub=10.0,
            dist_CR1_NR1_lb=0.5,
            dist_CR1_NR1_ub=5.0,
            dist_CR2_NR3_lb=0.5,
            dist_CR2_NR3_ub=5.0,
            ref_seq=None,
            max_dist_to_ref_seq=100,
            min_dist_to_ref_seq=0,
            frag_cfg=None, 
            max_pairwise_diversity=20,
            min_pairwise_diversity=0,
            esm_perplexity_ub=15.0,
            esm_perplexity_lb=6.0,
        ):

        sys.path.append("/projects/ml/itopt/policy_mpnn/software/pipelines")
        # if we used an apptainer we would could install cifutils and datahub directly
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/cifutils/src")
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/datahub/src")        
        os.environ["PYTHONPATH"] = ":" # pipeline will freak if this is not set

        from pipelines.pipeline import main as run_pipeline

        self.run_pipeline = run_pipeline
        self.pipeline_config = OmegaConf.load(pipeline_config_path)
        self.output_dir = output_dir
        self.run_name = run_name
        self.ptm_ub = ptm_ub
        self.ptm_lb = ptm_lb
        self.iptm_ub = iptm_ub
        self.iptm_lb = iptm_lb
        self.rmsd_3AH_ub = rmsd_3AH_ub
        self.rmsd_3AH_lb = rmsd_3AH_lb
        self.rmsd_DEP_ub = rmsd_DEP_ub
        self.rmsd_DEP_lb = rmsd_DEP_lb
        self.dist_CR1_NR1_ub = dist_CR1_NR1_ub
        self.dist_CR1_NR1_lb = dist_CR1_NR1_lb
        self.dist_CR2_NR3_ub = dist_CR2_NR3_ub
        self.dist_CR2_NR3_lb = dist_CR2_NR3_lb
        self.ref_seq = ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb
        self.frag_cfg = frag_cfg

        # make subdirectory for pipeline output
        pipeline_output_dir = os.path.join(self.output_dir, self.run_name, "pipeline_output")
        os.makedirs(pipeline_output_dir, exist_ok=True)
    
    def get_sequences(self, policy_output, chain_mask=None):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]

        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    def get_input_df(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame(
            {
                "sequence": sequences,
                "name": [f"seq_{i:04}" for i in range(len(sequences))],
                "origin.path": [f"seq_{i:04}.path" for i in range(len(sequences))],
            }
        )

        return df
    
    def get_dist_to_ref_seq(self, sequences):
        assert self.ref_seq is not None, "Reference sequence is None"
        dist_list = []
        for seq in sequences:
            dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
            dist_list.append(float(dist))

        return torch.tensor(dist_list)
    
    def get_pairwise_diversity(self, sequences):
        dist_list = []
        for i in range(len(sequences)):
            _list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
            dist_list.append(np.mean(_list))

        # return mean pairwise distance
        return torch.tensor(dist_list)

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}"
        )
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)

        if self.frag_cfg is not None:
            fragment_dict = make_fragment_dict(sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)
        df_out = self.run_pipeline(df_input, config)


        # add code to delete af3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "click/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)  # Clean up outputs to retry

        # get distance to reference sequence if provided
        if self.ref_seq is not None:
            dist_from_ref_seq = self.get_dist_to_ref_seq(df_out["sequence"].tolist()) 
            clampled_dist = torch.clamp(dist_from_ref_seq, min=self.min_dist_to_ref_seq, max=self.max_dist_to_ref_seq)
            norm_dist = (clampled_dist - self.min_dist_to_ref_seq) / (self.max_dist_to_ref_seq - self.min_dist_to_ref_seq + 1e-6)
            dist_from_ref_seq_reward = 1 - norm_dist


        # compute distance to other sequences in the batch to encourage diversity
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist())
        clamped_diversity = torch.clamp(pairwise_diversity, min=self.min_pairwise_diversity, max=self.max_pairwise_diversity)
        pairwise_diversity_reward = (clamped_diversity - self.min_pairwise_diversity) / (self.max_pairwise_diversity - self.min_pairwise_diversity + 1e-6)

        # Reward engineering
        lig_rmsd_3AH_PRD = torch.tensor(df_out["click.lig_rmsd_3AH_PRD"].tolist())
        lig_rmsd_DEP_PRD = torch.tensor(df_out["click.lig_rmsd_DEP_PRD"].tolist())
        dist_CR1_NR1 = torch.tensor(df_out["click.dist_CR1_NR1"].tolist())
        dist_CR2_NR3 = torch.tensor(df_out["click.dist_CR2_NR3"].tolist())
        
        # Normalize metrics to [0,1] range
        def normalize_metric(metric, lb, ub):
            metric_clamped = torch.clamp(metric, min=lb, max=ub)
            reward = (metric_clamped - lb) / (ub - lb)
            return reward

        lig_rmsd_3AH_PRD_reward = 1 - normalize_metric(lig_rmsd_3AH_PRD, self.rmsd_3AH_lb, self.rmsd_3AH_ub)
        lig_rmsd_DEP_PRD_reward = 1 - normalize_metric(lig_rmsd_DEP_PRD, self.rmsd_DEP_lb, self.rmsd_DEP_ub)
        dist_CR1_NR1_reward = 1 - normalize_metric(dist_CR1_NR1, self.dist_CR1_NR1_lb, self.dist_CR1_NR1_ub)
        dist_CR2_NR3_reward = 1 - normalize_metric(dist_CR2_NR3, self.dist_CR2_NR3_lb, self.dist_CR2_NR3_ub)

        # get iptm reward
        iptm = torch.tensor(df_out["click.iptm"].tolist())
        iptm_reward = normalize_metric(iptm, self.iptm_lb, self.iptm_ub)
        ptm = torch.tensor(df_out["click.ptm"].tolist())
        ptm_reward = normalize_metric(ptm, self.ptm_lb, self.ptm_ub)

        # get esm perplexity reward
        esm_perplexity = torch.tensor(df_out["esm_perplexity.perplexity"].tolist())
        esm_perplexity_clamped = torch.clamp(esm_perplexity, min=self.esm_perplexity_lb, max=self.esm_perplexity_ub)
        esm_perplexity_reward = 1 - (esm_perplexity_clamped - self.esm_perplexity_lb) / (self.esm_perplexity_ub - self.esm_perplexity_lb)

        # Combine all rewards
        reward_list = [
            lig_rmsd_3AH_PRD_reward,
            lig_rmsd_DEP_PRD_reward,
            # dist_CR1_NR1_reward,
            # dist_CR2_NR3_reward,
            iptm_reward,
            ptm_reward,
            pairwise_diversity_reward,
            esm_perplexity_reward,
        ]

        # compute distance to reference sequence
        if self.ref_seq is not None:
            reward_list.append(dist_from_ref_seq_reward)

        reward = torch.stack(reward_list, dim=1).mean(dim=1)
        
        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "lig_rmsd_3AH_PRD": lig_rmsd_3AH_PRD.mean().cpu().item(),
            "lig_rmsd_3AH_PRD_min": lig_rmsd_3AH_PRD.min().cpu().item(),
            "lig_rmsd_3AH_PRD_max": lig_rmsd_3AH_PRD.max().cpu().item(),
            "lig_rmsd_DEP_PRD": lig_rmsd_DEP_PRD.mean().cpu().item(),
            "lig_rmsd_DEP_PRD_min": lig_rmsd_DEP_PRD.min().cpu().item(),
            "lig_rmsd_DEP_PRD_max": lig_rmsd_DEP_PRD.max().cpu().item(),
            "dist_CR1_NR1": dist_CR1_NR1.mean().cpu().item(),
            "dist_CR1_NR1_min": dist_CR1_NR1.min().cpu().item(),
            "dist_CR1_NR1_max": dist_CR1_NR1.max().cpu().item(),
            "dist_CR2_NR3": dist_CR2_NR3.mean().cpu().item(),
            "dist_CR2_NR3_min": dist_CR2_NR3.min().cpu().item(),
            "dist_CR2_NR3_max": dist_CR2_NR3.max().cpu().item(),
            "iptm_mean": iptm.mean().cpu().item(),
            "iptm_min": iptm.min().cpu().item(),
            "iptm_max": iptm.max().cpu().item(),
            "ptm_mean": ptm.mean().cpu().item(),
            "ptm_min": ptm.min().cpu().item(),
            "ptm_max": ptm.max().cpu().item(),
            "pairwise_diversity_mean": pairwise_diversity.mean().cpu().item(),
            "pairwise_diversity_min": pairwise_diversity.min().cpu().item(),
            "pairwise_diversity_max": pairwise_diversity.max().cpu().item(),
            "esm_perplexity_mean": esm_perplexity.mean().cpu().item(),
            "esm_perplexity_min": esm_perplexity.min().cpu().item(),
            "esm_perplexity_max": esm_perplexity.max().cpu().item(),
        }

        if self.ref_seq is not None:
            metrics["dist_from_ref_seq_mean"] = dist_from_ref_seq.mean().cpu().item()
            metrics["dist_from_ref_seq_min"] = dist_from_ref_seq.min().cpu().item()
            metrics["dist_from_ref_seq_max"] = dist_from_ref_seq.max().cpu().item()

        return reward.to(device), metrics


class AF3PETaseReward(Reward):
    """
        Click enzyme reward using AF3 pipeline
    """

    def __init__(
            self, 
            pipeline_config_path, 
            output_dir, 
            run_name,
            oxyanion_lb=2.25,
            oxyanion_ub=10,
            hisNesterox_lb=3.5,
            hisNesterox_ub=10,
            hisNserO_lb=2.25,
            hisNserO_ub=10,
            his_ser_angle=90,
            his_ser_angle_tol=5,
            iptm_lb=0.0,
            iptm_ub=0.8,
            ptm_lb=0.0,
            ptm_ub=0.8,
            ref_seq=None,
            max_dist_to_ref_seq=100,
            min_dist_to_ref_seq=0,
            frag_cfg=None, 
            max_pairwise_diversity=50,
            min_pairwise_diversity=0,
            pae_min_ub=10.0,
            pae_min_lb=0.5,
            as_plddt_ub=1.0,
            as_plddt_lb=0.0,
            esm_perplexity_ub=15.0,
            esm_perplexity_lb=6.0,
        ):

        sys.path.append("/projects/ml/itopt/policy_mpnn/software/pipelines")
        # if we used an apptainer we would could install cifutils and datahub directly
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/cifutils/src")
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/datahub/src")        
        os.environ["PYTHONPATH"] = ":" # pipeline will freak if this is not set

        from pipelines.pipeline import main as run_pipeline

        self.run_pipeline = run_pipeline
        self.pipeline_config = OmegaConf.load(pipeline_config_path)
        self.output_dir = output_dir
        self.run_name = run_name
        self.oxyanion_ub = oxyanion_ub
        self.oxyanion_lb = oxyanion_lb
        self.hisNesterox_ub = hisNesterox_ub
        self.hisNesterox_lb = hisNesterox_lb
        self.hisNserO_ub = hisNserO_ub
        self.hisNserO_lb = hisNserO_lb
        self.his_ser_angle = his_ser_angle
        self.his_ser_angle_tol = his_ser_angle_tol
        self.frag_cfg = frag_cfg
        self.iptm_ub = iptm_ub
        self.iptm_lb = iptm_lb
        self.ptm_ub = ptm_ub
        self.ptm_lb = ptm_lb
        self.ref_seq = ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.pae_min_ub = pae_min_ub
        self.pae_min_lb = pae_min_lb
        self.as_plddt_ub = as_plddt_ub
        self.as_plddt_lb = as_plddt_lb
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb

        # make subdirectory for pipeline output
        pipeline_output_dir = os.path.join(self.output_dir, self.run_name, "pipeline_output")
        os.makedirs(pipeline_output_dir, exist_ok=True)
    
    def get_sequences(self, policy_output, chain_mask=None):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]

        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    def get_input_df(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame(
            {
                "sequence": sequences,
                "name": [f"seq_{i:04}" for i in range(len(sequences))],
                "origin.path": [f"seq_{i:04}.path" for i in range(len(sequences))],
            }
        )

        return df

    def get_dist_to_ref_seq(self, sequences):
        assert self.ref_seq is not None, "Reference sequence is None"
        dist_list = []
        for seq in sequences:
            dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
            dist_list.append(float(dist))

        return torch.tensor(dist_list)
    
    def get_pairwise_diversity(self, sequences):
        dist_list = []
        for i in range(len(sequences)):
            _list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
            dist_list.append(np.mean(_list))

        # return mean pairwise distance
        return torch.tensor(dist_list)
    
    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}"
        )
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)

        if self.frag_cfg is not None:
            fragment_dict = make_fragment_dict(sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)
        df_out = self.run_pipeline(df_input, config)


        # add code to delete af3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)  # Clean up outputs to retry
        
        # get distance to reference sequence if provided
        if self.ref_seq is not None:
            dist_from_ref_seq = self.get_dist_to_ref_seq(df_out["sequence"].tolist()) 
            clampled_dist = torch.clamp(dist_from_ref_seq, min=self.min_dist_to_ref_seq, max=self.max_dist_to_ref_seq)
            norm_dist = (clampled_dist - self.min_dist_to_ref_seq) / (self.max_dist_to_ref_seq - self.min_dist_to_ref_seq + 1e-6)
            dist_from_ref_seq_reward = 1 - norm_dist

        # compute distance to other sequences in the batch to encourage diversity
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist())
        clamped_diversity = torch.clamp(pairwise_diversity, min=0, max=20)
        pairwise_diversity_reward = (clamped_diversity - self.min_pairwise_diversity) / (self.max_pairwise_diversity - self.min_pairwise_diversity + 1e-6)


        # Reward engineering
        acylox_oxh1bbN = torch.tensor(df_out["petase_metrics.acylox_oxh1bbN.mean"].tolist())
        acylox_oxh2bbN = torch.tensor(df_out["petase_metrics.acylox_oxh2bbN.mean"].tolist())
        hisNE2_esterox = torch.tensor(df_out["petase_metrics.hisNE2_esterox.mean"].tolist())
        hisNE2_serOG = torch.tensor(df_out["petase_metrics.hisNE2_serOG.mean"].tolist())

        # Normalize metrics to [0,1] range
        acylox_oxh1bbN_clamped = torch.clamp(acylox_oxh1bbN, min=self.oxyanion_lb, max=self.oxyanion_ub)
        acylox_oxh1bbN_reward = 1 - (acylox_oxh1bbN_clamped - self.oxyanion_lb) / (self.oxyanion_ub - self.oxyanion_lb)

        acylox_oxh2bbN_clamped = torch.clamp(acylox_oxh2bbN, min=self.oxyanion_lb, max=self.oxyanion_ub)
        acylox_oxh2bbN_reward = 1 - (acylox_oxh2bbN_clamped - self.oxyanion_lb) / (self.oxyanion_ub - self.oxyanion_lb)

        hisNE2_esterox_clamped = torch.clamp(hisNE2_esterox, min=self.hisNesterox_lb, max=self.hisNesterox_ub)
        hisNE2_esterox_reward = 1 - (hisNE2_esterox_clamped - self.hisNesterox_lb) / (self.hisNesterox_ub - self.hisNesterox_lb)

        hisNE2_serOG_clamped = torch.clamp(hisNE2_serOG, min=self.hisNserO_lb, max=self.hisNserO_ub)
        hisNE2_serOG_reward = 1 - (hisNE2_serOG_clamped - self.hisNserO_lb) / (self.hisNserO_ub - self.hisNserO_lb)

        # for angle want reward = 1 if within tolerance of target angle, else linear decay
        his_ser_angle_reward = []
        for a in df_out["petase_metrics.his_ser_angle.mean"].tolist():
            if a > self.his_ser_angle - self.his_ser_angle_tol and a < self.his_ser_angle + self.his_ser_angle_tol:
                his_ser_angle_reward.append(1.0)
            else:
                his_ser_angle_reward.append(1 - np.min(
                    [
                        np.abs(a - self.his_ser_angle - self.his_ser_angle_tol),
                        np.abs(a - self.his_ser_angle + self.his_ser_angle_tol)
                    ]) / 90) # max angle is 90 because we are mostly looking for ideal angle to be 90

        his_ser_angle_reward = torch.tensor(his_ser_angle_reward)

        # get iptm reward
        iptm = torch.tensor(df_out["petase_metrics.iptm.mean"].tolist())
        iptm_clamped = torch.clamp(iptm, min=self.iptm_lb, max=self.iptm_ub)
        iptm_reward = (iptm_clamped - self.iptm_lb) / (self.iptm_ub - self.iptm_lb)
        
        # instead use pae min
        pae_min = torch.tensor(df_out["petase_metrics.pae_min.mean"].tolist())
        pae_min_clamped = torch.clamp(pae_min, min=self.pae_min_lb, max=self.pae_min_ub)
        pae_min_reward = 1 - (pae_min_clamped - self.pae_min_lb) / (self.pae_min_ub - self.pae_min_lb)

        # get ptm reward
        ptm = torch.tensor(df_out["petase_metrics.ptm.mean"].tolist())
        ptm_clamped = torch.clamp(ptm, min=self.ptm_lb, max=self.ptm_ub)
        ptm_reward = (ptm_clamped - self.ptm_lb) / (self.ptm_ub - self.ptm_lb)

        # get active site plddt reward
        as_plddt = torch.tensor(df_out["petase_metrics.as_plddt.mean"].tolist())
        as_plddt_clamped = torch.clamp(as_plddt, min=self.as_plddt_lb, max=self.as_plddt_ub)
        as_plddt_reward = (as_plddt_clamped - self.as_plddt_lb) / (self.as_plddt_ub - self.as_plddt_lb)

        # get esm perplexity reward
        esm_perplexity = torch.tensor(df_out["esm_perplexity.perplexity"].tolist())
        esm_perplexity_clamped = torch.clamp(esm_perplexity, min=self.esm_perplexity_lb, max=self.esm_perplexity_ub)
        esm_perplexity_reward = 1 - (esm_perplexity_clamped - self.esm_perplexity_lb) / (self.esm_perplexity_ub - self.esm_perplexity_lb)

        # Combine all rewards
        reward_list = [
            acylox_oxh1bbN_reward,
            acylox_oxh2bbN_reward,
            hisNE2_esterox_reward,
            hisNE2_serOG_reward,
            his_ser_angle_reward,
            iptm_reward,
            ptm_reward,
            pae_min_reward,
            as_plddt_reward,
            pairwise_diversity_reward,
            esm_perplexity_reward
        ]
        
        if self.ref_seq is not None:
            reward_list.append(dist_from_ref_seq_reward)

        reward = torch.stack(reward_list, dim=1).mean(dim=1)

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "alcox_oxh1bbN_mean": acylox_oxh1bbN.mean().cpu().item(),
            "alcox_oxh1bbN_min": acylox_oxh1bbN.min().cpu().item(),
            "alcox_oxh1bbN_max": acylox_oxh1bbN.max().cpu().item(),
            "alcox_oxh2bbN_mean": acylox_oxh2bbN.mean().cpu().item(),
            "alcox_oxh2bbN_min": acylox_oxh2bbN.min().cpu().item(),
            "alcox_oxh2bbN_max": acylox_oxh2bbN.max().cpu().item(),
            "hisNE2_esterox_mean": hisNE2_esterox.mean().cpu().item(),
            "hisNE2_esterox_min": hisNE2_esterox.min().cpu().item(),
            "hisNE2_esterox_max": hisNE2_esterox.max().cpu().item(),
            "hisNE2_serOG_mean": hisNE2_serOG.mean().cpu().item(),
            "hisNE2_serOG_min": hisNE2_serOG.min().cpu().item(),
            "hisNE2_serOG_max": hisNE2_serOG.max().cpu().item(),
            "his_ser_angle_mean": his_ser_angle_reward.mean().cpu().item(),
            "his_ser_angle_min": his_ser_angle_reward.min().cpu().item(),
            "his_ser_angle_max": his_ser_angle_reward.max().cpu().item(),
            "iptm_mean": iptm.mean().cpu().item(),
            "iptm_min": iptm.min().cpu().item(),
            "iptm_max": iptm.max().cpu().item(),
            "ptm_mean": ptm.mean().cpu().item(),
            "ptm_min": ptm.min().cpu().item(),
            "ptm_max": ptm.max().cpu().item(),
            "pae_min_mean": pae_min.mean().cpu().item(),
            "pae_min_min": pae_min.min().cpu().item(),
            "pae_min_max": pae_min.max().cpu().item(),
            "as_plddt_mean": as_plddt.mean().cpu().item(),
            "as_plddt_min": as_plddt.min().cpu().item(),
            "as_plddt_max": as_plddt.max().cpu().item(),
            "pairwise_diversity_mean": pairwise_diversity.mean().cpu().item(),
            "pairwise_diversity_min": pairwise_diversity.min().cpu().item(),
            "pairwise_diversity_max": pairwise_diversity.max().cpu().item(),
            "esm_perplexity_mean": esm_perplexity.mean().cpu().item(),
            "esm_perplexity_min": esm_perplexity.min().cpu().item(),
            "esm_perplexity_max": esm_perplexity.max().cpu().item(),
        }

        if self.ref_seq is not None:
            metrics["dist_from_ref_seq_mean"] = dist_from_ref_seq.mean().cpu().item()
            metrics["dist_from_ref_seq_min"] = dist_from_ref_seq.min().cpu().item()
            metrics["dist_from_ref_seq_max"] = dist_from_ref_seq.max().cpu().item()

        return reward.to(device), metrics



class AF3ligaseReward(Reward):
    """
        Click enzyme reward using AF3 pipeline
    """

    def __init__(
            self, 
            pipeline_config_path, 
            output_dir, 
            run_name,
            oxyanion_lb=2.25,
            oxyanion_ub=10,
            hisNdcysSG_lb=3.5,
            hisNdcysSG_ub=10,
            hisNderO_lb=2.25,
            hisNderO_ub=10,
            his_cys_angle=90,
            his_cys_angle_tol=5,
            iptm_lb=0.0,
            iptm_ub=0.8,
            ptm_lb=0.0,
            ptm_ub=0.8,
            ref_seq=None,
            max_dist_to_ref_seq=100,
            min_dist_to_ref_seq=0,
            frag_cfg=None, 
            max_pairwise_diversity=50,
            min_pairwise_diversity=0,
            pae_min_ub=10.0,
            pae_min_lb=0.5,
            as_plddt_ub=1.0,
            as_plddt_lb=0.0,
            esm_perplexity_ub=15.0,
            esm_perplexity_lb=6.0,
        ):

        sys.path.append("/projects/ml/itopt/policy_mpnn/software/pipelines")
        # if we used an apptainer we would could install cifutils and datahub directly
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/cifutils/src")
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/datahub/src")        
        os.environ["PYTHONPATH"] = ":" # pipeline will freak if this is not set

        from pipelines.pipeline import main as run_pipeline

        self.run_pipeline = run_pipeline
        self.pipeline_config = OmegaConf.load(pipeline_config_path)
        self.output_dir = output_dir
        self.run_name = run_name
        self.oxyanion_ub = oxyanion_ub
        self.oxyanion_lb = oxyanion_lb
        self.hisNdsterox_ub = hisNdsterox_ub
        self.hisNdsterox_lb = hisNdsterox_lb
        self.hisNdcysSG_ub = hisNdcysSG_ub
        self.hisNdcysSG_lb = hisNdcysSG_lb
        self.his_cys_angle = his_cys_angle
        self.his_cys_angle_tol = his_cys_angle_tol
        self.frag_cfg = frag_cfg
        self.iptm_ub = iptm_ub
        self.iptm_lb = iptm_lb
        self.ptm_ub = ptm_ub
        self.ptm_lb = ptm_lb
        self.ref_seq = ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.pae_min_ub = pae_min_ub
        self.pae_min_lb = pae_min_lb
        self.as_plddt_ub = as_plddt_ub
        self.as_plddt_lb = as_plddt_lb
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb

        # make subdirectory for pipeline output
        pipeline_output_dir = os.path.join(self.output_dir, self.run_name, "pipeline_output")
        os.makedirs(pipeline_output_dir, exist_ok=True)
    
    def get_sequences(self, policy_output, chain_mask=None):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]

        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    def get_input_df(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame(
            {
                "sequence": sequences,
                "name": [f"seq_{i:04}" for i in range(len(sequences))],
                "origin.path": [f"seq_{i:04}.path" for i in range(len(sequences))],
            }
        )

        return df

    def get_dist_to_ref_seq(self, sequences):
        assert self.ref_seq is not None, "Reference sequence is None"
        dist_list = []
        for seq in sequences:
            dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
            dist_list.append(float(dist))

        return torch.tensor(dist_list)
    
    def get_pairwise_diversity(self, sequences):
        dist_list = []
        for i in range(len(sequences)):
            _list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
            dist_list.append(np.mean(_list))

        # return mean pairwise distance
        return torch.tensor(dist_list)
    
    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}"
        )
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)

        if self.frag_cfg is not None:
            fragment_dict = make_fragment_dict(sequences, self.frag_cfg.fragment_bounds)
            samples = sample_sequences(fragment_dict, self.frag_cfg.sample_size, self.frag_cfg.min_sample)
            names = [x[0] for x in samples]
            sequences = [x[1] for x in samples]
        
        # Create a DataFrame for AF3
        df_input = self.get_input_df(sequences)
        df_out = self.run_pipeline(df_input, config)


        # add code to delete af3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)  # Clean up outputs to retry
        
        # get distance to reference sequence if provided
        if self.ref_seq is not None:
            dist_from_ref_seq = self.get_dist_to_ref_seq(df_out["sequence"].tolist()) 
            clampled_dist = torch.clamp(dist_from_ref_seq, min=self.min_dist_to_ref_seq, max=self.max_dist_to_ref_seq)
            norm_dist = (clampled_dist - self.min_dist_to_ref_seq) / (self.max_dist_to_ref_seq - self.min_dist_to_ref_seq + 1e-6)
            dist_from_ref_seq_reward = 1 - norm_dist

        # compute distance to other sequences in the batch to encourage diversity
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist())
        clamped_diversity = torch.clamp(pairwise_diversity, min=0, max=20)
        pairwise_diversity_reward = (clamped_diversity - self.min_pairwise_diversity) / (self.max_pairwise_diversity - self.min_pairwise_diversity + 1e-6)


        # Reward engineering
        acylox_oxh1bbN = torch.tensor(df_out["petase_metrics.acylox_oxh1bbN.mean"].tolist())
        acylox_oxh2bbN = torch.tensor(df_out["petase_metrics.acylox_oxh2bbN.mean"].tolist())
        hisNE2_esterox = torch.tensor(df_out["petase_metrics.hisNE2_esterox.mean"].tolist())
        hisNE2_serOG = torch.tensor(df_out["petase_metrics.hisNE2_serOG.mean"].tolist())

        # Normalize metrics to [0,1] range
        acylox_oxh1bbN_clamped = torch.clamp(acylox_oxh1bbN, min=self.oxyanion_lb, max=self.oxyanion_ub)
        acylox_oxh1bbN_reward = 1 - (acylox_oxh1bbN_clamped - self.oxyanion_lb) / (self.oxyanion_ub - self.oxyanion_lb)

        acylox_oxh2bbN_clamped = torch.clamp(acylox_oxh2bbN, min=self.oxyanion_lb, max=self.oxyanion_ub)
        acylox_oxh2bbN_reward = 1 - (acylox_oxh2bbN_clamped - self.oxyanion_lb) / (self.oxyanion_ub - self.oxyanion_lb)

        hisNE2_esterox_clamped = torch.clamp(hisNE2_esterox, min=self.hisNesterox_lb, max=self.hisNesterox_ub)
        hisNE2_esterox_reward = 1 - (hisNE2_esterox_clamped - self.hisNesterox_lb) / (self.hisNesterox_ub - self.hisNesterox_lb)

        hisNE2_serOG_clamped = torch.clamp(hisNE2_serOG, min=self.hisNserO_lb, max=self.hisNserO_ub)
        hisNE2_serOG_reward = 1 - (hisNE2_serOG_clamped - self.hisNserO_lb) / (self.hisNserO_ub - self.hisNserO_lb)

        # for angle want reward = 1 if within tolerance of target angle, else linear decay
        his_ser_angle_reward = []
        for a in df_out["petase_metrics.his_ser_angle.mean"].tolist():
            if a > self.his_ser_angle - self.his_ser_angle_tol and a < self.his_ser_angle + self.his_ser_angle_tol:
                his_ser_angle_reward.append(1.0)
            else:
                his_ser_angle_reward.append(1 - np.min(
                    [
                        np.abs(a - self.his_ser_angle - self.his_ser_angle_tol),
                        np.abs(a - self.his_ser_angle + self.his_ser_angle_tol)
                    ]) / 90) # max angle is 90 because we are mostly looking for ideal angle to be 90

        his_ser_angle_reward = torch.tensor(his_ser_angle_reward)

        # get iptm reward
        iptm = torch.tensor(df_out["petase_metrics.iptm.mean"].tolist())
        iptm_clamped = torch.clamp(iptm, min=self.iptm_lb, max=self.iptm_ub)
        iptm_reward = (iptm_clamped - self.iptm_lb) / (self.iptm_ub - self.iptm_lb)
        
        # instead use pae min
        pae_min = torch.tensor(df_out["petase_metrics.pae_min.mean"].tolist())
        pae_min_clamped = torch.clamp(pae_min, min=self.pae_min_lb, max=self.pae_min_ub)
        pae_min_reward = 1 - (pae_min_clamped - self.pae_min_lb) / (self.pae_min_ub - self.pae_min_lb)

        # get ptm reward
        ptm = torch.tensor(df_out["petase_metrics.ptm.mean"].tolist())
        ptm_clamped = torch.clamp(ptm, min=self.ptm_lb, max=self.ptm_ub)
        ptm_reward = (ptm_clamped - self.ptm_lb) / (self.ptm_ub - self.ptm_lb)

        # get active site plddt reward
        as_plddt = torch.tensor(df_out["petase_metrics.as_plddt.mean"].tolist())
        as_plddt_clamped = torch.clamp(as_plddt, min=self.as_plddt_lb, max=self.as_plddt_ub)
        as_plddt_reward = (as_plddt_clamped - self.as_plddt_lb) / (self.as_plddt_ub - self.as_plddt_lb)

        # get esm perplexity reward
        esm_perplexity = torch.tensor(df_out["esm_perplexity.perplexity"].tolist())
        esm_perplexity_clamped = torch.clamp(esm_perplexity, min=self.esm_perplexity_lb, max=self.esm_perplexity_ub)
        esm_perplexity_reward = 1 - (esm_perplexity_clamped - self.esm_perplexity_lb) / (self.esm_perplexity_ub - self.esm_perplexity_lb)

        # Combine all rewards
        reward_list = [
            acylox_oxh1bbN_reward,
            acylox_oxh2bbN_reward,
            hisNE2_esterox_reward,
            hisNE2_serOG_reward,
            his_ser_angle_reward,
            iptm_reward,
            ptm_reward,
            pae_min_reward,
            as_plddt_reward,
            pairwise_diversity_reward,
            esm_perplexity_reward
        ]
        
        if self.ref_seq is not None:
            reward_list.append(dist_from_ref_seq_reward)

        reward = torch.stack(reward_list, dim=1).mean(dim=1)

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "alcox_oxh1bbN_mean": acylox_oxh1bbN.mean().cpu().item(),
            "alcox_oxh1bbN_min": acylox_oxh1bbN.min().cpu().item(),
            "alcox_oxh1bbN_max": acylox_oxh1bbN.max().cpu().item(),
            "alcox_oxh2bbN_mean": acylox_oxh2bbN.mean().cpu().item(),
            "alcox_oxh2bbN_min": acylox_oxh2bbN.min().cpu().item(),
            "alcox_oxh2bbN_max": acylox_oxh2bbN.max().cpu().item(),
            "hisNE2_esterox_mean": hisNE2_esterox.mean().cpu().item(),
            "hisNE2_esterox_min": hisNE2_esterox.min().cpu().item(),
            "hisNE2_esterox_max": hisNE2_esterox.max().cpu().item(),
            "hisNE2_serOG_mean": hisNE2_serOG.mean().cpu().item(),
            "hisNE2_serOG_min": hisNE2_serOG.min().cpu().item(),
            "hisNE2_serOG_max": hisNE2_serOG.max().cpu().item(),
            "his_ser_angle_mean": his_ser_angle_reward.mean().cpu().item(),
            "his_ser_angle_min": his_ser_angle_reward.min().cpu().item(),
            "his_ser_angle_max": his_ser_angle_reward.max().cpu().item(),
            "iptm_mean": iptm.mean().cpu().item(),
            "iptm_min": iptm.min().cpu().item(),
            "iptm_max": iptm.max().cpu().item(),
            "ptm_mean": ptm.mean().cpu().item(),
            "ptm_min": ptm.min().cpu().item(),
            "ptm_max": ptm.max().cpu().item(),
            "pae_min_mean": pae_min.mean().cpu().item(),
            "pae_min_min": pae_min.min().cpu().item(),
            "pae_min_max": pae_min.max().cpu().item(),
            "as_plddt_mean": as_plddt.mean().cpu().item(),
            "as_plddt_min": as_plddt.min().cpu().item(),
            "as_plddt_max": as_plddt.max().cpu().item(),
            "pairwise_diversity_mean": pairwise_diversity.mean().cpu().item(),
            "pairwise_diversity_min": pairwise_diversity.min().cpu().item(),
            "pairwise_diversity_max": pairwise_diversity.max().cpu().item(),
            "esm_perplexity_mean": esm_perplexity.mean().cpu().item(),
            "esm_perplexity_min": esm_perplexity.min().cpu().item(),
            "esm_perplexity_max": esm_perplexity.max().cpu().item(),
        }

        if self.ref_seq is not None:
            metrics["dist_from_ref_seq_mean"] = dist_from_ref_seq.mean().cpu().item()
            metrics["dist_from_ref_seq_min"] = dist_from_ref_seq.min().cpu().item()
            metrics["dist_from_ref_seq_max"] = dist_from_ref_seq.max().cpu().item()

        return reward.to(device), metrics


class lysine_ligase_pipeline_reward(AF3RMSDPipelineReward):
    """
        Lysine ligase pipeline reward, inheriting from AF3RMSDPipelineReward
    """
    def get_dist_to_ref_seq(self, sequences):
        assert self.ref_seq is not None, "Reference sequence is None"
        dist_list = []
        for seq in sequences:
            dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
            dist_list.append(float(dist))

        return torch.tensor(dist_list)
    
    def get_pairwise_diversity(self, sequences):
        dist_list = []
        for i in range(len(sequences)):
            _list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
            dist_list.append(np.mean(_list))

        # return mean pairwise distance
        return torch.tensor(dist_list)
    
    def __init__(self, 
                pipeline_config_path, 
                output_dir, 
                run_name, 
                rmsd_ub=10.0, 
                rmsd_lb=0.0, 
                max_pairwise_diversity=20, 
                min_pairwise_diversity=0, 
                esm_perplexity_ub=15.0, 
                esm_perplexity_lb=6.0,
                apo_motif_aa_rmsd_lb=0.0,
                apo_motif_aa_rmsd_ub=10.0,
                apo_ca_rmsd_lb=0.0,
                apo_ca_rmsd_ub=10.0,
                aei_motif_aa_rmsd_lb=0.0,
                aei_motif_aa_rmsd_ub=10.0,
                aei_ca_rmsd_lb=0.0,
                aei_ca_rmsd_ub=10.0,
                nu_his_dist_lb=0.0,
                nu_his_dist_ub=10.0,
                nu_cys_dist_lb=0.0,
                nu_cys_dist_ub=10.0,
                max_retries=3,
                frag_cfg=None, 
                ):
        """
            Initialize the Lysine ligase reward. Adds PAE-based piecewise-linear reward scaling in addition to RMSD.

            Args:
                pipeline_config_path (str): Path to the pipeline config YAML.
                output_dir (str): Directory where all outputs will be written.
                run_name (str): Name of the current run.
                rmsd_ub (float): Upper bound for RMSD scaling (same semantics as parent).
                rmsd_lb (float): Lower bound for RMSD scaling.
                pae_ub (float): Upper bound for PAE scaling (reward→0 when >= this value).
                pae_lb (float): Lower bound for PAE scaling (reward→1 when <= this value).
                frag_cfg: Optional fragment-based reward configuration.
                max_retries (int): Max retries for pipeline failures.
        """
        super().__init__(
            pipeline_config_path=pipeline_config_path,
            output_dir=output_dir,
            run_name=run_name,
            rmsd_ub=rmsd_ub,
            rmsd_lb=rmsd_lb,
            frag_cfg=frag_cfg,
            max_retries=max_retries,
        )

        # Store PAE bounds for use during reward calculation
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb
        
        # Lysine ligase specific RMSD and distance bounds
        self.apo_motif_aa_rmsd_lb = apo_motif_aa_rmsd_lb
        self.apo_motif_aa_rmsd_ub = apo_motif_aa_rmsd_ub
        self.apo_ca_rmsd_lb = apo_ca_rmsd_lb
        self.apo_ca_rmsd_ub = apo_ca_rmsd_ub
        self.aei_motif_aa_rmsd_lb = aei_motif_aa_rmsd_lb
        self.aei_motif_aa_rmsd_ub = aei_motif_aa_rmsd_ub
        self.aei_ca_rmsd_lb = aei_ca_rmsd_lb
        self.aei_ca_rmsd_ub = aei_ca_rmsd_ub
        self.nu_his_dist_lb = nu_his_dist_lb
        self.nu_his_dist_ub = nu_his_dist_ub
        self.nu_cys_dist_lb = nu_cys_dist_lb
        self.nu_cys_dist_ub = nu_cys_dist_ub

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):


        def compute_reward_low(values, lb, ub):
            # Piecewise linear RMSD reward using existing lb/ub parameters
            #   RMSD ≤ rmsd_lb  → reward = 1.0
            #   rmsd_lb < RMSD < rmsd_ub → linear decrease from 1.0 to 0.0
            #   RMSD ≥ rmsd_ub  → reward = 0.0
            values_clamped = torch.clamp(values, min=lb, max=ub)
            rmsd_reward = torch.where(
                values <= lb,
                torch.ones_like(values),
                1 - (values_clamped - lb) / (ub - lb),
            )
            return rmsd_reward

        def compute_reward_high(values, lb, ub):
            # Piecewise linear reward for higher-is-better metrics
            #   value ≥ ub  → reward = 1.0
            #   lb < value < ub → linear increase from 0.0 to 1.0
            #   value ≤ lb  → reward = 0.0
            values_clamped = torch.clamp(values, min=lb, max=ub)
            reward = torch.where(
                values >= ub,
                torch.ones_like(values),
                (values_clamped - lb) / (ub - lb),
            )
            return reward

        config = copy.deepcopy(self.pipeline_config)
        
        if evaluate:
            eval_idx = 0
            while os.path.exists(os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")):
                eval_idx += 1 # find the next available eval index for supporting multiple evals during training
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_eval_{eval_idx}_batch_{step:04}")
        else:
            config.rundir = os.path.join(self.output_dir, self.run_name, "pipeline_output", f"pipeline_output_iter_{step:04}")

        # Get the sequences from policy output
        orig_sequences = self.get_sequences(policy_output, chain_mask=None)
        # Create a DataFrame for AF3
        df_input = self.get_input_df(orig_sequences)

        # Run the pipeline
        df_out = self.run_pipeline(df_input, config)

        # Clean up AF3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)

        ### AF3 REWARD  ###
        apo_motif_aa_rmsd = torch.tensor(df_out["apo_metrics.motif_allatom_align.motif_allatom_rmsd"].tolist())
        apo_ca_rmsd = torch.tensor(df_out["apo_metrics.chain_A_ca_align.chain_A_ca_rmsd"].tolist())
        apo_ptm = torch.tensor(df_out["apo.af3_ptm"].tolist())
        reward_apo_motif_aa_rmsd = compute_reward_low(apo_motif_aa_rmsd, self.apo_motif_aa_rmsd_lb, self.apo_motif_aa_rmsd_ub)
        reward_apo_ca_rmsd = compute_reward_low(apo_ca_rmsd, self.apo_ca_rmsd_lb, self.apo_ca_rmsd_ub)
        apo_reward = apo_ptm * reward_apo_motif_aa_rmsd * reward_apo_ca_rmsd

        aei_motif_aa_rmsd = torch.tensor(df_out["aei_metrics.motif_allatom_align.motif_allatom_rmsd"].tolist())
        aei_ca_rmsd = torch.tensor(df_out["aei_metrics.chain_A_ca_align.chain_A_ca_rmsd"].tolist())
        aei_ptm = torch.tensor(df_out["aei.af3_iptm"].tolist())
        reward_aei_motif_aa_rmsd = compute_reward_low(aei_motif_aa_rmsd, self.aei_motif_aa_rmsd_lb, self.aei_motif_aa_rmsd_ub)
        reward_aei_ca_rmsd = compute_reward_low(aei_ca_rmsd, self.aei_ca_rmsd_lb, self.aei_ca_rmsd_ub)
        aei_reward = aei_ptm * reward_aei_motif_aa_rmsd * reward_aei_ca_rmsd  
        
        af3_reward = apo_reward * aei_reward

        ### GEOMETRY REWARD ###
        aei_nu_his_dist = torch.tensor(df_out["aei_metrics.geometry.nu_his_dist"].tolist())
        aei_nu_cys_dist = torch.tensor(df_out["aei_metrics.geometry.nu_cys_dist"].tolist())
        reward_nu_his = compute_reward_low(aei_nu_his_dist, self.nu_his_dist_lb, self.nu_his_dist_ub)
        reward_nu_cys = compute_reward_low(aei_nu_cys_dist, self.nu_cys_dist_lb, self.nu_cys_dist_ub)
        geometry_reward = reward_nu_his * reward_nu_cys

        ### AUX REWARDS ###
        # compute distance to other sequences in the batch to encourage diversity
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist())
        pairwise_diversity_clamped = torch.clamp(pairwise_diversity, min=self.min_pairwise_diversity, max=self.max_pairwise_diversity)
        pairwise_diversity_reward = (pairwise_diversity_clamped - self.min_pairwise_diversity) / (self.max_pairwise_diversity - self.min_pairwise_diversity + 1e-6)
        
        esm_perplexity = torch.tensor(df_out["esm_perplexity.perplexity"].tolist())
        esm_perplexity_clamped = torch.clamp(esm_perplexity, min=self.esm_perplexity_lb, max=self.esm_perplexity_ub)
        esm_perplexity_reward = 1 - (esm_perplexity_clamped - self.esm_perplexity_lb) / (self.esm_perplexity_ub - self.esm_perplexity_lb)
        
        #### TOTAL REWARD ###
        total_reward = af3_reward * geometry_reward * pairwise_diversity_reward * esm_perplexity_reward


        metrics = {
            "apo_motif_aa_rmsd_mean": apo_motif_aa_rmsd.mean().cpu().item(),
            "apo_motif_aa_rmsd_min": apo_motif_aa_rmsd.min().cpu().item(),
            "apo_motif_aa_rmsd_max": apo_motif_aa_rmsd.max().cpu().item(),
            "apo_ca_rmsd_mean": apo_ca_rmsd.mean().cpu().item(),
            "apo_ca_rmsd_min": apo_ca_rmsd.min().cpu().item(),
            "apo_ca_rmsd_max": apo_ca_rmsd.max().cpu().item(),
            "apo_ptm_mean": apo_ptm.mean().cpu().item(),
            "apo_ptm_min": apo_ptm.min().cpu().item(),
            "apo_ptm_max": apo_ptm.max().cpu().item(),
            "aei_motif_aa_rmsd_mean": aei_motif_aa_rmsd.mean().cpu().item(),
            "aei_motif_aa_rmsd_min": aei_motif_aa_rmsd.min().cpu().item(),
            "aei_motif_aa_rmsd_max": aei_motif_aa_rmsd.max().cpu().item(),
            "aei_ca_rmsd_mean": aei_ca_rmsd.mean().cpu().item(),
            "aei_ca_rmsd_min": aei_ca_rmsd.min().cpu().item(),
            "aei_ca_rmsd_max": aei_ca_rmsd.max().cpu().item(),
            "aei_ptm_mean": aei_ptm.mean().cpu().item(),
            "aei_ptm_min": aei_ptm.min().cpu().item(),
            "aei_ptm_max": aei_ptm.max().cpu().item(),
            "nu_his_dist_mean": aei_nu_his_dist.mean().cpu().item(),
            "nu_his_dist_min": aei_nu_his_dist.min().cpu().item(),
            "nu_his_dist_max": aei_nu_his_dist.max().cpu().item(),
            "nu_cys_dist_mean": aei_nu_cys_dist.mean().cpu().item(),
            "nu_cys_dist_min": aei_nu_cys_dist.min().cpu().item(),
            "nu_cys_dist_max": aei_nu_cys_dist.max().cpu().item(),
            "pairwise_diversity_mean": pairwise_diversity.mean().cpu().item(),
            "pairwise_diversity_min": pairwise_diversity.min().cpu().item(),
            "pairwise_diversity_max": pairwise_diversity.max().cpu().item(),
            "esm_perplexity_mean": esm_perplexity.mean().cpu().item(),
            "esm_perplexity_min": esm_perplexity.min().cpu().item(),
            "esm_perplexity_max": esm_perplexity.max().cpu().item(),
            "esm_perplexity_reward_mean": esm_perplexity_reward.mean().cpu().item(),
            "pairwise_diversity_reward_mean": pairwise_diversity_reward.mean().cpu().item(),
            "af3_reward_mean": af3_reward.mean().cpu().item(),
            "geometry_reward_mean": geometry_reward.mean().cpu().item(),
            "total_reward_mean": total_reward.mean().cpu().item(),
            "total_reward_min": total_reward.min().cpu().item(),
            "total_reward_max": total_reward.max().cpu().item(),
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            metrics['sampled_sequences'] = orig_sequences
            

        return total_reward.to(device), metrics

