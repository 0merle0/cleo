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
        
        if self.frag_cfg is not None:
            sample_size = self.frag_cfg.sample_size if not evaluate else 32
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
        
        reward = torch.clamp(rmsd_reward1 * (pae_reward1 - pae_reward2), min=0.0, max=1.0)
        
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
            metrics['sampled_sequences'] = orig_sequences
            
            if self.frag_cfg is not None:
                metrics['recombined_sequences'] = sequences  # The recombined sequence strings
                metrics['recombined_rewards'] = recombined_rewards.cpu().numpy()  # Rewards before fragment aggregation
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used

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
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}" if not evaluate else f"pipeline_output_eval_{step:04}"
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

        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            metrics["rundir"] = config.rundir

            if self.frag_cfg is not None:
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used

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
            oxyanion_weight_1=1.0,
            oxyanion_weight_2=0.2, # smaller weight for oxh2 seems to help convergence with correct dock
            hisNesterox_lb=3.5,
            hisNesterox_ub=10,
            hisNesterox_weight=1.0,
            hisNserO_lb=2.25,
            hisNserO_ub=10,
            hisNserO_weight=1.0,
            his_ser_angle=90,
            his_ser_angle_tol=5,
            his_ser_angle_weight=1.0,
            iptm_lb=0.0,
            iptm_ub=0.8,
            iptm_weight=1.0,
            ptm_lb=0.0,
            ptm_ub=0.8,
            ptm_weight=1.0,
            ref_seq=None,
            max_dist_to_ref_seq=100,
            min_dist_to_ref_seq=0,
            dist_from_ref_seq_weight=1.0,
            frag_cfg=None, 
            max_pairwise_diversity=50,
            min_pairwise_diversity=0,
            pairwise_diversity_weight=1.0,
            pocket_idx_list=None,
            pocket_diversity_weight=1.0,
            pae_min_ub=10.0,
            pae_min_lb=0.5,
            pae_min_weight=1.0,
            as_plddt_ub=1.0,
            as_plddt_lb=0.0,
            as_plddt_weight=1.0,
            esm_perplexity_ub=15.0,
            esm_perplexity_lb=6.0,
            esm_perplexity_weight=1.0,
            af2_plddt_ub=80.0,
            af2_plddt_lb=0.0,
            af2_plddt_weight=1.0,
            ligand_rmsd_ub=20.0,
            ligand_rmsd_lb=0.0,
            ligand_rmsd_weight=1.0,
            reward_aggregation_mode="average",
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
        self.oxyanion_weight_1 = oxyanion_weight_1
        self.oxyanion_weight_2 = oxyanion_weight_2
        self.hisNesterox_ub = hisNesterox_ub
        self.hisNesterox_lb = hisNesterox_lb
        self.hisNesterox_weight = hisNesterox_weight
        self.hisNserO_ub = hisNserO_ub
        self.hisNserO_lb = hisNserO_lb
        self.hisNserO_weight = hisNserO_weight
        self.his_ser_angle = his_ser_angle
        self.his_ser_angle_tol = his_ser_angle_tol
        self.his_ser_angle_weight = his_ser_angle_weight
        self.frag_cfg = frag_cfg
        self.iptm_ub = iptm_ub
        self.iptm_lb = iptm_lb
        self.iptm_weight = iptm_weight
        self.ptm_ub = ptm_ub
        self.ptm_lb = ptm_lb
        self.ptm_weight = ptm_weight
        self.ref_seq = ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.dist_from_ref_seq_weight = dist_from_ref_seq_weight
        self.max_pairwise_diversity = max_pairwise_diversity
        self.min_pairwise_diversity = min_pairwise_diversity
        self.pairwise_diversity_weight = pairwise_diversity_weight
        self.pocket_idx_list = pocket_idx_list
        self.pocket_diversity_weight = pocket_diversity_weight
        self.pae_min_ub = pae_min_ub
        self.pae_min_lb = pae_min_lb
        self.pae_min_weight = pae_min_weight
        self.as_plddt_ub = as_plddt_ub
        self.as_plddt_lb = as_plddt_lb
        self.as_plddt_weight = as_plddt_weight
        self.esm_perplexity_ub = esm_perplexity_ub
        self.esm_perplexity_lb = esm_perplexity_lb
        self.esm_perplexity_weight = esm_perplexity_weight
        self.af2_plddt_ub = af2_plddt_ub
        self.af2_plddt_lb = af2_plddt_lb
        self.af2_plddt_weight = af2_plddt_weight
        self.ligand_rmsd_ub = ligand_rmsd_ub
        self.ligand_rmsd_lb = ligand_rmsd_lb
        self.ligand_rmsd_weight = ligand_rmsd_weight
        self.reward_aggregation_mode = reward_aggregation_mode

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
    
    def get_pairwise_diversity(self, sequences, pocket_idx_list=None):
        dist_list = []
        pocket_dist_list = []
        for i in range(len(sequences)):
            _list = []
            _pocket_list = []
            for j in range(len(sequences)):
                if i != j:
                    dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    _list.append(float(dist))
                    if pocket_idx_list is not None:
                        pocket_dist = sum(1 for idx in pocket_idx_list if sequences[i][idx] != sequences[j][idx])
                        _pocket_list.append(float(pocket_dist))

            dist_list.append(np.mean(_list))
            if pocket_idx_list is not None:
                pocket_dist_list.append(np.mean(_pocket_list))

        # return mean pairwise distance and pocket residue diversity if pocket_idx_list is provided
        if pocket_idx_list is not None:
            return torch.tensor(dist_list), torch.tensor(pocket_dist_list)
        else:
            return torch.tensor(dist_list)
    
    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}" if not evaluate else f"pipeline_output_eval_{step:04}"
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
        pairwise_diversity = self.get_pairwise_diversity(df_out["sequence"].tolist(), pocket_idx_list=self.pocket_idx_list)
        if self.pocket_idx_list is not None:
            pocket_diversity = pairwise_diversity[1]
            pairwise_diversity = pairwise_diversity[0]
            pocket_diversity_reward = pocket_diversity / len(self.pocket_idx_list) # want to maximize the diversity of each pocket to all other pockets

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

        # get ligand rmsd reward
        ligand_rmsd = torch.tensor(df_out["petase_metrics.protein_aligned_ligand_rmsd"].tolist())
        ligand_rmsd_clamped = torch.clamp(ligand_rmsd, min=self.ligand_rmsd_lb, max=self.ligand_rmsd_ub)
        ligand_rmsd_reward = 1 - (ligand_rmsd_clamped - self.ligand_rmsd_lb) / (self.ligand_rmsd_ub - self.ligand_rmsd_lb)


        # get esm perplexity reward
        if "esm_perplexity.perplexity" in df_out.columns:
            esm_perplexity = torch.tensor(df_out["esm_perplexity.perplexity"].tolist())
            esm_perplexity_clamped = torch.clamp(esm_perplexity, min=self.esm_perplexity_lb, max=self.esm_perplexity_ub)
            esm_perplexity_reward = 1 - (esm_perplexity_clamped - self.esm_perplexity_lb) / (self.esm_perplexity_ub - self.esm_perplexity_lb)
        else:
            esm_perplexity = torch.ones(len(sequences))
            esm_perplexity_reward = torch.ones(len(sequences))

        # get af2 plddt reward
        if "af2.af2_plddt" in df_out.columns:
            af2_plddt = torch.tensor(df_out["af2.af2_plddt"].tolist())
            af2_plddt_clamped = torch.clamp(af2_plddt, min=self.af2_plddt_lb, max=self.af2_plddt_ub)
            af2_plddt_reward = (af2_plddt_clamped - self.af2_plddt_lb) / (self.af2_plddt_ub - self.af2_plddt_lb)
        else:
            af2_plddt = torch.ones(len(sequences))
            af2_plddt_reward = torch.ones(len(sequences))

        
        # Combine all rewards
        if self.reward_aggregation_mode == "average":
            reward_list = [
                acylox_oxh1bbN_reward * self.oxyanion_weight_1,
                acylox_oxh2bbN_reward * self.oxyanion_weight_2,
                hisNE2_esterox_reward * self.hisNesterox_weight,
                hisNE2_serOG_reward * self.hisNserO_weight,
                his_ser_angle_reward * self.his_ser_angle_weight,
                iptm_reward * self.iptm_weight,
                ptm_reward * self.ptm_weight,
                pae_min_reward * self.pae_min_weight,
                as_plddt_reward * self.as_plddt_weight,
                pairwise_diversity_reward * self.pairwise_diversity_weight,
                esm_perplexity_reward * self.esm_perplexity_weight,
                af2_plddt_reward * self.af2_plddt_weight,
                ligand_rmsd_reward * self.ligand_rmsd_weight,
            ]
            
            denom = sum([
                self.oxyanion_weight_1, self.oxyanion_weight_2, self.hisNesterox_weight, self.hisNserO_weight,
                self.his_ser_angle_weight, self.iptm_weight, self.ptm_weight, self.pae_min_weight,
                self.as_plddt_weight, self.pairwise_diversity_weight, self.esm_perplexity_weight,
                self.af2_plddt_weight, self.ligand_rmsd_weight,
            ])

            if self.ref_seq is not None:
                reward_list.append(dist_from_ref_seq_reward * self.dist_from_ref_seq_weight)
                denom += self.dist_from_ref_seq_weight

            if self.pocket_idx_list is not None:
                reward_list.append(pocket_diversity_reward * self.pocket_diversity_weight)
                denom += self.pocket_diversity_weight

            reward = torch.stack(reward_list, dim=1).sum(dim=1) / denom

        elif self.reward_aggregation_mode == "custom":
            # trying out segregating metrics into str, conf, library and then multiplying
            reward_list = [
                acylox_oxh1bbN_reward * self.oxyanion_weight_1,
                acylox_oxh2bbN_reward * self.oxyanion_weight_2,
                iptm_reward * self.iptm_weight,
                af2_plddt_reward * self.af2_plddt_weight,
                hisNE2_esterox_reward * self.hisNesterox_weight,
                esm_perplexity_reward * self.esm_perplexity_weight,
                his_ser_angle_reward * self.his_ser_angle_weight,
            ]

            if self.ref_seq is not None:
                reward_list.append(dist_from_ref_seq_reward * self.dist_from_ref_seq_weight)

            if self.pocket_idx_list is not None:
                reward_list.append(pocket_diversity_reward * self.pocket_diversity_weight)

            reward = torch.stack(reward_list, dim=1).mean(dim=1)

        else:
            raise ValueError(f"Unknown reward aggregation mode: {self.reward_aggregation_mode}")
       
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
            "af2_plddt_mean": af2_plddt.mean().cpu().item(),
            "af2_plddt_min": af2_plddt.min().cpu().item(),
            "af2_plddt_max": af2_plddt.max().cpu().item(),
            "ligand_rmsd_mean": ligand_rmsd.mean().cpu().item(),
            "ligand_rmsd_min": ligand_rmsd.min().cpu().item(),
            "ligand_rmsd_max": ligand_rmsd.max().cpu().item(),
        }

        if self.ref_seq is not None:
            metrics["dist_from_ref_seq_mean"] = dist_from_ref_seq.mean().cpu().item()
            metrics["dist_from_ref_seq_min"] = dist_from_ref_seq.min().cpu().item()
            metrics["dist_from_ref_seq_max"] = dist_from_ref_seq.max().cpu().item()

        if self.pocket_idx_list is not None:
            metrics["pocket_diversity_mean"] = pocket_diversity.mean().cpu().item()
            metrics["pocket_diversity_min"] = pocket_diversity.min().cpu().item()
            metrics["pocket_diversity_max"] = pocket_diversity.max().cpu().item()

        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            metrics["rundir"] = config.rundir

            if self.frag_cfg is not None:
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used

        return reward.to(device), metrics

class LigasePipelineReward(Reward):
    """
        Penicillin active site, using af3 RMSD as reward
    """

    def __init__(self, pipeline_config_path, output_dir, rmsd_ub=10.0, rmsd_lb=0.0, frag_cfg=None):

        # sys.path.append("/projects/ml/itopt/policy_mpnn/software/pipelines")
        # # if we used an apptainer we would could install cifutils and datahub directly
        # sys.path.append("/projects/ml/itopt/policy_mpnn/software/cifutils/src")
        # sys.path.append("/projects/ml/itopt/policy_mpnn/software/datahub/src")        
        sys.path.append("/home/ssalike/git/pipelines")
        # if we used an apptainer we would could install cifutils and datahub directly
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/cifutils/src")
        sys.path.append("/projects/ml/itopt/policy_mpnn/software/datahub/src")      
        os.environ["PYTHONPATH"] = ":" # pipeline will freak if this is not set

        from pipelines.pipeline import main as run_pipeline

        self.run_pipeline = run_pipeline
        self.pipeline_config = OmegaConf.load(pipeline_config_path)
        self.output_dir = output_dir
        self.rmsd_ub = rmsd_ub
        self.rmsd_lb = rmsd_lb
        self.frag_cfg = frag_cfg

        # make subdirectory for pipeline output
        pipeline_output_dir = os.path.join(self.output_dir, "pipeline_output")
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
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}"
        )

        # Get the chain mask if available
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
        

        # Run the pipeline
        df_out = self.run_pipeline(df_input, config)

        # Reward shaping
        as_rmsd1 = torch.tensor(df_out["af3_ts2_hyd_alignment.motif_allatom_align.motif_allatom_rmsd"].tolist())
        as_rmsd2 = torch.tensor(df_out["af3_ts2_amine_alignment.motif_allatom_align.motif_allatom_rmsd"].tolist())
        
        # Clamp each RMSD individually
        as_rmsd1_clamped = torch.clamp(as_rmsd1, min=self.rmsd_lb, max=self.rmsd_ub)
        as_rmsd2_clamped = torch.clamp(as_rmsd2, min=self.rmsd_lb, max=self.rmsd_ub)
        
        # Normalize to [0,1] - higher as_rmsd1 is better, lower as_rmsd2 is better
        rmsd1_reward = (as_rmsd1_clamped - self.rmsd_lb) / (self.rmsd_ub - self.rmsd_lb)  # Higher is better
        rmsd2_reward = 1 - (as_rmsd2_clamped - self.rmsd_lb) / (self.rmsd_ub - self.rmsd_lb)  # Lower is better
        
        # Combine the rewards (multiply to require both to be good)
        rmsd_reward = rmsd1_reward * rmsd2_reward
        
        ptm = torch.tensor(df_out["af3_ptm"].tolist())

        # Combine rmsd and ptm rewards
        reward = rmsd_reward * ptm

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            padding = torch.ones(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        as_rmsd = as_rmsd1 - as_rmsd2  # Keep for metrics
        metrics = {
            "rmsd_diff_mean": as_rmsd.mean().cpu().item(),
            "rmsd_diff_min": as_rmsd.min().cpu().item(),
            "rmsd_diff_max": as_rmsd.max().cpu().item(),
            "rmsd1_mean": as_rmsd1.mean().cpu().item(),
            "rmsd1_min": as_rmsd1.min().cpu().item(),
            "rmsd1_max": as_rmsd1.max().cpu().item(),
            "rmsd2_mean": as_rmsd2.mean().cpu().item(),
            "rmsd2_min": as_rmsd2.min().cpu().item(),
            "rmsd2_max": as_rmsd2.max().cpu().item(),
            "ptm_mean": ptm.mean().cpu().item(),
            "ptm_min": ptm.min().cpu().item(),
            "ptm_max": ptm.max().cpu().item(),
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            metrics['policy_log_probs'] = policy_output.get('log_probs', None)

        return reward.to(device), metrics


class ATPBinder(Reward):
    """
        ATP binder, increase num hbonds, and iptm
    """

    def __init__(
            self, 
            pipeline_config_path, 
            output_dir, 
            run_name,
            iptm_lb = 0.0,
            iptm_ub = 0.9,
            hbond_lb = 0,
            hbond_ub = 15,
            frag_cfg=None, 
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
        self.iptm_ub = iptm_ub
        self.iptm_lb = iptm_lb
        self.hbond_ub = hbond_ub
        self.hbond_lb = hbond_lb
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
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}" if not evaluate else f"pipeline_output_eval_{step:04}"
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
        
        # Normalize metrics to [0,1] range
        def normalize_metric(metric, lb, ub):
            metric_clamped = torch.clamp(metric, min=lb, max=ub)
            reward = (metric_clamped - lb) / (ub - lb)
            return reward

        # get iptm reward
        iptm = torch.tensor(df_out["af3_iptm"].tolist())
        iptm_reward = normalize_metric(iptm, self.iptm_lb, self.iptm_ub)

        # get hbond reward
        hbond = torch.tensor(df_out["hbond.hbond_count"].tolist()).float()
        hbond_reward = normalize_metric(hbond, self.hbond_lb, self.hbond_ub)

        # Combine all rewards
        reward_list = [
            iptm_reward,
            hbond_reward,
        ]

        reward = torch.stack(reward_list, dim=1).mean(dim=1)

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            "iptm_mean": iptm.mean().cpu().item(),
            "iptm_min": iptm.min().cpu().item(),
            "iptm_max": iptm.max().cpu().item(),
            "hbond_mean": hbond.mean().cpu().item(),
            "hbond_min": hbond.min().cpu().item(),
            "hbond_max": hbond.max().cpu().item(),
        }

        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            metrics["rundir"] = config.rundir

            if self.frag_cfg is not None:
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used

        return reward.to(device), metrics


class MetalloPETase(Reward):
    """
        MetalloPETase specific reward function
    """

    def __init__(
            self, 
            pipeline_config_path, 
            output_dir, 
            run_name,
            iptm_lb = 0.0,
            iptm_ub = 0.9,
            iptm_weight = 1.0,
            ligand_rmsd_lb = 0,
            ligand_rmsd_ub = 10,
            ligand_rmsd_weight = 1.0,
            as_rmsd_lb = 0,
            as_rmsd_ub = 10,
            as_rmsd_weight = 1.0,
            as_plddt_lb = 50,
            as_plddt_ub = 90,
            as_plddt_weight = 1.0,
            ref_seq = None,
            dist_from_ref_seq_weight = 0.0,
            min_dist_to_ref_seq = 5,
            max_dist_to_ref_seq = 100,
            frag_cfg=None, 
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
        self.iptm_ub = iptm_ub
        self.iptm_lb = iptm_lb
        self.iptm_weight = iptm_weight
        self.ligand_rmsd_ub = ligand_rmsd_ub
        self.ligand_rmsd_lb = ligand_rmsd_lb
        self.ligand_rmsd_weight = ligand_rmsd_weight
        self.as_rmsd_ub = as_rmsd_ub
        self.as_rmsd_lb = as_rmsd_lb
        self.as_rmsd_weight = as_rmsd_weight
        self.as_plddt_ub = as_plddt_ub
        self.as_plddt_lb = as_plddt_lb
        self.as_plddt_weight = as_plddt_weight
        self.ref_seq = ref_seq
        self.dist_from_ref_seq_weight = dist_from_ref_seq_weight
        self.min_dist_to_ref_seq = min_dist_to_ref_seq
        self.max_dist_to_ref_seq = max_dist_to_ref_seq
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
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):

        config = copy.deepcopy(self.pipeline_config)
        config.rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "pipeline_output", 
            f"pipeline_output_iter_{step:04}" if not evaluate else f"pipeline_output_eval_{step:04}"
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

        print("Pipeline output df columns:", df_out.columns)


        # add code to delete af3 outputs to save space
        af3_out_dir = os.path.join(config.rundir, "af3*/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)  # Clean up outputs to retry
        
        # Normalize metrics to [0,1] range
        def normalize_metric(metric, lb, ub):
            metric_clamped = torch.clamp(metric, min=lb, max=ub)
            reward = (metric_clamped - lb) / (ub - lb)
            return reward

        # for SUBSTRATE
        # get iptm reward
        sub_iptm = torch.tensor(df_out["af3_sub_metrics.iptm"].tolist())
        sub_iptm_reward = normalize_metric(sub_iptm, self.iptm_lb, self.iptm_ub)

        # get ligand_rmsd reward
        sub_ligand_rmsd = torch.tensor(df_out["af3_sub_metrics.ligand_rmsd"].tolist()).float()
        sub_ligand_rmsd_reward = 1 - normalize_metric(sub_ligand_rmsd, self.ligand_rmsd_lb, self.ligand_rmsd_ub)

        # get as_plddt reward
        sub_as_plddt = torch.tensor(df_out["af3_sub_metrics.as_plddt"].tolist()).float()
        sub_as_plddt_reward = normalize_metric(sub_as_plddt, self.as_plddt_lb, self.as_plddt_ub)

        # get as_rmsd reward
        sub_as_rmsd = torch.tensor(df_out["af3_sub_metrics.as_rmsd"].tolist()).float()
        sub_as_rmsd_reward = 1 - normalize_metric(sub_as_rmsd, self.as_rmsd_lb, self.as_rmsd_ub)

        # for TSA
        # get iptm reward
        tsa_iptm = torch.tensor(df_out["af3_tsa_metrics.iptm"].tolist())
        tsa_iptm_reward = normalize_metric(tsa_iptm, self.iptm_lb, self.iptm_ub)

        # get ligand_rmsd reward
        tsa_ligand_rmsd = torch.tensor(df_out["af3_tsa_metrics.ligand_rmsd"].tolist()).float()
        tsa_ligand_rmsd_reward = 1 - normalize_metric(tsa_ligand_rmsd, self.ligand_rmsd_lb, self.ligand_rmsd_ub)

        # get as_plddt reward
        tsa_as_plddt = torch.tensor(df_out["af3_tsa_metrics.as_plddt"].tolist()).float()
        tsa_as_plddt_reward = normalize_metric(tsa_as_plddt, self.as_plddt_lb, self.as_plddt_ub)

        # get as_rmsd reward
        tsa_as_rmsd = torch.tensor(df_out["af3_tsa_metrics.as_rmsd"].tolist()).float()
        tsa_as_rmsd_reward = 1 - normalize_metric(tsa_as_rmsd, self.as_rmsd_lb, self.as_rmsd_ub)

        # get distance to reference sequence
        if self.ref_seq is not None:
            dist_from_ref_seq = self.get_dist_to_ref_seq(df_out["sequence"].tolist()) 
            clampled_dist = torch.clamp(dist_from_ref_seq, min=self.min_dist_to_ref_seq, max=self.max_dist_to_ref_seq)
            norm_dist = (clampled_dist - self.min_dist_to_ref_seq) / (self.max_dist_to_ref_seq - self.min_dist_to_ref_seq + 1e-6)
            dist_from_ref_seq_reward = 1 - norm_dist


        # Combine all rewards
        reward_list = [
            sub_iptm_reward * self.iptm_weight,
            sub_ligand_rmsd_reward * self.ligand_rmsd_weight,
            sub_as_plddt_reward * self.as_plddt_weight,
            sub_as_rmsd_reward * self.as_rmsd_weight,
            tsa_iptm_reward * self.iptm_weight,
            tsa_ligand_rmsd_reward * self.ligand_rmsd_weight,
            tsa_as_plddt_reward * self.as_plddt_weight,
            tsa_as_rmsd_reward * self.as_rmsd_weight,
        ]

        denom = sum([self.iptm_weight, self.ligand_rmsd_weight, self.as_plddt_weight, self.as_rmsd_weight]) * 2  # times 2 for sub and tsa

        if self.ref_seq is not None:
            reward_list.append(dist_from_ref_seq_reward * self.dist_from_ref_seq_weight)
            denom += self.dist_from_ref_seq_weight

        reward = torch.stack(reward_list, dim=1) / denom

        if self.frag_cfg is not None:
            reward = get_fragment_rewards(sequences, reward, fragment_dict, self.frag_cfg.fragment_bounds)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            # should the padding be zero or ones? ( i think zero is better )
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        metrics = {
            # for SUBSTRATE
            "sub_iptm_mean": sub_iptm.mean().cpu().item(),
            "sub_iptm_min": sub_iptm.min().cpu().item(),
            "sub_iptm_max": sub_iptm.max().cpu().item(),
            "sub_ligand_rmsd_mean": sub_ligand_rmsd.mean().cpu().item(),
            "sub_ligand_rmsd_min": sub_ligand_rmsd.min().cpu().item(),
            "sub_ligand_rmsd_max": sub_ligand_rmsd.max().cpu().item(),
            "sub_as_plddt_mean": sub_as_plddt.mean().cpu().item(),
            "sub_as_plddt_min": sub_as_plddt.min().cpu().item(),
            "sub_as_plddt_max": sub_as_plddt.max().cpu().item(),
            "sub_as_rmsd_mean": sub_as_rmsd.mean().cpu().item(),
            "sub_as_rmsd_min": sub_as_rmsd.min().cpu().item(),
            "sub_as_rmsd_max": sub_as_rmsd.max().cpu().item(),
            "sub_prot_iptm": np.mean(df_out["af3_sub_metrics.prot_iptm"].tolist()),
            "sub_prot_iptm_min": np.min(df_out["af3_sub_metrics.prot_iptm"].tolist()),
            "sub_prot_iptm_max": np.max(df_out["af3_sub_metrics.prot_iptm"].tolist()),
            "sub_zn_iptm": np.mean(df_out["af3_sub_metrics.zn_iptm"].tolist()),
            "sub_zn_iptm_min": np.min(df_out["af3_sub_metrics.zn_iptm"].tolist()),
            "sub_zn_iptm_max": np.max(df_out["af3_sub_metrics.zn_iptm"].tolist()),
            "sub_ligand_iptm": np.mean(df_out["af3_sub_metrics.ligand_iptm"].tolist()),
            "sub_ligand_iptm_min": np.min(df_out["af3_sub_metrics.ligand_iptm"].tolist()),
            "sub_ligand_iptm_max": np.max(df_out["af3_sub_metrics.ligand_iptm"].tolist()),

            # for TSA
            "tsa_iptm_mean": tsa_iptm.mean().cpu().item(),
            "tsa_iptm_min": tsa_iptm.min().cpu().item(),
            "tsa_iptm_max": tsa_iptm.max().cpu().item(),
            "tsa_ligand_rmsd_mean": tsa_ligand_rmsd.mean().cpu().item(),
            "tsa_ligand_rmsd_min": tsa_ligand_rmsd.min().cpu().item(),
            "tsa_ligand_rmsd_max": tsa_ligand_rmsd.max().cpu().item(),
            "tsa_as_plddt_mean": tsa_as_plddt.mean().cpu().item(),
            "tsa_as_plddt_min": tsa_as_plddt.min().cpu().item(),
            "tsa_as_plddt_max": tsa_as_plddt.max().cpu().item(),
            "tsa_as_rmsd_mean": tsa_as_rmsd.mean().cpu().item(),
            "tsa_as_rmsd_min": tsa_as_rmsd.min().cpu().item(),
            "tsa_as_rmsd_max": tsa_as_rmsd.max().cpu().item(),
            "tsa_prot_iptm": np.mean(df_out["af3_tsa_metrics.prot_iptm"].tolist()),
            "tsa_prot_iptm_min": np.min(df_out["af3_tsa_metrics.prot_iptm"].tolist()),
            "tsa_prot_iptm_max": np.max(df_out["af3_tsa_metrics.prot_iptm"].tolist()),
            "tsa_zn_iptm": np.mean(df_out["af3_tsa_metrics.zn_iptm"].tolist()),
            "tsa_zn_iptm_min": np.min(df_out["af3_tsa_metrics.zn_iptm"].tolist()),
            "tsa_zn_iptm_max": np.max(df_out["af3_tsa_metrics.zn_iptm"].tolist()),
            "tsa_ligand_iptm": np.mean(df_out["af3_tsa_metrics.ligand_iptm"].tolist()),
            "tsa_ligand_iptm_min": np.min(df_out["af3_tsa_metrics.ligand_iptm"].tolist()),
            "tsa_ligand_iptm_max": np.max(df_out["af3_tsa_metrics.ligand_iptm"].tolist()),
        }

        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            # Store policy log probabilities for evaluation
            metrics["rundir"] = config.rundir

            if self.frag_cfg is not None:
                metrics['fragment_dict'] = fragment_dict  # Fragment dictionary
                metrics['fragment_bounds'] = self.frag_cfg.fragment_bounds  # Fragment bounds used

        return reward.to(device), metrics

# AF3 reward pre-pipeline
class af3_reward(Reward):
    """
        Reward based on AlphaFold3 score
        
        Uses configurable thresholds to map metric values to the [0,1] range.
        For metrics like RMSD where lower is better.
    """
    def __init__(self, output_dir, af3_config, metric_name="rmsd", confidence_normalize_metric=None,lower_threshold=0.25, upper_threshold=3.0, normalization_type="linear"):
        """
        Initialize AF3 reward function
        
        Args:
            af3_config: Configuration for AF3 inference
            metric_name: Column name to use for the reward calculation (default: "rmsd")
            lower_threshold: Values below this get reward 1.0
            upper_threshold: Values above this get reward 0.0
        """
        self.af3_config = af3_config
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.normalization_type = normalization_type
        self.confidence_normalize_metric = confidence_normalize_metric 

        if normalization_type == "exponential":
            print(f"Using exponential reward normalization with lower_threshold: {lower_threshold} and upper_threshold: {upper_threshold}")
            # Calculate the steepness to achieve min reward at upper_threshold
            self.exp_steepness = -np.log(lower_threshold)
        
    def sequences_to_dataframe(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame({
            'name': [f'seq_{i:04}' for i in range(len(sequences))],
            'seq': sequences
        })
        return df
        
    def normalize_metric(self, values):
        """
        Normalize metric values to [0,1] range based on thresholds
        For metrics like RMSD where lower is better
        """
        normalized = np.ones_like(values, dtype=np.float32)
        
        # Values above upper_threshold get 0.0
        above_upper = values > self.upper_threshold
        normalized[above_upper] = 0.0
        
        # Values between thresholds get scaled linearly
        between = (values >= self.lower_threshold) & (values <= self.upper_threshold)
        if np.any(between):
            if self.normalization_type == "linear":
                # Linear scaling - same as before
                range_size = self.upper_threshold - self.lower_threshold
                normalized[between] = 1.0 - (values[between] - self.lower_threshold) / range_size
            elif self.normalization_type == "exponential":
                # Exponential scaling
                normalized_values = (values[between] - self.lower_threshold) / (self.upper_threshold - self.lower_threshold)
                # Apply exponential decay: e^(-k*x) - gives 1.0 at lower and 0 at upper threshold
                normalized[between] = (np.exp(-self.exp_steepness * normalized_values) - np.exp(-self.exp_steepness)) / (1 - np.exp(-self.exp_steepness))
            
        return normalized
        
    def __call__(self, step, policy_output, feature_dict, device, evaluate=False):
        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output)
        
        # Create a DataFrame for AF3
        input_df = self.sequences_to_dataframe(sequences)
        # need to update the datadir based on the current iteration
        iter = policy_output["iter"] if "iter" in policy_output else 0
        self.af3_config.datadir = os.path.join(self.output_dir, f"af3_outputs/iter_{step}")
        # Create a temporary directory for AF3 outputs if needed
        os.makedirs(self.af3_config.datadir, exist_ok=True)
        
        # Run AF3 inference
        result_df = af3_main(self.af3_config, input_df=input_df)
        # Look for the exact metric name in result columns
        if self.metric_name in result_df.columns:
            metric_values = result_df[self.metric_name].values
        else:
            # If the exact name is not found, look for it with state prefixes
            state_columns = [col for col in result_df.columns if col.endswith(f"_{self.metric_name}")]
            if state_columns:
                metric_values = result_df[state_columns[0]].values
            else:
                # If not found at all, return zeros
                print(f"Warning: Metric '{self.metric_name}' not found in AF3 results")
                print(f"Available columns: {result_df.columns.tolist()}")
                metric_values = np.zeros(len(sequences))
        
        # Normalize to [0,1] range
        normalized_scores = self.normalize_metric(metric_values)

        if self.confidence_normalize_metric:
            conf_values = result_df[self.confidence_normalize_metric].values
            # pdb_lib.set_trace()
            normalized_scores = normalized_scores * conf_values

        input_df['reward'] = normalized_scores
        # save a csv to self.af3_config.datadir
        input_df.to_csv(os.path.join(self.af3_config.datadir, f"af3_metrics.csv"), index=False)
        # Convert to tensor and return
        reward = torch.tensor(normalized_scores, dtype=torch.float32)
        # convert raw rmsds to a dictionary
        metrics = {
            f"{self.metric_name}_mean": result_df[self.metric_name].values.mean(),
            f"{self.metric_name}_std": result_df[self.metric_name].values.std(),
            f"{self.metric_name}_min": result_df[self.metric_name].values.min(),
            f"{self.metric_name}_max": result_df[self.metric_name].values.max(),
            f"{self.confidence_normalize_metric}_mean": conf_values.mean() if self.confidence_normalize_metric else None,
            f"{self.confidence_normalize_metric}_std": conf_values.std() if self.confidence_normalize_metric else None,
            f"{self.confidence_normalize_metric}_min": conf_values.min() if self.confidence_normalize_metric else None,
            f"{self.confidence_normalize_metric}_max": conf_values.max() if self.confidence_normalize_metric else None
        }
        
        # Add evaluation-specific data when in evaluation mode
        if evaluate:
            metrics['policy_log_probs'] = policy_output.get('log_probs', None)

        return reward.to(device), metrics



