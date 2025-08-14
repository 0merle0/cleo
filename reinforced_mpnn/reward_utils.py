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


class AF3PETaseReward(Reward):
    """
        Penicillin active site, using af3 RMSD as reward
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
        self.oxyanion_ub = oxyanion_ub
        self.oxyanion_lb = oxyanion_lb
        self.hisNesterox_ub = hisNesterox_ub
        self.hisNesterox_lb = hisNesterox_lb
        self.hisNserO_ub = hisNserO_ub
        self.hisNserO_lb = hisNserO_lb
        self.his_ser_angle = his_ser_angle
        self.his_ser_angle_tol = his_ser_angle_tol
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

        # Combine all rewards
        reward = (
            acylox_oxh1bbN_reward + 
            acylox_oxh2bbN_reward + 
            hisNE2_esterox_reward + 
            hisNE2_serOG_reward + 
            his_ser_angle_reward +
            iptm
        ) / 6. # take straight average of all the metrics

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
        }

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



