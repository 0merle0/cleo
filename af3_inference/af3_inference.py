#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
"""
Streamlined AlphaFold3 inference pipeline.
Takes a CSV with design information and runs AF3 predictions for multiple states.
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import time
import subprocess
from typing import Dict, List, Optional, Any
import shutil

# Import utility functions for Slurm job submission
from util import slurm_tools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main_af3(input_df, cfg: HydraConfig) -> List[int]:
    """
    Main function to orchestrate the preprocessing and prediction pipeline.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        List of Slurm job IDs submitted
    """
    # Create output directories
    datadir = Path(cfg.datadir)
        
    # Set default prefix if not provided
    prefix = cfg.get('prefix', 'af3')
    suffix = cfg.get('suffix', '')
    prefix = f'{prefix}_{suffix}' if suffix else prefix
    
    # Set chunk size 
    chunk_size = len(input_df) if cfg.chunk == -1 else cfg.chunk
    
    # Process ligand data
    ligand_data = json.dumps(OmegaConf.to_container(cfg.ligand_data)) if hasattr(cfg, 'ligand_data') and cfg.ligand_data else None
    bonds_json = json.dumps(OmegaConf.to_container(cfg.bonds)) if hasattr(cfg, 'bonds') and cfg.bonds else None
    ptm_json = json.dumps(OmegaConf.to_container(cfg.ptm_json)) if hasattr(cfg, 'ptm_json') and cfg.ptm_json else None

    # Create output directories
    json_subdir = datadir / f"{prefix}_json"
    output_dir = datadir / f"{prefix}_outputs"
    os.makedirs(json_subdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths from configuration
    paths = _get_paths(cfg)
    
    # Create job list files
    preproc_job_fn = datadir / f'jobs_{prefix}.preprocess.list'
    pred_job_fn = datadir / f'jobs_{prefix}.predict.list'
    
    # Generate job files
    num_chunks = _generate_job_files(
        input_df=input_df,
        chunk_size=chunk_size,
        preproc_job_fn=preproc_job_fn,
        pred_job_fn=pred_job_fn,
        json_subdir=json_subdir,
        output_dir=output_dir,
        cfg=cfg,
        ligand_data=ligand_data,
        bonds_json=bonds_json,
        ptm_json=ptm_json,
        paths=paths
    )
    
    # Submit preprocessing jobs
    preproc_job_name = f'prep_{prefix}'
    preproc_job_id = _submit_preprocessing_jobs(
        job_file=preproc_job_fn,
        job_name=preproc_job_name,
        num_chunks=num_chunks,
        cfg=cfg
    )
    
    if preproc_job_id > 0:        
        # Wait for preprocessing jobs to complete before submitting prediction jobs
        logger.info(f"Waiting for preprocessing job {preproc_job_id} to complete...")
        slurm_tools.wait_for_jobs([preproc_job_id])
        logger.info(f"Preprocessing job {preproc_job_id} completed")
    
    job_ids = []
    # Submit prediction jobs if requested
    if cfg.slurm.submit:
        pred_job_name = (cfg.slurm.J if cfg.slurm.J is not None 
                        else f'{prefix}_{datadir.name}')
        
        pred_job_id = _submit_prediction_jobs(
            job_file=pred_job_fn,
            job_name=pred_job_name,
            num_chunks=num_chunks,
            cfg=cfg
        )
        
        if pred_job_id > 0:
            job_ids.append(pred_job_id)
    else:
        # print the commands that would be submitted (from pred_job_fn)
        with open(pred_job_fn, 'r') as f:
            print(f.read())
    
    return job_ids


def _get_paths(cfg: HydraConfig) -> Dict[str, str]:
    """Extract paths from configuration with defaults."""
    paths = {}
    
    # Get paths from config if available, otherwise use defaults
    if hasattr(cfg, 'paths'):
        paths['apptainer'] = cfg.paths.get('apptainer', 
                                "/projects/ml/cifutils/apptainer/cifutils_2025-01-27.sif")
        
        paths['af3_container'] = cfg.paths.get('af3_container', 
                                "/software/containers/users/ikalvet/mlfold3/mlfold3_01.sif")
        
        paths['process_script'] = cfg.paths.get('process_script', 
                                 "/home/ssalike/git/rf_flow_working_b/rf_diffusion/benchmark/util/process_fasta_to_json.py")
        
        paths['af3_script'] = cfg.paths.get('af3_script', 
                            "/opt/alphafold3/run_alphafold_custom.py")
    else:
        # Set default paths
        paths['apptainer'] = "/projects/ml/cifutils/apptainer/cifutils_2025-01-27.sif"
        paths['af3_container'] = "/software/containers/users/ikalvet/mlfold3/mlfold3_01.sif"
        paths['process_script'] = "/home/ssalike/git/rf_flow_working_b/rf_diffusion/benchmark/util/process_fasta_to_json.py"
        paths['af3_script'] = "/opt/alphafold3/run_alphafold_custom.py"
    
    return paths



def _generate_job_files(
    input_df: pd.DataFrame, 
    chunk_size: int,
    preproc_job_fn: Path, 
    pred_job_fn: Path,
    json_subdir: Path, 
    output_dir: Path,
    cfg: HydraConfig,
    ligand_data: Optional[str],
    bonds_json: Optional[str],
    ptm_json: Optional[str],
    paths: Dict[str, str]
) -> int:
    """
    Generate job files for preprocessing and prediction.
    
    Returns:
        Number of chunks (jobs)
    """
    with open(preproc_job_fn, 'w') as preproc_file, \
         open(pred_job_fn, 'w') as pred_file:
        
        num_chunks = 0
        
        # Process files in chunks
        for i in range(0, len(input_df), chunk_size):
            num_chunks += 1
            chunk_dir = json_subdir / f'folder{i}'
            os.makedirs(chunk_dir, exist_ok=True)
            
            # Construct arguments for preprocessing
            ligand_arg = f"--ligand_data '{ligand_data}'" if ligand_data else ""
            bonds_arg = f"--bonds '{bonds_json}'" if bonds_json else ""
            ptm_arg = f"--ptm '{ptm_json}'" if ptm_json else ""
            
            num_files = min(chunk_size, len(input_df) - i)
            
            # Write preprocessing command
            preproc_cmd = (f'apptainer exec --nv {paths["apptainer"]} '
                           f'python {paths["process_script"]} '
                           f'--input_csv {cfg.input_csv} '
                           f'--output_dir {chunk_dir} '
                           f'--start_idx {i} '
                           f'--num_files {num_files} '
                           f'{ligand_arg} '
                           f'{bonds_arg} '
                           f'{ptm_arg}')
            
            print(preproc_cmd, file=preproc_file)
            
            # Write prediction command - AlphaFold3 specific
            chunk_out_dir = output_dir / f'folder{i}'
            os.makedirs(chunk_out_dir, exist_ok=True)
            
            pred_cmd = (f'{paths["af3_container"]} python {paths["af3_script"]} '
                       f'--input_dir {chunk_dir} '
                       f'--output_dir {output_dir} --num_diffusion_samples 1')
            
            # Add additional arguments if specified
            if hasattr(cfg, 'af3_args') and cfg.af3_args:
                for key, value in cfg.af3_args.items():
                    pred_cmd += f' --{key} {value}'
            
            print(pred_cmd, file=pred_file)
    
    return num_chunks


def _submit_preprocessing_jobs(
    job_file: Path, 
    job_name: str, 
    num_chunks: int, 
    cfg: HydraConfig
) -> int:
    """Submit preprocessing jobs and return the job ID."""
    job_id, _ = slurm_tools.array_submit(
        str(job_file),  # Convert Path to string
        p='cpu',  # Preprocessing typically runs on CPU
        gres=None,
        log=cfg.slurm.keep_logs, 
        J=job_name, 
        m=4,
        in_proc=cfg.slurm.pre_process_locally,
    )
    # job_id = -1
    print("preproc job file: ", str(job_file))
    
    if job_id > 0:
        logger.info(f'Submitted preprocessing array job {job_id} with {num_chunks} jobs')
        
    return job_id


def _submit_prediction_jobs(
    job_file: Path, 
    job_name: str, 
    num_chunks: int, 
    cfg: HydraConfig
) -> int:
    """Submit prediction jobs and return the job ID."""
    job_id, _ = slurm_tools.array_submit(
        str(job_file),  # Convert Path to string 
        p=cfg.slurm.p, 
        gres=None if cfg.slurm.p=='cpu' else cfg.slurm.gres,
        log=cfg.slurm.keep_logs, 
        J=job_name, 
        in_proc=cfg.slurm.in_proc,
        m=12
    )
    # t=cfg.slurm.t

    if job_id > 0:
        logger.info(f'Submitted prediction array job {job_id} with {num_chunks} jobs')
        
    return job_id

def process_state(input_df, state, specs, cfg):
    """
    Process a single state from the specs.
    
    Args:
        input_df: Input dataframe with design information
        state: State to process
        specs_json: Path to specifications JSON file
        output_dir: Base output directory
        residue_mapping: Dictionary mapping input residues to design residues
        cfg: Configuration object
        
    Returns:
        DataFrame containing metrics for this state
    """
    state_config = specs[state].get('af3', {})
    cfg.ligand_data = state_config['ligands'] if 'ligands' in state_config else cfg.ligand_data
    cfg.bonds = state_config['bonds'] if 'bonds' in state_config else cfg.bonds
    cfg.ptm_json = state_config['PTM'] if 'PTM' in state_config else cfg.ptm_json
    cfg.prefix = state

    if not cfg.skip_af3_jobs:
        jobid_af3 = []
        jobid_af3 += main_af3(input_df, cfg)
        print('Waiting for AF3 jobs to finish...', jobid_af3)
        slurm_tools.wait_for_jobs(jobid_af3)
    else:
        print('Skipping AF3 job submission as requested...')


    ##### ANALYSIS #####
    from analyze_chai_results import main as main_analyze_chai

    prefix = cfg.prefix
    suffix = cfg.get('suffix', '')
    analysis_prefix = f'{prefix}_{suffix}' if suffix else prefix
    cfg.outdir = os.path.join(cfg.datadir, f'{analysis_prefix}_outputs')
    cfg.datadir = os.path.join(cfg.datadir, f'{analysis_prefix}_analysis') # reset datadir to analysis directory
    os.makedirs(cfg.datadir, exist_ok=True)

    if cfg.analyze.update_contigs:
        shutil.copy(cfg.input_csv, os.path.join(cfg.datadir, 'compiled_metrics.csv'))
    
    state_metrics = {state: specs[state]['metrics_info']}
    state_specs_file = os.path.join(cfg.datadir, f'{analysis_prefix}_specs.json')

    with open(state_specs_file, 'w') as f:
        json.dump(state_metrics, f, indent=2)

    cfg.specs_json = state_specs_file
    metrics_csv = os.path.join(cfg.datadir, f'{analysis_prefix}_metrics.csv')

    if hasattr(cfg.analyze, 'cautious') and cfg.analyze.cautious and os.path.exists(metrics_csv):
        print(f"Analysis output {metrics_csv} already exists, skipping AF3 analysis")
        combined_df = pd.read_csv(metrics_csv)
    else:
        print('Running AF3 analysis')
        cfg.slurm = cfg.analyze_slurm
        combined_df = main_analyze_chai(cfg)
        print('AF3 analysis jobs finished, moving onto next state')
        if not combined_df.empty:  # Only save if the dataframe isn't empty
            combined_df.to_csv(metrics_csv, index=False)

    return combined_df


@hydra.main(version_base=None, config_path='../config', config_name='af3_inference')
def main(cfg: HydraConfig, input_df: Optional[pd.DataFrame] = None):
    """
    Main function for the AF3 inference pipeline.
    
    Args:
        cfg: Hydra configuration object
        input_df: Optional pandas DataFrame. If not provided, will load from cfg.input_csv
    """
    # Load input CSV file if DataFrame not provided
    if input_df is None:
        logger.info(f"Loading input CSV from {cfg.input_csv}")
        input_df = pd.read_csv(cfg.input_csv)
    else:
        return_result_df = True
        logger.info("Using provided input DataFrame")
        
    logger.info(f"Loaded {len(input_df)} designs")
    
    # Create output directory
    os.makedirs(cfg.datadir, exist_ok=True)
    # Save input DataFrame to the output directory
    input_csv_path = os.path.join(cfg.datadir, 'input.csv')
    input_df.to_csv(input_csv_path, index=False)
    # change current working directory to output directory
    os.chdir(cfg.datadir)
    states = cfg.states  
    design_specs_json = cfg.specs_json

    with open(design_specs_json, 'r') as f:
        design_specs = json.load(f)
    
    all_metrics = {}

    for state in states:
        print(f"Processing state: {state}")
        metrics_df = process_state(
            input_df=input_df, 
            state=state, 
            specs=design_specs,
            cfg=cfg
        )
        all_metrics[state] = metrics_df

                # Store metrics for this state
        if metrics_df is not None and not metrics_df.empty:
            # Add state prefix to column names except 'name'
            metrics_df_renamed = metrics_df.rename(
                columns={col: f"{state}_{col}" if col != 'name' else col for col in metrics_df.columns}
            )
            all_metrics[state] = metrics_df_renamed
    
    # Merge all metrics with the input dataframe
    result_df = input_df.copy()
    
    # Merge with metrics from each state
    for state, metrics_df in all_metrics.items():
        if not metrics_df.empty:
            # Rename 'name' column to 'design_name' for joining
            metrics_df = metrics_df.rename(columns={'name': 'design_name'})
            
            # Merge with result dataframe
            result_df = pd.merge(result_df, metrics_df, on='design_name', how='left')
    
    # Save the final result
    result_csv = os.path.join(cfg.output_dir, 'final_results.csv')
    result_df.to_csv(result_csv, index=False)

    if return_result_df:
        return result_df
        
if __name__ == "__main__":
    main()
