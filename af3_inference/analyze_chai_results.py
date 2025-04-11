#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import sys
import os
import glob
import copy
import pandas as pd
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
import af3_inference.util.slurm_tools as slurm_tools
from pathlib import Path
import json

script_dir = os.path.dirname(os.path.realpath(__file__))

def circular_diff_deg(a, b):
    """Returns the minimal absolute difference between angles a and b in degrees [0, 180]."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def check_distance_constraint(row, column_name, constraint):
    """1-sigma bounding for distance constraints"""
    mean_col = f"{column_name}_mean"
    var_col = f"{column_name}_variance"
    
    # Check if columns exist
    if mean_col not in row or var_col not in row:
        print(f"Warning: Constraint columns {mean_col} and/or {var_col} not found. Automatically passing.")
        return True
        
    mean_val = row[mean_col]
    var_val = row[var_col]
    
    # Check for null/nan values
    if pd.isna(mean_val) or pd.isna(var_val):
        print(f"Warning: Null/NaN values found for {column_name}. Automatically failing.")
        return False
        
    sigma = np.sqrt(var_val)
    ctype = constraint["type"]
    limit = constraint["value"]

    if ctype == "max":
        return (mean_val + sigma) <= limit
    elif ctype == "min":
        return (mean_val - sigma) >= limit
    else:
        raise ValueError(f"Invalid distance constraint type: {ctype}")

def check_angle_constraint(row, column_name, constraint):
    """1-sigma bounding for angle constraints"""
    mean_col = f"{column_name}_mean"
    var_col = f"{column_name}_variance"
    
    # Check if columns exist
    if mean_col not in row or var_col not in row:
        print(f"Warning: Constraint columns {mean_col} and/or {var_col} not found. Automatically passing.")
        return True
        
    mean_val = row[mean_col]
    var_val = row[var_col]
    
    # Check for null/nan values
    if pd.isna(mean_val) or pd.isna(var_val):
        print(f"Warning: Null/NaN values found for {column_name}. Automatically passing.")
        return True
        
    sigma = np.sqrt(var_val)
    ctype = constraint["type"]
    limit = constraint["value"]

    if ctype == "angle_max":
        angle_ucb = (mean_val + sigma) % 360
        diff_ucb = circular_diff_deg(angle_ucb, 0)
        return diff_ucb <= limit
    elif ctype == "angle_min":
        angle_lcb = (mean_val - sigma) % 360
        diff_lcb = circular_diff_deg(angle_lcb, 0)
        return diff_lcb >= limit
    else:
        raise ValueError(f"Invalid angle constraint type: {ctype}")

def row_passes_constraints(row, filters):
    """Check if row passes all constraints"""
    for column_name, rule in filters.items():
        ctype = rule["type"]
        if ctype in ["max", "min"]:
            if not check_distance_constraint(row, column_name, rule):
                return False
        elif ctype in ["angle_max", "angle_min"]:
            if not check_angle_constraint(row, column_name, rule):
                return False
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")
    return True

def filter_dataframe(df, filters):
    """Filter dataframe based on constraints"""
    return df[df.apply(lambda r: row_passes_constraints(r, filters), axis=1)]

@hydra.main(version_base=None, config_path='configs/', config_name='analyze_chai')
def main(conf: HydraConfig):
    '''
    ### Expected conf keys ###
    datadir:        Directory containing CHAI prediction results
    input_csv:      Optional CSV file containing list of enzymes to analyze
    chunk:          How many designs to analyze in each job
    tmp_pre:        Name prefix for temporary files
    design_model:   Path to reference design model PDB
    motif_ligand_atoms: Dictionary of additional atoms to track
    distance_pairs: Dictionary of atom pairs to measure distances between
    motif_dict:     Dictionary of motif definitions
    
    slurm:
        J:          Job name
        p:          Partition
        gres:       Gres specification
        submit:     Submit jobs to SLURM? <True, False>
        in_proc:    Run on current node? <True, False>
        keep_logs:  Keep SLURM logs? <True, False>
    '''
    os.makedirs(conf.datadir, exist_ok=True)
    cwd = os.path.dirname(conf.datadir.rstrip('/'))
    os.chdir(cwd)

    # Get list of enzymes to analyze
    if conf.input_csv:
        df = pd.read_csv(conf.input_csv)
        enzyme_list = df['name'].tolist()
    elif conf.fasta_dir:
        enzyme_list = [os.path.basename(f).split('.')[0] for f in glob.glob(os.path.join(conf.fasta_dir, '*.fasta'))]
    else:
        # Find all PDB files and extract unique enzyme names
        json_path = conf.specs_json
        with open(json_path, 'r') as f:
            specs = json.load(f)
        # get the datadir for the first state i.e first key in specs
        datadir = specs[list(specs.keys())[0]]['out_dir']   
        pdb_files = glob.glob(os.path.join(datadir, "*.pdb")) + glob.glob(os.path.join(datadir, "*.cif"))
        enzyme_list = []

        for pdb in pdb_files:
            # Extract name between _outputs/ and .pdb
            base_name = os.path.basename(pdb)
            # Remove any additional suffixes that might be present
            enz_name = os.path.splitext(base_name)[0].split('pred.')[1]
            enzyme_list.append(enz_name)
        # Get unique names
        enzyme_list = list(set(enzyme_list))
    
    if len(enzyme_list) == 0:
        sys.exit('No enzymes found to analyze. Exiting.')

    job_ids = []
    
    # Create job list
    job_fn = os.path.join(conf.datadir, f'{conf.tmp_pre}.jobs.analyze_chai.list')
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
    
    apptainer_datahub = '/home/ssalike/git/datahub_latest/datahub_2025-02-15.sif'

    for i in range(0, len(enzyme_list), conf.chunk):
        tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.analyze.{i}'
        with open(tmp_fn, 'w') as outf:
            chunk = enzyme_list[i:min(i+conf.chunk, len(enzyme_list))]
            for enzyme in chunk:
                print(enzyme, file=outf)

        csv_out_dir = os.path.join(conf.datadir, f'{conf.tmp_pre}_analysis_csvs') 
        os.makedirs(csv_out_dir, exist_ok=True)
        
        # Create command for analysis script
        cmd = f'{apptainer_datahub} python {script_dir}/util/compile_metrics.py '\
              f'--input_list {tmp_fn} '\
              f'--outcsv {csv_out_dir}/chai_metrics.csv.{i} '\
              f'--specs_json {conf.specs_json}'
        
        # Add outdir if it's not None
        if conf.outdir is not None:
            cmd += f' --datadir {conf.outdir}'
            
        print(cmd, file=job_list_file)
    
    # Submit jobs
    if conf.slurm.submit:
        job_list_file.close()
        job_name = conf.slurm.J or f'chai_{os.path.basename(conf.datadir.strip("/"))}'
        
        analyze_job, proc = slurm_tools.array_submit(
            job_fn, 
            p=conf.slurm.p,
            gres=None if conf.slurm.p=='cpu' else conf.slurm.gres,
            log=conf.slurm.keep_logs,
            J=job_name,
            in_proc=conf.slurm.in_proc
        )

        # return analyze_job
        
        if analyze_job > 0:
            job_ids.append(analyze_job)
            print(f'Submitted array job {analyze_job} with {int(np.ceil(len(enzyme_list)/conf.chunk))} jobs to analyze {len(enzyme_list)} enzymes')
            
            # Wait for jobs to complete if not running in_proc
        if not conf.slurm.in_proc:
            slurm_tools.wait_for_jobs([analyze_job])
            
        # Combine CSV files
        csv_out_dir = os.path.join(conf.datadir, f'{conf.tmp_pre}_analysis_csvs')
        csv_files = glob.glob(os.path.join(csv_out_dir, 'chai_metrics.csv.*'))
        
        if csv_files:
            # Read and combine all CSV files
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty CSV file found: {csv_file}")
                    continue
            
            if dfs:
                combined_df = pd.concat(dfs).reset_index(drop=True)
                # Save combined CSV to datadir
                output_path = os.path.join(conf.datadir, 'metrics_combined.csv')
                combined_df.to_csv(output_path, index=False)
                print(f"Combined metrics saved to: {output_path}")
    
    return combined_df

if __name__ == "__main__":
    main()