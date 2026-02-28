"""
Structure prediction oracle using Boltz (https://github.com/jwohlwend/boltz).

Provides :func:`boltz_from_df`, a reward-pipeline step that writes Boltz
input YAML files, runs predictions across available GPUs in parallel, and
collects confidence metrics (pTM, ipTM, pLDDT, etc.) back into the
sequence DataFrame.
"""

import os
import json
import copy
import subprocess
import numpy as np
import torch
import glob
import pandas as pd
from omegaconf import OmegaConf


def run_multi_gpu_commands(cmd_list):
    """Runs a list of commands on multiple gpus in parallel."""

    # launching on parallel gpus
    num_gpus = torch.cuda.device_count()
    print(f"Running locally on {num_gpus} GPUs")

    # assert num commands == num_gpus
    assert len(cmd_list) == num_gpus, f"Number of commands ({len(cmd_list)}) must equal number of GPUs ({num_gpus})"

    processes = []

    for i, cmd in enumerate(cmd_list):
        gpu_idx = i % num_gpus
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        print(f"[GPU {gpu_idx}] Launching: {cmd[:100]}...")
        p = subprocess.Popen(cmd, shell=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((p, cmd, gpu_idx))

        # Wait if we filled up all GPUs
        if len(processes) == num_gpus or i == len(cmd_list) - 1:
            for p, cmd, gpu in processes:
                stdout, stderr = p.communicate()  # wait and collect output
                if p.returncode != 0:
                    print(f"[GPU {gpu}] FAILED: {cmd[:100]}...")
                    print("stdout:\n", stdout.decode())
                    print("stderr:\n", stderr.decode())
                    raise RuntimeError(f"Job failed on GPU {gpu} with code {p.returncode}")
                else:
                    print(f"[GPU {gpu}] FINISHED: {cmd[:100]}...")
            processes = []  # clear the list before the next batch
    

def chunk(_list, num_chunks):
    """splits list items into num_chunks chunks as evenly as possible"""
    n = len(_list)
    q, r = divmod(n, num_chunks)

    chunks, start = [], 0
    for i in range(min(num_chunks, n)):
        end = start + q + (i < r)
        chunks.append(_list[start:end])
        start = end

    return chunks

# For running Boltz (https://github.com/jwohlwend/boltz)
def make_boltz_command(input_folder, output_folder):
    cmd = f"boltz predict {input_folder} --out_dir {output_folder};"
    return cmd

def boltz_from_df(df_input, cfg, step_name="boltz"):
    """
    Runs Boltz and aggregates base metrics from the output.
    """
    assert cfg.rundir is not None, "rundir must be specified in cfg"

    # define and create input and output directories
    outdir = f'{cfg.rundir}/{step_name}/outputs'
    inputdir = f'{cfg.rundir}/{step_name}/inputs'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(inputdir, exist_ok=True)

    # get all sequences and names from input df
    sequences = df_input['sequence'].tolist()
    names = df_input['name'].tolist()

    # load boltz template for prediction
    template_path = cfg.template_path
    boltz_template = OmegaConf.load(template_path)


    # get chunk size based on number of available gpus
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Running locally on {num_gpus} GPUs")
        chunk_size = num_gpus
    else:
        chunk_size = 1
        print("No GPUs detected, running on CPU. This may be very slow for large numbers of sequences.")


    # build list of commands to run in parallel
    cmd_list = []
    for i, seqs_chunked in enumerate(chunk([(n,s) for n,s in zip(names,sequences)], chunk_size)):
        
        batch_folder = os.path.join(inputdir, f"batch_{i:04}")
        os.makedirs(batch_folder, exist_ok=True)

        # write input files
        for j, (name, seq) in enumerate(seqs_chunked):
            _template = copy.deepcopy(boltz_template)
            _template.sequences[0].protein.sequence = seq

            file_path = os.path.join(batch_folder, f"{name}.yaml")
            OmegaConf.save(_template, file_path)

        _cmd = make_boltz_command(batch_folder, outdir)    
        cmd_list.append(_cmd)

    
    # Write jobs file (this is just for record keeping, the actual execution is done in run_multi_gpu_commands)
    jobs_file = os.path.join(cfg.rundir, f'jobs.{step_name}.list')
    if os.path.exists(jobs_file):
        os.remove(jobs_file)
    with open(jobs_file, 'w') as f:
        f.writelines([cmd + '\n' for cmd in cmd_list])
    print(f"Wrote {len(cmd_list)} commands to {jobs_file}")

    # run commands on multiple gpus
    run_multi_gpu_commands(cmd_list)

    # collect output data
    output_folders = glob.glob(os.path.join(outdir, "*", "predictions", "*"))
    assert len(output_folders) > 0, f"No output folders found in {outdir}"

    output_data = {
        "name": [],
        f"{step_name}_path": [],
        f"{step_name}_confidence_score": [],
        f"{step_name}_ptm": [],
        f"{step_name}_iptm": [],
        f"{step_name}_ligand_iptm": [],
        f"{step_name}_protein_iptm": [],
        f"{step_name}_complex_plddt": [],
        f"{step_name}_complex_iplddt": [],
        f"{step_name}_complex_pde": [],
        f"{step_name}_complex_ipde": [],
    }

    # go through each folder and extract metrics from confidence json, and path to cif file
    for folder in output_folders:
        name = os.path.basename(folder)
        cif_path = os.path.join(folder, f"{name}_model_0.cif")
        conf_json_path = os.path.join(folder, f"confidence_{name}_model_0.json")

        with open(conf_json_path, "r") as f:
            conf_data = json.load(f)

        output_data["name"].append(name)
        output_data[f"{step_name}_path"].append(cif_path)
        output_data[f"{step_name}_confidence_score"].append(conf_data["confidence_score"])
        output_data[f"{step_name}_ptm"].append(conf_data["ptm"])
        output_data[f"{step_name}_iptm"].append(conf_data["iptm"])
        output_data[f"{step_name}_ligand_iptm"].append(conf_data["ligand_iptm"])
        output_data[f"{step_name}_protein_iptm"].append(conf_data["protein_iptm"])
        output_data[f"{step_name}_complex_plddt"].append(conf_data["complex_plddt"])
        output_data[f"{step_name}_complex_iplddt"].append(conf_data["complex_iplddt"])
        output_data[f"{step_name}_complex_pde"].append(conf_data["complex_pde"])
        output_data[f"{step_name}_complex_ipde"].append(conf_data["complex_ipde"])

    df_output = pd.DataFrame(output_data)
    df_merged = pd.merge(df_input, df_output, on='name', how='inner')

    return df_merged
