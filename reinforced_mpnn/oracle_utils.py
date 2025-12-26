import os, sys
import json
import copy
import subprocess
import numpy as np
import torch
import glob
import pandas as pd


# THIS IS SPECIFC TO DIGS, WILL NEED TO GENERALIZE LATER, currently commands are 
def make_af3_command(input_folder, output_folder):
    cmd = f"""apptainer run --nv /software/containers/users/ikalvet/mlfold3/mlfold3_01.sif python /opt/alphafold3/run_alphafold.py
    --input_dir {input_folder} --output_dir {output_folder} --run_data_pipeline=false
    --model_dir=/databases/alphafold --num_diffusion_samples=1"""
    return cmd.replace("\n", " ")

def make_modelhub_command(input_folder, output_folder):
    # first part of the command sets a random port to avoid conflicts
    cmd = f"""export SINGULARITYENV_MASTER_PORT=$((1024 + RANDOM % 64512));
    apptainer exec --nvccli /net/software/containers/versions/modelhub_inference/rf3.sif rf3 fold
    inputs='{input_folder}' out_dir='{output_folder}' diffusion_batch_size=1"""
    return cmd.replace("\n", " ")

def make_boltz_command(input_folder, output_folder):
    cmd = f"""source activate boltz;
    boltz predict {input_folder} --out_dir {output_folder}; conda deactivate"""
    return cmd.replace("\n", " ")





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


def af3_from_df(df_input, cfg, step_name="af3"):
    '''
    Runs AlphaFold3 and aggregates base metrics from the output.
    '''
    assert cfg.rundir is not None, "rundir must be specified in cfg"

    outdir   = f'{cfg.rundir}/{step_name}/outputs'
    inputdir = f'{cfg.rundir}/{step_name}/inputs'

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(inputdir, exist_ok=True)

    sequences = df_input['sequence'].tolist()
    names = df_input['name'].tolist()
    template_path = cfg.template_path
    with open(template_path, 'r') as f:
        template = json.load(f)

    # check to see if template MSA is provided, and use if it is
    use_msa = False
    if hasattr(cfg, 'msa_template_path') and cfg.msa_template_path is not None:
        msa_template_path = cfg.msa_template_path
        with open(msa_template_path, 'r') as f:
            msa_template = f.readlines()
        use_msa = True


    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Running locally on {num_gpus} GPUs")
        chunk_size = num_gpus
    else:
        chunk_size = 1
        print("No GPUs detected, running on CPU.")


    # AF3 does not natively support JSON list, will need to make unique commands for each
    # instead use --input_dir flag to point to directory with json files
    # will need to batch the json files into chunks
    cmd_list = []
    for i, seqs_chunked in enumerate(chunk([(n,s) for n,s in zip(names,sequences)], chunk_size)):
        batch_folder = os.path.join(inputdir, f'batch{i:03}')
        os.makedirs(batch_folder, exist_ok=True)

        for j, (name, seq) in enumerate(seqs_chunked):
            _template = copy.deepcopy(template)
            if "protein" not in _template["sequences"][0]:
                raise ValueError("Item 0 of template JSON does not contain 'protein' key.")
            _template["sequences"][0]["protein"]["sequence"] = seq
            _template["name"] = name
            _template["modelSeeds"] = [int(np.random.randint(0,1e5))]
            
            # if using MSA template, add to json
            if use_msa:
                _msa = copy.deepcopy(msa_template)
                _msa[1] = seq + '\n'  # replace second line with new sequence
                a3m_path = os.path.join(batch_folder, f'batch{i:03}_input{j:03}.a3m')
                with open(a3m_path, 'w') as f:
                    f.writelines(_msa)
                _template["sequences"][0]["protein"]["unpairedMsaPath"] = a3m_path

            # save to batch subdir
            json_file = os.path.join(batch_folder, f'batch{i:03}_input{j:03}.json')
            with open(json_file, 'w') as f:
                json.dump(_template, f, indent=4)

        _cmd = (
            f'{cfg.command} '
            f'--input_dir {batch_folder} '
            f'--output_dir {outdir} '
            f'{cfg.command_args}; '
        )
        cmd_list.append(_cmd.replace('\n', '').replace('\t', ''))

    # Write jobs file and submit
    jobs_file = os.path.join(cfg.rundir, f'jobs.{cfg.step}.list')
    if os.path.exists(jobs_file):
        os.remove(jobs_file)

    print("Writing jobs file:", jobs_file)
    f = open(jobs_file, 'w') if jobs_file else None
    [print(cmd, file=f) for cmd in cmd_list]
    f.close() if jobs_file else None

    # run commands on multiple gpus
    run_multi_gpu_commands(cmd_list)


    # parse af3 outputs and aggregate base metrics
    af3_cif_paths = glob.glob(os.path.join(outdir, '*'))
    assert len(af3_cif_paths) > 0, f"{len(af3_cif_paths)} AF3 output files found in {outdir}"

    output_data = {
        "name": [],
        f"{step_name}_path": [],
        f"{step_name}_has_clash": [],
        f"{step_name}_iptm": [],
        f"{step_name}_ptm": [],
        f"{step_name}_ranking_score": [],
        f"{step_name}_fraction_disordered": [],
    }

    for p in af3_cif_paths:

        name = os.path.basename(p)
        af3_cif = f"{p}/{name}_model.cif"
        af3_conf_path = f"{p}/{name}_summary_confidences.json"

        with open(af3_conf_path, 'r') as f:
            metrics = json.load(f)

        output_data["name"].append(name)
        output_data[f"{step_name}_path"].append(af3_cif)
        output_data[f"{step_name}_has_clash"].append(metrics['has_clash'])
        output_data[f"{step_name}_iptm"].append(metrics['iptm'])
        output_data[f"{step_name}_ptm"].append(metrics['ptm'])
        output_data[f"{step_name}_ranking_score"].append(metrics['ranking_score'])
        output_data[f"{step_name}_fraction_disordered"].append(metrics['fraction_disordered'])

    df_output = pd.DataFrame(output_data)

    # merge with input df on name
    df_merged = pd.merge(df_input, df_output, on='name', how='inner')

    return df_merged


