#!/usr/bin/env python3
"""
Utility functions for submitting SLURM jobs.
"""

# from icecream import ic
import subprocess
import re
import os
import time

def slurm_submit(cmd, p='cpu', c=1, mem=12, gres=None, J=None, wait_for=[], hold_until_finished=False, log=False, **kwargs):
    '''
    wait_for = wait for these slurm jobids to exist okay
    hold_until_finished =  if True, don't return command line control until the slurm job is done
    '''
    job_name = J if J else os.environ["USER"]+'_auto_submit'
    log_file = f'%A_%a_{J}.log' if log else '/dev/null'
    # The RAM on this gpu is busted, raises ECC errors on torch.load
    exclude_gpu = "--exclude=gpu135,gpu111,gpu130,gpu53,gpu54,gpu60,gpu51,gpu58" if gres and gres.startswith('gpu') else ""
    cmd_sbatch = f'sbatch --wrap "{cmd}" -p {p} -c {c} --mem {mem}g '\
        f'-J {job_name} '\
        f'{f"--gres {gres}" if gres else ""} '\
        f'{exclude_gpu} '\
        f'{"-W" if hold_until_finished else ""} '\
        f'{"--dependency afterok:" + ":".join(map(str, wait_for)) if wait_for else ""} '\
        f'-o {log_file} ' \
        f'--export PYTHONPATH={os.environ["PYTHONPATH"]} '
    cmd_sbatch += ' '.join([f'{"--"+k if len(k)>1 else "-"+k} {v}' for k,v in kwargs.items() if v is not None])

    proc = subprocess.run(cmd_sbatch, shell=True, stdout=subprocess.PIPE)
    slurm_job = re.findall(r'\d+', str(proc.stdout))[0]
    slurm_job = int(slurm_job)

    return slurm_job, proc

def line_count(path):
    with open(path) as fh:
        return len(fh.readlines())

def array_submit(job_list_file, p='gpu', gres='gpu:rtx2080:1', wait_for=None, log=False, in_proc=False, already_ran=None, m=12, **kwargs):
    print(f'array_submit: in_proc: {in_proc}')

    if already_ran is not None:
        job_list_file = prune_jobs(job_list_file, already_ran)

    job_count = line_count(job_list_file)
    # ic(job_count)
    if job_count == 0:
        return -1, None

    if in_proc:
        with open(job_list_file) as f:
            jobs = f.readlines()
        for job in jobs:
            # For logging (hides retcode)
            # job = re.sub('>>', '2>&1 | tee', job)
            job = re.sub('>>.*', '', job)

            print(f'running job after: {job}')

            proc = subprocess.run(job, shell=True, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise Exception(f'FAILED command: {job}. \n'
                                f'stderr: {proc.stderr.decode()}')
        return -1, None
    return slurm_submit(
        cmd = 'eval \\`sed -n \\${SLURM_ARRAY_TASK_ID}p '+job_list_file+'\\`',
        a = f'1-$(cat {job_list_file} | wc -l)',
        p = p,
        c = 1,
        mem = m,
        gres = gres,
        wait_for = wait_for,
        log = log,
        **kwargs
    )

def prune_jobs(
    job_list_file,
    already_ran):
    '''
    Inputs:
        job_list_file: path to file containing newline delimited jobs
        already_ran: dictionary mapping each line in job_list_file
            to a function which returns True if the expected outputs already exist.
    Returns:
        Path to pruned job list.
    '''
    job_list_file_pruned = job_list_file + '.pruned'
    with open(job_list_file) as f:
        jobs = f.readlines()

    pruned_jobs = []
    for job in jobs:
        job = job.strip()
        print(f'Checking {job=}')
        if not already_ran[job]():
            pruned_jobs.append(job)
    with open(job_list_file_pruned, 'w') as f:
        f.writelines(j + '\n' for j in pruned_jobs)
    
    print(f"Pruned job_list_file: {job_list_file} from {len(jobs)} to {len(pruned_jobs)} jobs")
    return job_list_file_pruned

def is_running(job_ids):
    '''Returns list of bools corresponding to whether each slurm ID in input
    list corresponds to a currently queued/running job.'''

    idstr = ','.join(map(str,job_ids))

    proc = subprocess.run(f'squeue -j {idstr}', shell=True, stdout=subprocess.PIPE)
    stdout = proc.stdout.decode()

    out = [False]*len(job_ids)
    for line in stdout.split('\n'):
        for i,id_ in enumerate(job_ids):
            if id_ == -1 or line.startswith(str(id_)):
                out[i] = True

    return out

def wait_for_jobs(job_ids, interval=10):
    '''Returns when all the SLURM jobs given in `job_ids` aren't running
    anymore.'''
    if job_ids:
        while True:
            if any(is_running(job_ids)):
                time.sleep(interval)
            else:
                break
    return 