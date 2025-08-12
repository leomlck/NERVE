import os
import io
import pandas as pd
import itertools
import wandb
import subprocess
import time

# Define hyperparameter lists
dataset           = 'ABCD-RBC'
embedding_types   = ['outer']
decoding_types    = ['linear']
network_methods   = ['p1']
mask_ratios       = [0.5]
learning_rates    = [1e-3]
weight_decays     = [1e-2]
n_layers_enco_list= [4]
n_heads_enco_list = [4]  # 2, 4
d_model_enco_list = [256] # 128, 256
n_layers_deco_list= [1]
n_heads_deco_list = [2]
d_model_deco_list = [64]
norm_pix_loss_list= [0]
num_epochs_list   = [4000]
seed_list         = [42]

# Define max number of jobs in queue before waiting
MAX_RUNNING_JOBS = 50  # Adjust based on cluster limits

def count_running_jobs():
    """Check the number of currently running or pending SLURM jobs for the user."""
    result = subprocess.run(['squeue', '-u', os.getenv('USER')], capture_output=True, text=True)
    job_count = len(result.stdout.strip().split("\n")) - 1  # Exclude header
    return max(job_count, 0)  # Ensure non-negative count

def wait_for_job_slots():
    """Wait until the number of jobs is below MAX_RUNNING_JOBS."""
    while count_running_jobs() >= MAX_RUNNING_JOBS:
        print(f"Waiting for job slots... ({count_running_jobs()} jobs running)")
        time.sleep(5*60)  # Check every 60 seconds

# Loop through all hyperparameter combinations
for (embedding, decoding, network, mask_ratio, lr, wd, n_layers_enco, n_heads_enco, d_model_enco,
     n_layers_deco, n_heads_deco, d_model_deco, norm_pix_loss, num_epochs, seed) in itertools.product(
        embedding_types, decoding_types, network_methods, mask_ratios, learning_rates, weight_decays,
        n_layers_enco_list, n_heads_enco_list, d_model_enco_list, n_layers_deco_list, n_heads_deco_list, d_model_deco_list,
        norm_pix_loss_list, num_epochs_list, seed_list):

    # Wait for free job slots
    wait_for_job_slots()

    # Create a unique job description that reflects key parameters
    job_description = f"fc_mae_{dataset}_{embedding}_enc_{decoding}_dec_{network}"
    wandb_job_id = wandb.util.generate_id()

    # Define the SLURM script header
    start_script = f"""#!/bin/bash
#SBATCH --job-name={job_description}
##SBATCH --begin=now+6hours
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lem4012@med.cornell.edu
#SBATCH -e ./job_err/%j-job_err.err
#SBATCH -o ./job_out/%j-job_out.out

# Increase the open files limit
ulimit -n 8192

module load anaconda3
module load cuda/11.8.0
source activate pyenv
"""

    # Define the command with hyperparameters injected
    command = f"""python train.py \
 --dataset {dataset} \
 --output_dir /midtier/sablab/scratch/lem4012/save/fc_mae_models \
 --wandb_id {wandb_job_id} \
 --description {job_description} \
 --permutation 0 \
 \
 --input_dim 400 \
 --n_layers_enco {n_layers_enco} \
 --nhead_enco {n_heads_enco} \
 --d_model_enco {d_model_enco} \
 --n_layers_deco {n_layers_deco} \
 --nhead_deco {n_heads_deco} \
 --d_model_deco {d_model_deco} \
 --dropout_tsf 0.1 \
 --mask_ratio {mask_ratio} \
 --norm_pix_loss {norm_pix_loss} \
 --embedding_type {embedding} \
 --decoding_type {decoding} \
 --network_method {network} \
 \
 --batch_size 4096 \
 --eval_every 20 \
 --val_size 0.1 \
 \
 --optimizer AdamW \
 --learning_rate {lr} \
 --weight_decay {wd} \
 --num_epochs {num_epochs} \
 --decay_type cosine \
 --warmup_epochs 500 \
 --max_grad_norm 1.0 \
 --use_amp 1 \
 \
 --use_distill 0 \
 --loss_weight 0.3 \
 --teacher_mask_ratio 0.25 \
\
 --seed {seed} \
 --resume 0
"""

    # Write the SLURM script to a temporary file
    with open(job_description, 'w') as fh:
        fh.write(start_script)
        fh.write(command)

    # Submit the job using sbatch and capture the output
    stdout = pd.read_csv(io.StringIO(os.popen(f"sbatch {job_description}").read()), sep='\s+')
    print(stdout)

    # Remove the temporary SLURM script file
    os.remove(job_description)

    # Extract and print the job ID (last column of stdout)
    JOBID = str(stdout.columns[-1])
    print(f'New Job ID: {JOBID}')
