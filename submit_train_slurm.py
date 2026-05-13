"""Example SLURM launcher for NERVE training.

Edit account/partition/module/environment settings for your cluster before use.
"""

import io
import os
import pandas as pd
import wandb

# Experiment settings
dataset = "ABCD-RBC"
embedding = "outer"   # outer = proposed bilinear tokenization
encoding = embedding
decoding = "outer"
networks = "p1"       # p1: 17-network Schaefer partition; p3: coarser grouping; vanilla: image-style blocks

job_description = f"nerve_{dataset}_{encoding}_enc_{decoding}_dec_nw_{networks}"
wandb_job_id = wandb.util.generate_id()

# Cluster settings: replace placeholders for your cluster.
start_script = """#!/bin/bash
#SBATCH --job-name=nerve
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH -e ./job_err/%j-job_err.err
#SBATCH -o ./job_out/%j-job_out.out

# Example environment setup; edit for your cluster.
# module load anaconda3
# module load cuda/11.8.0
# conda activate nerve
"""

command = f"""python train.py \\
 --dataset {dataset} \\
 --wandb_id {wandb_job_id} \\
 --description {job_description} \\
 --output_dir outputs/checkpoints \\
 --wandb_project nerve \\
 \\
 --k_fold -1 \\
 --permutation 0 \\
 \\
 --input_dim 400 \\
 --n_layers_enco 4 \\
 --nhead_enco 4 \\
 --d_model_enco 256 \\
 --n_layers_deco 1 \\
 --nhead_deco 2 \\
 --d_model_deco 64 \\
 --dropout_tsf 0.1 \\
 --mask_ratio 0.5 \\
 --norm_pix_loss 0 \\
 --embedding_type {embedding} \\
 --decoding_type {decoding} \\
 --network_method {networks} \\
 \\
 --batch_size 1024 \\
 --eval_every 50 \\
 --val_size 0.1 \\
 \\
 --optimizer AdamW \\
 --learning_rate 1e-2 \\
 --weight_decay 1e-2 \\
 --num_epochs 4000 \\
 --decay_type cosine \\
 --warmup_epochs 400 \\
 --max_grad_norm 1.0 \\
 --use_amp 1 \\
 --seed 42 \\
 --resume 0
"""

os.makedirs("job_err", exist_ok=True)
os.makedirs("job_out", exist_ok=True)
script_path = f"{job_description}.sh"
with open(script_path, "w") as fh:
    fh.write(start_script)
    fh.write(command)

stdout = pd.read_csv(io.StringIO(os.popen(f"sbatch {script_path}").read()), sep=r"\s+")
print(stdout)
os.remove(script_path)
print(f"New Job ID: {stdout.columns[-1]}")
