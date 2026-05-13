#!/usr/bin/env python
"""Example SLURM launcher for extracting NERVE encoder features."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Submit feature extraction jobs for one or more W&B runs.")
    parser.add_argument('--wandb-ids', type=str, nargs='+', required=True, help='W&B run IDs')
    parser.add_argument('--dataset', type=str, default='ABCD', help='Dataset name for feature extraction')
    parser.add_argument('--pooling', type=str, default='avg', choices=['avg', 'cls', 'avg_wcls', 'all'])
    parser.add_argument('--output-dir', type=str, default='outputs/checkpoints', help='Checkpoint root directory')
    parser.add_argument('--features-dir', type=str, default='outputs/features', help='Output directory for feature CSVs')
    parser.add_argument('--wandb-project', type=str, default='nerve')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--slurm-script', type=str, default='extract_features.slurm')
    args = parser.parse_args()

    commands = []
    for run_id in args.wandb_ids:
        cmd = (
            f"python extract_features.py --dataset {args.dataset} --wandb_id {run_id} "
            f"--pooling {args.pooling} --output_dir {args.output_dir} "
            f"--features_dir {args.features_dir} --wandb_project {args.wandb_project}"
        )
        if args.wandb_entity is not None:
            cmd += f" --wandb_entity {args.wandb_entity}"
        commands.append(cmd)

    slurm_script = """#!/bin/bash
#SBATCH --job-name=nerve_extract
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
    for cmd in commands:
        slurm_script += f"\necho 'Running command: {cmd}'\n{cmd}\n"

    os.makedirs('job_err', exist_ok=True)
    os.makedirs('job_out', exist_ok=True)
    with open(args.slurm_script, 'w') as f:
        f.write(slurm_script)

    print(f"SLURM script written to {args.slurm_script}")
    result = subprocess.run(['sbatch', args.slurm_script], capture_output=True, text=True)
    print(result.stdout if result.returncode == 0 else result.stderr)
    os.remove(args.slurm_script)


if __name__ == '__main__':
    main()
