#!/usr/bin/env python
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(
        description="Submit a SLURM job to run get_features for wandb runs."
    )
    parser.add_argument('--wandb-ids', type=str, nargs='+', required=True,
                        help="List of wandb run IDs")
    parser.add_argument('--slurm-script', type=str, default='run_get_features.slurm',
                        help="Output SLURM script file name")
    parser.add_argument('--dataset', type=str, default="ABCD", help="Dataset name")
    args = parser.parse_args()

    # Gather commands for each wandb run.
    commands = []
    for run_id in args.wandb_ids:
        for pooling in ['avg', 'cls', 'avg_wcls']:
            cmd = 'python inference.py --dataset {} --wandb_id {} --pooling {}'.format(args.dataset, run_id, pooling)
            commands.append(cmd)

    # Create a SLURM script that will run all the commands sequentially.
    # Modify the SBATCH directives (e.g., time, partition, gpus, modules) as needed.
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lem4012@med.cornell.edu
#SBATCH -e ./job_err/%j-job_err.err
#SBATCH -o ./job_out/%j-job_out.out

ulimit -n 8192

module load anaconda3
module load cuda/11.8.0
source activate pyenv
"""
    # Append each command to the SLURM script.
    for cmd in commands:
        slurm_script += f"\necho 'Running command: {cmd}'\n{cmd}\n"

    # Write the SLURM script to a file.
    with open(args.slurm_script, "w") as f:
        f.write(slurm_script)
    
    print(f"SLURM script written to {args.slurm_script}")
    
    # Submit the SLURM job.
    result = subprocess.run(["sbatch", args.slurm_script], capture_output=True, text=True)
    if result.returncode == 0:
        print("SLURM job submitted successfully:")
        print(result.stdout)
    else:
        print("Error submitting SLURM job:")
        print(result.stderr)

    # Remove the temporary script file
    os.remove(args.slurm_script)

if __name__ == "__main__":
    main()
