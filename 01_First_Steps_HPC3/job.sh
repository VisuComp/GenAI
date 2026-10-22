#!/bin/bash
#SBATCH --job-name=mnist-mps          # name shown in squeue
#SBATCH --partition=teaching          # queue: teaching (12 h) | research (12 h)
#SBATCH --gres=shard:1                # one share of the MPS-shared GPU (hpc3-perlman)
#SBATCH --cpus-per-task=4             # CPU cores (used for data loading)
#SBATCH --mem=8G                      # RAM
#SBATCH --time=00:15:00               # wall-time limit; the job is killed after this
#SBATCH --output=logs/%x_%j.out       # stdout+stderr; %x = job name, %j = job id (logs/ must exist!)

cd "$SLURM_SUBMIT_DIR"                # the directory you ran `sbatch` in
source ~/venvs/genai/bin/activate     # jobs start with a fresh shell -> activate the venv

nvidia-smi                            # shows the whole physical card, not just your share
srun python train.py --epochs 5
