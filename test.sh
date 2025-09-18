#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --partition=a100       
#SBATCH --time=24:00:00         
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --gres=gpu:a100:1
nvidia-smi
python3 FLUX_final_experiments.py