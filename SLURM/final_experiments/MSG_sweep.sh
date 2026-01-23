#!/bin/bash
#SBATCH --job-name=msg_sweep
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/msg_sweep-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/MSG_guidance_sweep.py --target=counting --model=sd3 --output_path=/export/scratch/ru63zus/final_experiments/msg_sweeps --max_prompts=101

