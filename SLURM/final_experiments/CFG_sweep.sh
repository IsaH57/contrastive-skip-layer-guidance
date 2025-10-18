#!/bin/bash
#SBATCH --job-name=MI_finder
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/cfg_sweep-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/CFG_guidance_sweep.py --target=aesthetics --model=sd3 --output_path=/export/scratch/ru63zus/final_experiments/cfg_sweeps --max_prompts=3