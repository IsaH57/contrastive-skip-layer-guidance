#!/bin/bash
#SBATCH --job-name=track_times
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/track_times-%j.out
#SBATCH --gres=gpu:a100:1

nvidia-smi

# Define the root of your repository
export PYTHONPATH="/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance"

# Run the script
python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/experiment_scripts/track_times.py \
  --output_path=/export/scratch/ru63zus/final_experiments/eccv \
  --max_prompts=10 \
  --target=hands \
  --model=pixart