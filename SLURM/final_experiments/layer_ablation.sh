#!/bin/bash
#SBATCH --job-name=layer_ablation
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/layer_ablation-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi

OUTPUT_ROOT="/export/scratch/ru63zus/new_concepts/layer_ablations"
MAX_PROMPTS=100

python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/experiment_scripts/layer_ablation.py \
  --target=symmetry \
  --model=sd35 \
  --output_path="${OUTPUT_ROOT}" \
  --max_prompts="${MAX_PROMPTS}"

python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/experiment_scripts/layer_ablation.py \
  --target=ukiyo \
  --model=sd35 \
  --output_path="${OUTPUT_ROOT}" \
  --max_prompts="${MAX_PROMPTS}"

  python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/experiment_scripts/layer_ablation.py \
  --target=background \
  --model=sd35 \
  --output_path="${OUTPUT_ROOT}" \
  --max_prompts="${MAX_PROMPTS}"
