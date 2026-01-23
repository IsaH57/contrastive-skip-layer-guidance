#!/bin/bash
#SBATCH --job-name=geneval
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/find_layers-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
source /export/scratch/ra63vex/anaconda3/bin/activate geneval_v4
python3 eval/geneval/generation/diffusers_generate.py \
    "/export/home/ru63zus/repos/contrastive-skip-layer-guidance/eval/geneval/prompts/evaluation_metadata.jsonl" \
    --model "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --outdir "/export/scratch/ru63zus/final_experiments/geneval"