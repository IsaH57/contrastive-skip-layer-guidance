#!/bin/bash
#SBATCH --job-name=eval_ocr
#SBATCH --partition=a100     
#SBATCH --time=2:00:00         
#SBATCH --output=outputs/eval_text-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 eval/eval_text.py --path=/export/scratch/ru63zus/final_experiments/layer_count_ablation/flux_complex_text_msg_guidance_ablation_20251105_224311 --model=flux