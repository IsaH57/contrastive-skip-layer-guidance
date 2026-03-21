#!/bin/bash
#SBATCH --job-name=eval_vlm
#SBATCH --partition=a100
#SBATCH --time=8:00:00
#SBATCH --output=outputs/eval_vlm-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --dependency=afterok:49633

cd ..
cd ..
nvidia-smi


# Defaults can be overridden via environment variables before calling sbatch.
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/export/scratch/ru63zus/hub/internVL-14B/models--OpenGVLab--InternVL3-14B}"
NUM_EVALS_PER_IMAGE="${NUM_EVALS_PER_IMAGE:-3}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.3}"
VLM_TOP_P="${VLM_TOP_P:-0.9}"

python3 eval/eval_vlm.py \
  --task=ukiyo \
  --model=sd35 \
  --path=/export/scratch/ru63zus/new_concepts/layer_ablations/sd35_ukiyo_cfgskip_False_layer_ablation_20260311_052034 \
  --vlm_model_path="${VLM_MODEL_PATH}" \
  --num_evals_per_image="${NUM_EVALS_PER_IMAGE}" \
  --do_sample \
  --temperature="${VLM_TEMPERATURE}" \
  --top_p="${VLM_TOP_P}" \
  --use_flash_attn
