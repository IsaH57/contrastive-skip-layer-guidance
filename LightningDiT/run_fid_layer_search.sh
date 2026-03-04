#!/bin/bash
#SBATCH --job-name=fid_layer_search
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT/outputs/fid_layer_search-%j.out
#SBATCH --chdir=/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT
#SBATCH --gres=gpu:a100:1

set -euo pipefail

REPO_DIR=${REPO_DIR:-/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT}
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-/export/scratch/ru63zus/lightningdit/logs}
OUTPUT_DIR=${OUTPUT_DIR:-/export/scratch/ru63zus/lightningdit/output}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

CONFIG="$REPO_DIR/configs/lightningdit_xl_vavae_f16d32.yaml"
CKPT_PATH=/export/scratch/ru63zus/ckpts/lightningdit-xl-imagenet256-800ep.pt
FID_REFERENCE=/export/scratch/ru63zus/assets/VIRTUAL_imagenet256_labeled.npz
CFG_SCALE=1.0
SKIP_SCALE=2.0
LAYERS=${LAYERS:-}
FID_NUM=1000
PER_PROC_BATCH=4
NUM_SAMPLING_STEPS=25
EXP_NAME_PREFIX=lightningdit_xl_vavae_f16d32
OUTPUT_DIR=${OUTPUT_DIR}
RUN_CFG_BASELINE=${RUN_CFG_BASELINE:-1}
CFG_BASELINE_OUT_JSON=${CFG_BASELINE_OUT_JSON:-$OUTPUT_DIR/fid_cfg_baseline_results.json}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
PRECISION=${PRECISION:-bf16}

LAYER_ARGS=()
if [[ -n "${LAYERS}" ]]; then
  LAYER_ARGS+=(--layers "$LAYERS")
fi

if [[ "${RUN_CFG_BASELINE}" -ne 0 ]]; then
  accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    "$REPO_DIR/tools/fid_sweep.py" \
    --config "$CONFIG" \
    --ckpt-path "$CKPT_PATH" \
    --fid-reference "$FID_REFERENCE" \
    --methods cfg \
    --cfg-scale "$CFG_SCALE" \
    --fid-num "$FID_NUM" \
    --per-proc-batch-size "$PER_PROC_BATCH" \
    --num-sampling-steps "$NUM_SAMPLING_STEPS" \
    --exp-name-prefix "$EXP_NAME_PREFIX" \
    --output-dir "$OUTPUT_DIR" \
    --out-json "$CFG_BASELINE_OUT_JSON"
fi

accelerate launch \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $NODE_RANK \
  --num_processes $(($GPUS_PER_NODE*$NNODES)) \
  --num_machines $NNODES \
  --mixed_precision $PRECISION \
  "$REPO_DIR/tools/fid_layer_search.py" \
  --config "$CONFIG" \
  --ckpt-path "$CKPT_PATH" \
  --fid-reference "$FID_REFERENCE" \
  --cfg-scale "$CFG_SCALE" \
  --skip-scale "$SKIP_SCALE" \
  "${LAYER_ARGS[@]}" \
  --fid-num "$FID_NUM" \
  --per-proc-batch-size "$PER_PROC_BATCH" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --exp-name-prefix "$EXP_NAME_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --out-json "$OUTPUT_DIR/fid_layer_search_results.json"
