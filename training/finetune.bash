#!/usr/bin/env bash
set -x

export BATCH_SIZE=4
export MODEL="facebook/opt-350m"
export NUMBER_OF_GPUS=1
export OUTPUT_DIR="checkpoints"
LOG_NAME=$(date "+%Y-%m-%d_%H-%M-%S")

# Set HuggingFace Datasets to offline mode by default: since we're using local
# JSON files, hitting their servers means something went wrong. If you're doing
# something else, adjust this accordingly.
export HF_DATASETS_OFFLINE=1

# HuggingFace transformers should be allowed to hit their servers though, to
# download pre-trained models during the first execution for example.
# export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/runs"

torchrun \
  --nproc_per_node ${NUMBER_OF_GPUS} \
  --master_port 19198 \
  ./colossalai/run_clm.py \
  --train_file "./data/train.json" \
  --learning_rate "2e-5" \
  --checkpointing_steps 64 \
  --mem_cap 0 \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --per_device_train_batch_size "$BATCH_SIZE" "$@" \
  2>&1 | tee "$OUTPUT_DIR/logs/$LOG_NAME.log"
