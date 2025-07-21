#!/bin/bash

# Script to run optimized multi-phase training with 2x A5500 GPUs

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Create output directory
mkdir -p ./output/optimized_run

# Run training with optimized settings
python train.py \
    --config configs/default_config.yaml \
    --mode multi_phase \
    --output_dir ./output/optimized_run \
    --device cuda \
    --wandb \
    2>&1 | tee ./output/optimized_run/training.log

echo "Training completed. Check ./output/optimized_run for results."