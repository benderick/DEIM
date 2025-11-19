#!/bin/bash

# CODrone水平框目标检测训练脚本

set -e

cd ~/futurama/DEIM
eval "$(conda shell.zsh hook)" || eval "$(conda shell.bash hook)"
conda activate deim

# 优化多线程性能（避免系统过载）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/dfine/dfine.yml

echo "✅ Training completed!"
