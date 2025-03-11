#!/bin/bash
# ./run_hifno_aug.sh

export CUDA_VISIBLE_DEVICES=0,6,7

export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"

GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

LOG_DIR="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src/logs/walker_walk/hifno_multigpu_aug"

torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_hifno_aug.py \
    --algorithm hifno_multigpu \
    --hidden_dim 1024 \
    --domain_name walker \
    --task_name walk \
    --seed 1 \
    --lr 1e-4 \
    --embed_dim 144 \
    --batch_size 24 \
    --num_scales 3 \
    --depth 3 \
    --patch_size 4 \
    --log_dir ${LOG_DIR}
