#!/bin/bash
# ./run_hifno_aug.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"

torchrun \
    --nproc_per_node=6 \
    --master_port=29500 \
    train_hifno_aug.py \
    --algorithm hifno_multigpu \
    --hidden_dim 1024 \
    --domain_name walker \
    --task_name walk \
    --seed 1 \
    --lr 1e-4 \
    --embed_dim 196 \
    --batch_size 24 \
    --num_scales 3 \
    --depth 3 \
    --patch_size 4
