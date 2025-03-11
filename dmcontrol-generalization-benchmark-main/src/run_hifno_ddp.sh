#!/bin/bash
# ./run_hifno_ddp.sh

export CUDA_VISIBLE_DEVICES=0,2,3,4


GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
GPU_COUNT=$((GPU_COUNT + 1))
echo "检测到 $GPU_COUNT 个可用GPU"

export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"

# torchrun \
#     --nproc_per_node=$GPU_COUNT \
#     --master_port=29500 \
#     train_hifno_multigpu.py \
#     --algorithm hifno_multigpu \
#     --hidden_dim 2048 \
#     --domain_name walker \
#     --task_name walk \
#     --seed 1 \
#     --lr 1e-4 \
#     --embed_dim 196 \
#     --batch_size 24 \
#     --num_scales 3 \
#     --depth 4 \
#     --patch_size 4

# hifno_bisim_1_multigpu  同时使用两种损失
torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_hifno_multigpu.py \
    --algorithm hifno_bisim_1_multigpu \
    --hidden_dim 144 \
    --domain_name walker \
    --task_name walk \
    --seed 1 \
    --lr 1e-4 \
    --embed_dim 144 \
    --batch_size 24 \
    --num_scales 3 \
    --depth 4 \
    --patch_size 4 \
    --use_sc_loss True \
    --use_clip_bisim_loss True \
    --lambda_SC 0.5 \
    --lambda_clip 0.5 \
    --clip_loss_weight 0.5

# hifno_bisim_1_multigpu  只使用语义类内一致性损失
# torchrun \
#     --nproc_per_node=$GPU_COUNT \
#     --master_port=29501 \
#     train_hifno_multigpu.py \
#     --algorithm hifno_bisim_1_multigpu \
#     --hidden_dim 144 \
#     --domain_name walker \
#     --task_name walk \
#     --seed 1 \
#     --lr 1e-4 \
#     --embed_dim 144 \
#     --batch_size 24 \
#     --num_scales 3 \
#     --depth 4 \
#     --patch_size 4 \
#     --use_sc_loss True \
#     --use_clip_bisim_loss False \
#     --lambda_SC 1.0 \
#     --clip_loss_weight 0.4

# hifno_bisim_1_multigpu  只使用CLIP引导的双模拟损失
# torchrun \
#     --nproc_per_node=$GPU_COUNT \
#     --master_port=29502 \
#     train_hifno_multigpu.py \
#     --algorithm hifno_bisim_1_multigpu \
#     --hidden_dim 144 \
#     --domain_name walker \
#     --task_name walk \
#     --seed 1 \
#     --lr 1e-4 \
#     --embed_dim 144 \
#     --batch_size 24 \
#     --num_scales 3 \
#     --depth 4 \
#     --patch_size 4 \
#     --use_sc_loss False \
#     --use_clip_bisim_loss True \
#     --lambda_clip 1.0 \
#     --clip_loss_weight 0.4
