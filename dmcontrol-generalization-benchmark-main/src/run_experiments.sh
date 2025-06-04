#!/bin/bash

timestamp=$(date +%Y%m%d_%H%M%S)
algorithms=("pieg")
domains=("walker" "cartpole" "ball_in_cup" "finger" "cheetah" "reacher")
tasks=("walk" "swingup" "catch" "spin" "run" "easy")

feature_dim=50
update_every_steps=2
num_expl_steps=2000
stddev_clip=0.3
stddev_schedule="linear(1.0,0.1,500000)"

for algo in "${algorithms[@]}"; do
    for i in "${!tasks[@]}"; do
        task_name="${tasks[i]}"
        domain_name="${domains[i]}"
        log_dir="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/pieg/${algo}_${domain_name}_${task_name}_${timestamp}"
        
        echo "-------algorithm : $algo----domain : $domain_name----task : $task_name----log_dir : ${log_dir}-------"
        
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --algorithm $algo \
            --domain_name $domain_name \
            --task_name $task_name \
            --seed 1 \
            --log_dir "${log_dir}" \
            --eval_mode train \
            --feature_dim $feature_dim \
            --update_every_steps $update_every_steps \
            --num_expl_steps $num_expl_steps \
            --stddev_clip $stddev_clip \
            --stddev_schedule $stddev_schedule \
            --hidden_dim 1024 \
            --lr 1e-4 \
            --critic_target_tau 0.01
            
        if [ $? -ne 0 ]; then
            echo "Error encountered with algorithm $algo on task $task_name. Exiting..."
            exit 1
        fi
    done
done


#   CUDA_VISIBLE_DEVICES=1 ./run_experiments_1.sh
