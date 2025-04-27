#!/bin/bash

# 设置算法为drqv2
algorithms=("drqv2_notem")

# 设置任务和对应的域
tasks=("walk" "stand" "swingup" "catch" "spin")
domains=("walker" "walker" "cartpole" "ball_in_cup" "finger")

# 设置种子值
seeds=(88)

timestamp=$(date +%Y%m%d_%H%M%S)

for algo in "${algorithms[@]}"; do
    for i in "${!tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            task_name="${tasks[i]}"
            domain_name="${domains[i]}"
            log_dir="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/drqv2_notem/${algo}_${domain_name}_${task_name}_seed${seed}_${timestamp}"
            echo "-------algorithm: $algo----domain: $domain_name----task: $task_name----seed: $seed----log_dir: ${log_dir}-------"
            python train.py --algorithm $algo --domain_name $domain_name --task_name $task_name --seed $seed --log_dir "${log_dir}" --eval_mode train
            if [ $? -ne 0 ]; then
                echo "Error encountered with algorithm $algo on task $task_name with seed $seed. Exiting..."
                exit 1
            fi
        done
    done
done




# CUDA_VISIBLE_DEVICES=7 ./run_experiments_14drqv2.sh