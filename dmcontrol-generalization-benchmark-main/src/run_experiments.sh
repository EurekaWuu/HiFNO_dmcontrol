#!/bin/bash

algorithms=("drqv2_official")

# 设置任务和对应的域
tasks=("run" "stand" "trot")
domains=("dog" "dog" "dog")

# 设置种子值
seeds=()

timestamp=$(date +%Y%m%d_%H%M%S)

for algo in "${algorithms[@]}"; do
    for i in "${!tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            task_name="${tasks[i]}"
            domain_name="${domains[i]}"
            log_dir="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/drqv2_official/${algo}_${domain_name}_${task_name}_seed${seed}_${timestamp}"
            echo "-------algorithm: $algo----domain: $domain_name----task: $task_name----seed: $seed----log_dir: ${log_dir}-------"
            python train.py --algorithm $algo --domain_name $domain_name --task_name $task_name --seed $seed --log_dir "${log_dir}" --eval_mode train
            if [ $? -ne 0 ]; then
                echo "Error encountered with algorithm $algo on task $task_name with seed $seed. Exiting..."
                exit 1
            fi
        done
    done
done


#   CUDA_VISIBLE_DEVICES= ./run_experiments_3.sh
