#!/bin/bash

# 定义算法
algorithms=("svea" "rad" "drq" "pad" "soda" "curl")

# 定义任务列表和域
tasks=("walk" "stand" "swingup" "catch" "spin")
domains=("walker" "walker" "cartpole" "ball_in_cup" "finger")

# 获取当前时间戳
timestamp=$(date +%Y%m%d_%H%M%S)

# 运行实验
for algo in "${algorithms[@]}"; do
    for i in "${!tasks[@]}"; do
        task_name="${tasks[i]}"
        domain_name="${domains[i]}"
        log_dir="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol-generalization-benchmark-main/dmcontrol-generalization-benchmark-main/logs/${algo}_${domain_name}_${task_name}_${timestamp}"
        echo "-------algorithm : $algo----domain : $domain_name----task : $task_name----log_dir : ${log_dir}-------"
        python train.py --algorithm $algo --domain_name $domain_name --task_name $task_name --seed 500 --log_dir "${log_dir}"
        if [ $? -ne 0 ]; then
            echo "Error encountered with algorithm $algo on task $task_name. Exiting..."
            exit 1
        fi
    done
done



#   ./run_experiments.sh
