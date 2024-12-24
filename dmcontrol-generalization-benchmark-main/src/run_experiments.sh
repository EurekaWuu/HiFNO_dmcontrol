#!/bin/bash


# algorithms=("svea" "rad" "drq" "pad" "soda" "curl")


# tasks=("walk" "stand" "swingup" "catch" "spin")
# domains=("walker" "walker" "cartpole" "ball_in_cup" "finger")


algorithms=("svea")


tasks=("stacker_2" "stacker_4" "easy" "hard")  
domains=("stacker" "stacker" "reacher" "reacher")   


timestamp=$(date +%Y%m%d_%H%M%S)


for algo in "${algorithms[@]}"; do
    for i in "${!tasks[@]}"; do
        task_name="${tasks[i]}"
        domain_name="${domains[i]}"
        log_dir="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol-generalization-benchmark-main/dmcontrol-generalization-benchmark-main/logs/${algo}_${domain_name}_${task_name}_${timestamp}"
        echo "-------algorithm : $algo----domain : $domain_name----task : $task_name----log_dir : ${log_dir}-------"
        python train.py --algorithm $algo --domain_name $domain_name --task_name $task_name --seed 8 --log_dir "${log_dir}" --eval_mode train
        if [ $? -ne 0 ]; then
            echo "Error encountered with algorithm $algo on task $task_name. Exiting..."
            exit 1
        fi
    done
done


#   CUDA_VISIBLE_DEVICES=2 ./run_experiments.sh
