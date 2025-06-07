#!/bin/bash


trap 'echo "脚本被中断，退出..."; exit 1' INT


export DMCGB_DATASETS="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src/env/data"



clean_gpu_memory() {

    python -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('已清理GPU内存缓存')
"
}


eval_modes=("color_hard" "video_easy" "video_hard")


log_base_dir="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/test"

mkdir -p "$log_base_dir"

# 初始化失败计数
failed_count=0
# 创建失败模型列表文件
timestamp=$(date +%Y%m%d_%H%M%S)
failed_models_summary="${log_base_dir}/failed_models_summary_${timestamp}.txt"
echo "评估失败模型汇总 - $(date)" > "$failed_models_summary"
echo "----------------------------------------" >> "$failed_models_summary"


algorithm="svea"


seed=1


declare -A task_mapping
task_mapping["ball_in_cup"]="catch"
task_mapping["cartpole"]="balance balance_sparse swingup swingup_sparse two_poles three_poles"
task_mapping["cheetah"]="run"
task_mapping["finger"]="spin turn_easy turn_hard"
task_mapping["humanoid"]="stand walk run run_pure_state"
task_mapping["humanoid_CMU"]="stand run"
task_mapping["point_mass"]="easy hard"
task_mapping["quadruped"]="walk run escape fetch"
task_mapping["reacher"]="easy hard"
task_mapping["stacker"]="stack_2 stack_4"
task_mapping["swimmer"]="swimmer6 swimmer15"
task_mapping["walker"]="stand walk run"


declare -a model_paths=(
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_ball_in_cup_catch_20241014_224006/ball_in_cup_catch/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_cartpole_balance_20241231_200859/cartpole_balance/svea/8/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_cartpole_swingup_20241224_111440/cartpole_swingup/svea/88/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_finger_spin_20241014_224006/finger_spin/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_finger_turn_easy_20241230_145140/finger_turn_easy/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_humanoid_CMU_run_seed1_20250520_191320/humanoid_CMU_run/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_humanoid_CMU_stand_seed1_20250520_192022/humanoid_CMU_stand/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_humanoid_run_pure_state_seed1_20250520_161436/humanoid_run_pure_state/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_humanoid_run_seed1_20250520_161444/humanoid_run/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_humanoid_stand_seed1_20250520_161425/humanoid_stand/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_humanoid_walk_seed1_20250520_160841/humanoid_walk/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_point_mass_easy_20241231_200744/point_mass_easy/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_point_mass_hard_20241231_200744/point_mass_hard/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_quadruped_escape_seed1_20250520_164823/quadruped_escape/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_quadruped_fetch_seed1_20250520_164843/quadruped_fetch/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_quadruped_run_seed1_20250520_164806/quadruped_run/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_quadruped_walk_20241224_163630/quadruped_walk/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_reacher_easy_20241221_083116/reacher_easy/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_reacher_hard_20241223_171423/reacher_hard/svea/1/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_stacker_stack_2_20250105_170552/stacker_stack_2/svea/100/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_stacker_stack_4_20250102_145057/stacker_stack_4/svea/100/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_swimmer_swimmer6_20250105_170552/swimmer_swimmer6/svea/100/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_walker_stand_20241224_111440/walker_stand/svea/88/model/500000.pt"
"/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_walker_walk_20241224_111447/walker_walk/svea/88/model/500000.pt"
)


timestamp=$(date +%Y%m%d_%H%M%S)

echo "评估模型 $(date)"
echo "数据集目录: $DMCGB_DATASETS"


for model_path in "${model_paths[@]}"; do
    # 提取模型所在的目录
    model_dir=$(dirname "$model_path")
    model_id=$(basename "$model_path" .pt)
    
    # 初始化当前模型的失败标志
    model_failed=false
    
    for domain in "${!task_mapping[@]}"; do
        for task in ${task_mapping[$domain]}; do
            if [[ "$model_path" == *"${domain}_${task}"* ]]; then
                domain_name=$domain
                task_name=$task
                break 2  
            fi
        done
    done
    
    
    if [ -z "$domain_name" ] || [ -z "$task_name" ]; then
        echo "无法从预定义映射中匹配 $model_path，尝试直接从路径提取..."
        
        
        if [[ "$model_path" =~ .*/([^/]+)/([^/]+)/svea/ ]]; then
            model_folder="${BASH_REMATCH[1]}"
            domain_task_name="${BASH_REMATCH[2]}"
            
            
            if [[ "$domain_task_name" =~ ([^_]+)_(.+) ]]; then
                domain_name="${BASH_REMATCH[1]}"
                task_name="${BASH_REMATCH[2]}"
            else
                echo "无法解析 $domain_task_name"
                # 记录到汇总文件
                echo "[$(date +%Y-%m-%d_%H:%M:%S)] 无法解析 $domain_task_name" >> "$failed_models_summary"
                ((failed_count++))
                continue
            fi
        else
            echo "无法从 $model_path 提取domain和task"
            # 记录到汇总文件
            echo "[$(date +%Y-%m-%d_%H:%M:%S)] 无法从 $model_path 提取domain和task" >> "$failed_models_summary"
            ((failed_count++))
            continue
        fi
    fi
    
    
    task_log_dir="${log_base_dir}/${algorithm}/${domain_name}_${task_name}"
    mkdir -p "$task_log_dir"
    
    # 为当前模型创建失败日志文件
    failed_evals_log="${model_dir}/failed_evaluations.txt"
    
    
    echo "----------------------------------------"
    echo "处理模型: ${model_path}"
    echo "domain: ${domain_name}, task: ${task_name}"
    echo "评估结果将保存在: ${task_log_dir}"
    
    
    for eval_mode in "${eval_modes[@]}"; do
        echo "评估模式: ${eval_mode}"
        
        python eval.py \
            --algorithm $algorithm \
            --domain_name "$domain_name" \
            --task_name "$task_name" \
            --seed "$seed" \
            --eval_mode "$eval_mode" \
            --train_steps 500000 \
            --log_dir "$task_log_dir" \
            --model_path "$model_path"
        
        
        if [ $? -ne 0 ]; then
            echo "评估失败 - ${domain_name}_${task_name}, eval_mode: ${eval_mode}"
            
            # 如果这是第一次失败，初始化失败日志文件
            if [ "$model_failed" = false ]; then
                echo "评估失败记录 - 模型: $(basename "$model_path") - $(date)" > "$failed_evals_log"
                echo "----------------------------------------" >> "$failed_evals_log"
                model_failed=true
            fi
            
            # 记录评估失败到模型目录下的失败日志
            echo "[$(date +%Y-%m-%d_%H:%M:%S)] 评估失败 - domain: ${domain_name}, task: ${task_name}, eval_mode: ${eval_mode}" >> "$failed_evals_log"
            
            # 同时记录到汇总文件
            echo "[$(date +%Y-%m-%d_%H:%M:%S)] 评估失败 - 模型: ${model_path}" >> "$failed_models_summary"
            echo "    domain: ${domain_name}, task: ${task_name}, eval_mode: ${eval_mode}" >> "$failed_models_summary"
            echo "----------------------------------------" >> "$failed_models_summary"
            
            ((failed_count++))
            continue
        fi
        
        echo "完成: ${domain_name}_${task_name}, eval_mode: ${eval_mode}"
    done
    
    # 如果没有失败记录但文件已创建，添加成功信息
    if [ "$model_failed" = true ]; then
        echo "评估日志已保存到: $failed_evals_log"
    fi
    
    unset domain_name task_name
    
    echo "模型评估完成，清理GPU内存..."
    clean_gpu_memory
done

echo "所有评估完成 $(date)"
echo "共有 $failed_count 次评估失败"

# 如果没有失败的评估，添加说明
if [ $failed_count -eq 0 ]; then
    echo "所有评估成功完成！" >> "$failed_models_summary"
else
    echo "失败模型汇总已保存到: $failed_models_summary"
fi