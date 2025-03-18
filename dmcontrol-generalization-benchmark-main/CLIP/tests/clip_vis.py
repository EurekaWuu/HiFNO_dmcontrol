import os
import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')
import torch
import numpy as np
import argparse
from datetime import datetime

try:
    from env.wrappers import make_env
except ImportError:
    print("警告: 无法导入环境模块，请确保路径正确")
    make_env = None

from vis import (
    WalkerStateVisualizer, 
    create_reward_class_plot, 
    create_class_summary, 
    get_hifno_descriptions,
    get_save_dir,
    load_model,
    run_interactive_visualization,
    run_episode_visualization
)

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP State Classification Visualization Tool')
    parser.add_argument('--mode', type=str, default='episode', choices=['episode', 'interactive'],
                        help='Visualization mode: episode (complete episode) or interactive')
    parser.add_argument('--domain_name', type=str, default='walker',
                        help='Environment domain name')
    parser.add_argument('--task_name', type=str, default='walk',
                        help='Task name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of episodes to collect')
    parser.add_argument('--steps_per_episode', type=int, default=20,
                        help='Steps per episode to collect')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='svea',
                        choices=['svea', 'drq', 'hifno', 'hifno_bisim'],
                        help='Type of trained model')
    
    parser.add_argument('--episode_length', type=int, default=1000,
                      help='Length of each episode')
    parser.add_argument('--action_repeat', type=int, default=4,
                      help='Number of action repeats')
    parser.add_argument('--image_size', type=int, default=84,
                      help='Size of observation images')
    parser.add_argument('--frame_step', type=int, default=20,
                      help='Number of steps between frame captures')
    
    parser.add_argument('--save_video', action='store_true',
                        help='Save simulation video')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frames per second')
    parser.add_argument('--frames_per_segment', type=int, default=300,
                        help='Number of frames to collect per segment')
    parser.add_argument('--render_size', type=int, default=256,
                        help='Size of rendered video (height and width)')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Temperature for CLIP classification (lower values make predictions more confident)')
    parser.add_argument('--use_multi_descriptions', action='store_true',
                        help='Use multiple description variants for each class')
    parser.add_argument('--aggregation_method', type=str, default='max',
                      choices=['max', 'mean', 'sum'],
                      help='Method to aggregate confidences from multiple descriptions per class')
    
    parser.add_argument('--model_name', type=str, default="ViT-B/32",
                      choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "RN50", "RN101"],
                      help='CLIP model type to use')
    
    parser.add_argument('--save_examples', action='store_true',
                      help='Save example visualizations for each class')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if make_env is None:
        print("错误: 无法导入环境模块，请确保环境设置正确")
        return
    
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'
    )
    
    descriptions = get_hifno_descriptions()
    if not args.use_multi_descriptions and isinstance(descriptions[0], list):
        descriptions = [desc_list[0] for desc_list in descriptions]
        print("使用单一描述模式")
    elif args.use_multi_descriptions:
        print(f"使用多描述变体模式，共有{sum(len(desc_list) for desc_list in descriptions)}个描述")
        print(f"使用聚合方法: {args.aggregation_method}")
    
    visualizer = WalkerStateVisualizer(
        descriptions=descriptions,
        fps=args.fps,
        frames_per_segment=args.frames_per_segment,
        temperature=args.temperature,
        model_name=args.model_name
    )
    
    print(f"使用CLIP模型: {args.model_name}")
    
    if args.mode == 'interactive':
        run_interactive_visualization(args, env, visualizer)
    elif args.mode == 'video' or args.mode == 'episode':  
        model = None
        if args.model_path and args.model_type:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = load_model(args.model_type, env, device, args.model_path)
            print(f"加载了 {args.model_type} 模型: {args.model_path}")
        
        save_dir = get_save_dir(args)
        
        print(f"视频和可视化结果将保存到: {save_dir}")
        
        frame_indices = None
        if args.frame_step > 0:
            total_steps = args.frames_per_segment
            frame_indices = range(0, total_steps, args.frame_step)
            print(f"使用frame_step={args.frame_step}进行采样，预计采样{len(frame_indices)}个状态")
        
        run_episode_visualization(
            env=env,
            visualizer=visualizer,
            num_episodes=args.num_episodes,
            frames_per_segment=args.frames_per_segment,
            frame_step=args.frame_step,
            model=model,
            save_video=args.save_video,
            save_dir=save_dir,
            fps=args.fps,
            frame_indices=frame_indices,
            aggregation_method=args.aggregation_method,
            save_examples=args.save_examples 
        )
    else:
        print(f"错误: 未知的模式 {args.mode}")
        return

if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=5 python clip_vis.py --domain_name walker --task_name walk --seed 42 --num_episodes 2 --steps_per_episode 30

# 交互模式
CUDA_VISIBLE_DEVICES=5 python clip_vis.py --mode interactive

# 使用多描述变体和较低的温度系数
CUDA_VISIBLE_DEVICES=5 python clip_vis.py \
    --domain_name walker \
    --task_name walk \
    --seed 2 \
    --model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --model_type svea \
    --num_episodes 1 \
    --episode_length 2000 \
    --action_repeat 2 \
    --frames_per_segment 1000 \
    --frame_step 15 \
    --fps 30 \
    --save_video \
    --use_multi_descriptions \
    --temperature 0.5 \
    --aggregation_method mean \
    --model_name ViT-L/14

    ViT-B/32

# 生成examples文件夹
CUDA_VISIBLE_DEVICES=5 python clip_vis.py \
    --save_examples

CUDA_VISIBLE_DEVICES=5 python clip_vis.py --domain_name walker --task_name walk --seed 42 --model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_walker_walk_20241224_111447/model/500000.pt" --model_type svea --num_episodes 2 --episode_length 800 --action_repeat 2 --frame_step 4

# 使用ViT-L/14模型
CUDA_VISIBLE_DEVICES=5 python clip_vis.py --domain_name walker --task_name walk --seed 42 --model_name ViT-L/14 --num_episodes 1 --frame_step 10

# 使用ViT-L/14@336px高分辨率模型
CUDA_VISIBLE_DEVICES=5 python clip_vis.py --domain_name walker --task_name walk --seed 42 --model_name "ViT-L/14@336px" --num_episodes 1 --frame_step 20
'''