import os
import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')
import torch
import numpy as np
import matplotlib.pyplot as plt
import clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import argparse
from datetime import datetime
from sklearn.manifold import TSNE
import cv2


try:
    from env.wrappers import make_env
except ImportError:
    print("警告: 无法导入环境模块，请确保路径正确")
    make_env = None

class WalkerStateVisualizer:
    def __init__(self, descriptions=None, fps=30, frames_per_segment=100):
        self.fps = fps
        self.frames_per_segment = frames_per_segment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.descriptions = descriptions if descriptions is not None else get_hifno_descriptions()
            
        text_inputs = torch.cat([clip.tokenize(desc) for desc in self.descriptions]).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
    
    def collect_walker_states(self, domain_name='walker', task_name='walk', seed=0, 
                            num_episodes=4, steps_per_episode=None, trained_agent=None, frame_indices=None,
                            episode_length=1000, action_repeat=1, image_size=84, save_video=True, save_dir=None):
        env = make_env(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            episode_length=episode_length,
            action_repeat=action_repeat,
            image_size=image_size,
            mode='train'
        )
        
        states = []
        actions = []
        rewards = []
        video_frames = []
        

        total_steps = self.frames_per_segment 
        
        sample_interval = None  

        if frame_indices is None:

            sample_interval = max(1, total_steps // 100)
            frame_indices = range(0, total_steps, sample_interval)
        
        print(f"视频总步数: {total_steps}, 预期视频长度: {total_steps/self.fps:.1f}秒 (连续片段)")
        if sample_interval:
            print(f"状态采样间隔: {sample_interval}, 预期采样状态数: {len(frame_indices)}")
        else:
            print(f"使用预定义的frame_indices, 采样点数: {len(frame_indices)}")
        
        state = env.reset()
        states.append(state)  
        episode_reward = 0
        

        print("预热阶段...")
        for _ in range(200):  
            with torch.no_grad():
                action = trained_agent.select_action(state)
            state, _, _, _ = env.step(action)
        
        print("开始主要录制阶段 (连续片段)...")
        for step in range(total_steps):
            if trained_agent is not None:
                with torch.no_grad():
                    action = trained_agent.select_action(state)
            else:
                action = env.action_space.sample()
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            

            if save_video:
                render_img = env.render(mode='rgb_array', height=image_size*2, width=image_size*2)
                video_frames.append(render_img)
            

            if step in frame_indices:
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                if step % 20 == 0:  
                    print(f"采样状态 {len(states)-1}/{len(frame_indices)}, Step {step}, Reward {reward:.2f}")
            
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}, Reward {reward:.2f}")
            
            if done:
                print(f"环境返回done，重置环境（步数 {step}）")
                state = env.reset()
                episode_reward = 0
            else:
                state = next_state
        

        if save_video and video_frames:
            print(f"收集了 {len(video_frames)} 帧视频数据")
            if save_dir:
                save_video_path = os.path.join(save_dir, f"{domain_name}_{task_name}_{seed}_simulation.mp4")
            else:
                save_video_path = os.path.join(os.path.dirname(os.getcwd()), 
                                             f'visualization_results/videos/{domain_name}_{task_name}_{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
                os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
            
            self._save_video(video_frames, save_video_path, fps=self.fps)
        
        return np.array(states), np.array(actions), np.array(rewards)
    
    def _save_video(self, frames, save_path, fps=None):
        if fps is None:
            fps = self.fps
        
        if not frames:
            print("没有帧可以保存!")
            return
        
        first_frame = frames[0]
        height, width, layers = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print(f"正在保存视频，总帧数: {len(frames)}, 分辨率: {width}x{height}, FPS: {fps}, 预期时长: {len(frames)/fps:.1f}秒")
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        video.release()
        print(f"视频已保存到: {save_path}")
    
    def preprocess_state(self, state):

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            

        if state.shape[0] > 3:
            state = state[-3:]
            

        state = F.interpolate(state.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
        

        if state.max() > 1.0:
            state = state / 255.0
            

        state = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))(state)
        
        return state
    
    def classify_state(self, state):

        state_tensor = self.preprocess_state(state).to(self.device)

        with torch.no_grad():

            image_features = self.model.encode_image(state_tensor.unsqueeze(0))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            

            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[0]
            

        top_class = similarity.argmax().item()
        
        results = [(desc, conf.item()) for desc, conf in zip(self.descriptions, similarity)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return top_class, results
    
    def visualize_classification(self, state, save_path=None):

        top_class, results = self.classify_state(state)
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        

        if isinstance(state, torch.Tensor):
            state_np = state.numpy()
        else:
            state_np = state
            

        if state_np.shape[0] > 3:
            display_state = state_np[-3:]
        else:
            display_state = state_np
            
 
        display_state = np.transpose(display_state, (1, 2, 0))
        if display_state.max() <= 1.0:
            display_state = display_state * 255
        display_state = display_state.astype(np.uint8)
        
        ax1.imshow(display_state)
        ax1.set_title('Environment State')
        ax1.axis('off')
        

        confidences = [conf for _, conf in results]
        

        short_descriptions = []
        for desc in [desc for desc, _ in results]:
            if len(desc) > 30:
                short_descriptions.append(desc[:30] + "...")
            else:
                short_descriptions.append(desc)
        
        y_pos = np.arange(len(short_descriptions))
        
        ax2.barh(y_pos, confidences, color='skyblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_descriptions)
        ax2.set_xlabel('Confidence')
        ax2.set_title('CLIP Classification Results')
        

        for i, v in enumerate(confidences):
            ax2.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return top_class, results
    
    def visualize_tsne(self, states, save_path=None):

        processed_states = []
        for state in states:
            processed_state = self.preprocess_state(state)
            processed_states.append(processed_state)
            

        batch = torch.stack(processed_states).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            
            similarities = features @ self.text_features.T
            class_indices = similarities.argmax(dim=1).cpu().numpy()
            
            features_np = features.cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_np)
        
        plt.figure(figsize=(10, 8))
        
        unique_classes = np.unique(class_indices)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        short_descriptions = {}
        for cls in unique_classes:
            desc = self.descriptions[cls]
            if len(desc) > 30:
                short_descriptions[cls] = desc[:30] + "..."
            else:
                short_descriptions[cls] = desc
        
        for i, cls in enumerate(unique_classes):
            mask = class_indices == cls
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=short_descriptions[cls], alpha=0.7)
        
        plt.legend()
        plt.title("t-SNE Visualization of CLIP Features")
        plt.grid(alpha=0.3)
        

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return features_2d, class_indices




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
    
    return parser.parse_args()

def get_hifno_descriptions():

    return [
        "One leg is supporting on the ground while the other foot swings forward.",
        "Both legs are supporting the ground.",
        "The torso tilts, falling, lying on the ground, or kneeling, with significant loss of balance."
    ]




def get_save_dir(args):

    base_dir = '/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/visualization_results/clip'
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    

    sub_dir = f"{args.domain_name}-{args.task_name}-{args.seed}-{timestamp}"
    
    save_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results will be saved to: {save_dir}")
    return save_dir

def create_reward_class_plot(states, rewards, visualizer, save_path):

    classes = []
    for state in states:
        top_class, _ = visualizer.classify_state(state)
        classes.append(top_class)
    

    unique_classes = np.unique(classes)
    class_rewards = {cls: [] for cls in unique_classes}
    
    for i, cls in enumerate(classes):
        if i < len(rewards): 
            class_rewards[cls].append(rewards[i])
    

    class_avg_rewards = {}
    for cls, rews in class_rewards.items():
        if rews:  
            class_avg_rewards[cls] = np.mean(rews)
        else:
            class_avg_rewards[cls] = 0
    

    plt.figure(figsize=(12, 6))
    
    classes = list(class_avg_rewards.keys())
    avg_rewards = list(class_avg_rewards.values())
    

    descriptions = []
    for cls in classes:
        desc = visualizer.descriptions[cls]
        if len(desc) > 30:
            desc = desc[:30] + "..."
        descriptions.append(desc)
    
    plt.bar(range(len(descriptions)), avg_rewards, color='skyblue')
    plt.xlabel('State Category')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards by State Category')
    plt.xticks(range(len(descriptions)), descriptions, rotation=45, ha='right')
    plt.tight_layout()
    

    plt.savefig(save_path)
    plt.close()

def run_interactive_visualization(args, visualizer, env, save_dir):

    state = env.reset()
    
    for i in range(1000): 

        save_path = os.path.join(save_dir, f"interactive_{i:03d}.png")
        visualizer.visualize_classification(state, save_path)
        
        print(f"\nStep: {i}")
        print("Options:")
        print("1. Execute random action")
        print("2. View current state classification")
        print("3. Reset environment")
        print("4. Exit")
        
        choice = input("Choose (default 1): ") or "1"
        
        if choice == "1":
            # 随机动作
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print(f"Action executed, reward: {reward}")
            
            if done:
                print("Episode ended, resetting environment")
                state = env.reset()
                
        elif choice == "2":
            # 查看分类
            top_class, results = visualizer.classify_state(state)
            print("\nClassification results:")
            for desc, conf in results:
                truncated_desc = desc[:70] + "..." if len(desc) > 70 else desc
                print(f"{truncated_desc}: {conf:.4f}")
                
        elif choice == "3":
            # 重置环境
            state = env.reset()
            print("Environment reset")
            
        elif choice == "4":
            # 退出
            break
            
        else:
            print("Invalid choice, try again")

def main():
    args = parse_args()
    
    if not make_env:
        print("Error: Environment module could not be imported")
        return
    
    # 使用hifno详细描述
    descriptions = get_hifno_descriptions()
    

    save_dir = get_save_dir(args)
    

    visualizer = WalkerStateVisualizer(
        descriptions=descriptions,
        fps=args.fps,
        frames_per_segment=args.frames_per_segment
    )
    

    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'  
    )
    
    if args.mode == 'episode':
        run_episode_visualization(args, visualizer, env, descriptions, save_dir)
    else:  # interactive
        run_interactive_visualization(args, visualizer, env, save_dir)

def run_episode_visualization(args, visualizer, env, descriptions, save_dir):

    trained_agent = load_trained_agent(args.model_path, args.model_type) if args.model_path else None
    

    frame_indices = None
    

    states, actions, rewards = visualizer.collect_walker_states(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        trained_agent=trained_agent,
        frame_indices=frame_indices,  
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        save_video=args.save_video,  
        save_dir=save_dir  
    )
    
    print(f"Collected {len(states)} states")
    

    class_dirs = {}
    for i in range(len(descriptions)):
        class_dir = os.path.join(save_dir, f"class_{i}")
        os.makedirs(class_dir, exist_ok=True)
        class_dirs[i] = class_dir
        

        with open(os.path.join(class_dir, "description.txt"), "w") as f:
            f.write(descriptions[i])
    

    classes = []
    for i, state in enumerate(states):
        # 分类状态
        top_class, results = visualizer.classify_state(state)
        classes.append(top_class)
        
        
        class_save_path = os.path.join(class_dirs[top_class], f"state_{i:03d}.png")
        plt.figure(figsize=(8, 8))
        
        
        if isinstance(state, torch.Tensor):
            state_np = state.numpy()
        else:
            state_np = state
            
        if state_np.shape[0] > 3:
            display_state = state_np[-3:]
        else:
            display_state = state_np
            
        display_state = np.transpose(display_state, (1, 2, 0))
        if display_state.max() <= 1.0:
            display_state = display_state * 255
        display_state = display_state.astype(np.uint8)
        
        plt.imshow(display_state)
        plt.title(f"Class: {top_class}, Confidence: {results[0][1]:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(class_save_path)
        plt.close()
        
        top_desc, top_conf = results[0]
        truncated_desc = top_desc[:50] + "..." if len(top_desc) > 50 else top_desc
        print(f"State {i}: {truncated_desc} (confidence: {top_conf:.2f})")
    

    tsne_path = os.path.join(save_dir, "tsne_visualization.png")
    visualizer.visualize_tsne(states, tsne_path)
    

    reward_path = os.path.join(save_dir, "reward_by_class.png")
    create_reward_class_plot(states, rewards, visualizer, reward_path)
    

    class_summary_path = os.path.join(save_dir, "class_summary.png")
    create_class_summary(states, visualizer, class_summary_path)
    
    print(f"All visualization results saved to: {save_dir}")
    print("Organized by class in subdirectories")
    print("Each class directory contains a description.txt file with the full class description")

def create_class_summary(states, visualizer, save_path):

    classes = []
    for state in states:
        top_class, _ = visualizer.classify_state(state)
        classes.append(top_class)
    

    unique_classes = np.unique(classes)
    class_examples = {}
    
    for cls in unique_classes:

        indices = [i for i, c in enumerate(classes) if c == cls]
        if indices:

            class_examples[cls] = states[indices[0]]
    

    n_classes = len(class_examples)
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (cls, state) in enumerate(class_examples.items()):
        if i < len(axes):
            ax = axes[i]
            

            if isinstance(state, torch.Tensor):
                state_np = state.numpy()
            else:
                state_np = state
                
            if state_np.shape[0] > 3:
                display_state = state_np[-3:]
            else:
                display_state = state_np
                
            display_state = np.transpose(display_state, (1, 2, 0))
            if display_state.max() <= 1.0:
                display_state = display_state * 255
            display_state = display_state.astype(np.uint8)
            
            ax.imshow(display_state)
            

            desc = visualizer.descriptions[cls]
            if len(desc) > 40:
                desc = desc[:40] + "..."
                
            ax.set_title(f"Class {cls}: {desc}")
            ax.axis('off')
    

    for i in range(len(class_examples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_trained_agent(model_path, model_type):

    if not model_path or not os.path.exists(model_path):
        return None
        
    print(f"Loading trained model from: {model_path}")
    try:
        if model_type == 'svea':
            agent = torch.load(model_path)
        else:

            agent = torch.load(model_path)
        agent.eval()
        return agent
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=5 python clip_vis.py --domain_name walker --task_name walk --seed 42 --num_episodes 2 --steps_per_episode 30

# 交互模式
CUDA_VISIBLE_DEVICES=5 python clip_vis.py --mode interactive

CUDA_VISIBLE_DEVICES=5 python clip_vis.py \
    --domain_name walker \
    --task_name walk \
    --seed 2 \
    --model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --model_type svea \
    --num_episodes 1 \
    --episode_length 2000 \
    --action_repeat 1 \
    --frames_per_segment 1000 \
    --fps 30 \
    --save_video


CUDA_VISIBLE_DEVICES=5 python clip_vis.py --domain_name walker --task_name walk --seed 42 --model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_walker_walk_20241224_111447/model/500000.pt" --model_type svea --num_episodes 2 --episode_length 800 --action_repeat 2 --frame_step 4


'''