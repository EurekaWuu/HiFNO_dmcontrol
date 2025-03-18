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
from collections import defaultdict


try:
    from env.wrappers import make_env
except ImportError:
    print("警告: 无法导入环境模块，请确保路径正确")
    make_env = None

class WalkerStateVisualizer:
    def __init__(self, descriptions=None, fps=30, frames_per_segment=100, temperature=1.0, model_name="ViT-B/32"):
        self.fps = fps
        self.frames_per_segment = frames_per_segment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载CLIP模型: {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.temperature = temperature
        self.model_name = model_name
        
        if descriptions is not None:
            self.descriptions = descriptions
            self.class_map = {} 
            

            if isinstance(descriptions[0], list):
                text_inputs_list = []
                idx = 0
                for class_idx, desc_list in enumerate(descriptions):
                    for desc in desc_list:
                        text_inputs_list.append(clip.tokenize(desc).to(self.device))
                        self.class_map[idx] = class_idx
                        idx += 1
                text_inputs = torch.cat(text_inputs_list)
                

                self.display_descriptions = [desc_list[0] for desc_list in descriptions]
            else:
                text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
                self.display_descriptions = descriptions
                for i in range(len(descriptions)):
                    self.class_map[i] = i
        else:
            descriptions = get_hifno_descriptions()
            if isinstance(descriptions[0], list):
                self.descriptions = descriptions
                self.display_descriptions = [desc_list[0] for desc_list in descriptions]
            else:
                self.descriptions = descriptions
                self.display_descriptions = descriptions
            
            text_inputs = torch.cat([clip.tokenize(desc) if not isinstance(desc, list) else 
                                    clip.tokenize(desc[0]) for desc in descriptions]).to(self.device)
            self.class_map = {i: i for i in range(len(descriptions))}
        
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

        if hasattr(state, '__array__'):

            state = np.array(state)
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            
        if state.shape[0] > 3:
            state = state[-3:]
        
        # 根据模型类型选择不同的输入尺寸
        input_size = 224
        if self.model_name == "ViT-L/14@336px":
            input_size = 336
            
        state = F.interpolate(state.unsqueeze(0), size=(input_size, input_size), mode='bilinear', align_corners=False)[0]
        
        if state.max() > 1.0:
            state = state / 255.0
            
        state = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))(state)
        
        return state
    
    def classify_state(self, state, temperature=1.0, aggregation_method='max'):
        processed_state = self.preprocess_state(state)
        batch = torch.stack([processed_state]).to(self.device)

        with torch.no_grad():

            image_features = self.model.encode_image(batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            

            similarities = (image_features @ self.text_features.T) / temperature


            if hasattr(self, 'class_map') and isinstance(self.descriptions[0], list):
                class_similarities = {}
                for feat_idx, conf in enumerate(similarities[0]):
                    class_idx = self.class_map[feat_idx]
                    

                    if class_idx not in class_similarities:
                        class_similarities[class_idx] = []
                        

                    class_similarities[class_idx].append(conf.item())
                

                aggregated_similarities = {}
                for cls, conf_list in class_similarities.items():
                    if aggregation_method == 'max':
                        # 使用最高置信度
                        aggregated_similarities[cls] = max(conf_list)
                    elif aggregation_method == 'mean':
                        # 使用平均置信度
                        aggregated_similarities[cls] = sum(conf_list) / len(conf_list)
                    elif aggregation_method == 'sum':
                        # 使用置信度总和
                        aggregated_similarities[cls] = sum(conf_list)
                    else:
                        # 默认使用最高置信度
                        aggregated_similarities[cls] = max(conf_list)
                

                top_class = max(aggregated_similarities, key=aggregated_similarities.get)
                

                results = sorted(
                    [(cls, conf) for cls, conf in aggregated_similarities.items()], 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                return top_class, results
            else:
                probs = similarities.squeeze().softmax(dim=0).cpu().numpy()
                sorted_indices = np.argsort(-probs)
                results = [(i, probs[i]) for i in sorted_indices]
                return sorted_indices[0], results
    
    def visualize_classification(self, state, save_path=None, aggregation_method='max'):
        try:
            if hasattr(state, 'copy'):
                state_copy = state.copy()
            elif hasattr(state, '__array__'):
                state_copy = np.array(state)
            else:
                state_copy = state
                
            class_index, confidence_scores = self.classify_state(state_copy, self.temperature, aggregation_method)
            
            plt.figure(figsize=(14, 8))  
            
            state_rgb = self.rgb_from_state(state)
            plt.subplot(1, 2, 1)
            plt.imshow(state_rgb)
            plt.title("State Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            
            if hasattr(self, 'class_map') and isinstance(self.descriptions[0], list):
                categories = []
                confidence = []
                original_labels = []
                
                for cls_idx, conf in confidence_scores[:5]:  
                    if cls_idx < len(self.display_descriptions):

                        original_labels.append(self.display_descriptions[cls_idx])
                        categories.append(f"Class {cls_idx}")
                        confidence.append(conf)
                    else:
                        original_labels.append(f"Unknown Class {cls_idx}")
                        categories.append(f"Class {cls_idx}")
                        confidence.append(conf)
                
                top_class_idx = class_index
                colors = ['green' if i == 0 else 'gray' for i in range(len(categories))]
                
                bars = plt.barh(categories, confidence, color=colors)
                plt.xlabel('Confidence')
                plt.title(f'Classification Results (Using {aggregation_method} aggregation)')
                

                max_conf = max(confidence) if confidence else 0.5
                for i, (bar, conf, desc) in enumerate(zip(bars, confidence, original_labels)):

                    plt.text(conf+0.01, bar.get_y()+bar.get_height()/2, f'{conf:.3f}', 
                            va='center', ha='left', fontsize=10)
                    

                    if i == 0:  
                        plt.figtext(0.5, 0.02, f"Class {top_class_idx}: {original_labels[0]}", 
                                   ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            else:
                categories = []
                confidence = []
                original_labels = []
                
                for i, (desc_idx, conf) in enumerate(confidence_scores[:5]):  
                    if isinstance(desc_idx, int) and desc_idx < len(self.display_descriptions):
                        original_labels.append(self.display_descriptions[desc_idx])
                        categories.append(f"Class {desc_idx}")
                    else:
                        original_labels.append(str(desc_idx))
                        categories.append(f"Class {desc_idx}")
                    confidence.append(conf)
                
                top_class_idx = class_index if isinstance(class_index, int) else 0
                colors = ['green' if i == 0 else 'gray' for i in range(len(categories))]
                
                bars = plt.barh(categories, confidence, color=colors)
                plt.xlabel('Confidence')
                plt.title('Classification Results')
                
                
                for bar, conf in zip(bars, confidence):
                    plt.text(conf+0.01, bar.get_y()+bar.get_height()/2, f'{conf:.3f}', 
                            va='center', ha='left', fontsize=10)
                
                if len(original_labels) > 0:
                    plt.figtext(0.5, 0.02, f"Class {top_class_idx}: {original_labels[0]}", 
                               ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
            return class_index, confidence_scores
        except Exception as e:
            print(f"可视化分类结果时出错: {e}")
            if save_path:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
            return None, None
    
    def visualize_tsne(self, states, save_path=None, aggregation_method='max'):
        processed_states = []
        for state in states:
            processed_state = self.preprocess_state(state)
            processed_states.append(processed_state)
            

        batch = torch.stack(processed_states).to(self.device)
        
        with torch.no_grad():

            batch_size = 32  
            if "ViT-L" in self.model_name:
                batch_size = 16  
            
            features_list = []
            for i in range(0, len(batch), batch_size):
                print(f"处理批次 {i//batch_size + 1}/{(len(batch) + batch_size - 1)//batch_size}...")
                batch_chunk = batch[i:i+batch_size]
                features_chunk = self.model.encode_image(batch_chunk)
                features_chunk = features_chunk / features_chunk.norm(dim=-1, keepdim=True)
                features_list.append(features_chunk)
                

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            features = torch.cat(features_list, dim=0)
            
            similarities = features @ self.text_features.T
            
            if hasattr(self, 'class_map') and isinstance(self.descriptions[0], list):
                
                class_indices = []
                for i in range(len(features)):
                    
                    sim = similarities[i]
                    
                    
                    class_similarities = {}
                    for feat_idx, conf in enumerate(sim):
                        class_idx = self.class_map[feat_idx]
                        
                       
                        if class_idx not in class_similarities:
                            class_similarities[class_idx] = []
                            
                        
                        class_similarities[class_idx].append(conf.item())
                    

                    aggregated_similarities = {}
                    for cls, conf_list in class_similarities.items():
                        if aggregation_method == 'max':
                            # 使用最高置信度
                            aggregated_similarities[cls] = max(conf_list)
                        elif aggregation_method == 'mean':
                            # 使用平均置信度
                            aggregated_similarities[cls] = sum(conf_list) / len(conf_list)
                        elif aggregation_method == 'sum':
                            # 使用置信度总和
                            aggregated_similarities[cls] = sum(conf_list)
                        else:
                            # 默认使用最高置信度
                            aggregated_similarities[cls] = max(conf_list)
                    

                    top_class = max(aggregated_similarities, key=aggregated_similarities.get)
                    class_indices.append(top_class)
                
                class_indices = np.array(class_indices)
            else:

                class_indices = similarities.argmax(dim=1).cpu().numpy()
            

            features_np = features.cpu().numpy()
        
   
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_np)
        
    
        plt.figure(figsize=(10, 8))
        
        unique_classes = np.unique(class_indices)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
    
        short_descriptions = {}
        for cls in unique_classes:
            if cls < len(self.display_descriptions):  
                desc = self.display_descriptions[cls]
                if len(desc) > 30:
                    short_descriptions[cls] = desc[:30] + "..."
                else:
                    short_descriptions[cls] = desc
            else:

                short_descriptions[cls] = f"Unknown Class {cls}"
        

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

    def rgb_from_state(self, state):
        # 处理LazyFrames对象
        if hasattr(state, '__array__'):
            state_np = np.array(state)
        elif isinstance(state, torch.Tensor):
            state_np = state.numpy()
        else:
            state_np = state
            
        # 提取RGB通道（如果有多个通道）
        if state_np.shape[0] > 3:
            display_state = state_np[-3:]
        else:
            display_state = state_np
            
        # 转换为HWC格式 (高度,宽度,通道)
        if display_state.shape[0] == 3:
            display_state = np.transpose(display_state, (1, 2, 0))
            
        # 归一化到0-255范围
        if display_state.max() <= 1.0:
            display_state = display_state * 255
            
        # 转换为uint8类型
        display_state = display_state.astype(np.uint8)
            
        return display_state

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

def get_hifno_descriptions():
    return [
        # 类别0: 行走姿势 (单腿支撑)
        [
            "WALKING POSITION: Robot has ONE leg on ground, with the OTHER leg clearly off ground and extended forward.",
            "Single leg stance with one foot off the ground in mid-step.",
            "Robot with one leg supporting body weight while the other leg is in the air.",
            "Biped walker with asymmetric leg positions: one grounded, one suspended.",
            "Robot in dynamic walking motion with one leg forward and one leg back."
        ],
        
        # 类别1: 站立姿势 (双腿支撑)
        [
            "STANDING POSITION: Robot has BOTH legs firmly planted on ground with weight evenly distributed between them.",
            "Double leg support with both feet touching the ground.",
            "Balanced robot with two legs simultaneously in contact with the floor.",
            "Stable stance with both legs supporting the body weight.",
            "Robot standing with weight distributed equally on both legs."
        ],
        
        # 类别2: 不稳定姿势
        [
            "UNSTABLE POSITION: Robot is tilting, falling, or lying on ground with significant loss of balance.",
            "Robot in unstable posture about to fall or already fallen.",
            "Biped walker losing balance with irregular body orientation.",
            "Robot tipping over or collapsed on the ground.",
            "Unbalanced position with robot unable to maintain upright posture."
        ]
    ]





def get_save_dir(args):

    base_dir = '/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/visualization_results/clip'
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    

    sub_dir = f"{args.domain_name}-{args.task_name}-{args.seed}-{timestamp}"
    
    save_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results will be saved to: {save_dir}")
    return save_dir

def create_reward_class_plot(states, rewards, visualizer, save_path, aggregation_method='max'):
    class_counts = defaultdict(int)
    class_rewards = defaultdict(list)
    
    classified_count = 0
    for state, reward in zip(states, rewards):
        try:
            class_idx, _ = visualizer.classify_state(state, visualizer.temperature, aggregation_method)
            
            if hasattr(visualizer, 'display_descriptions') and class_idx < len(visualizer.display_descriptions):
                class_name = visualizer.display_descriptions[class_idx]
            else:
                class_name = f"Class {class_idx}"
                
            class_counts[class_name] += 1
            class_rewards[class_name].append(reward)
            classified_count += 1
        except Exception as e:
            print(f"  分类状态时出错: {e}")
    
    if classified_count == 0:
        print("  警告: 没有成功分类任何状态，无法创建奖励-类别关系图")
        return
    
    avg_rewards = {}
    for class_name, rewards_list in class_rewards.items():
        avg_rewards[class_name] = sum(rewards_list) / len(rewards_list) if rewards_list else 0
    
    plt.figure(figsize=(10, 6))

    sorted_classes = sorted(avg_rewards.keys(), key=lambda x: avg_rewards[x], reverse=True)
    classes = [f"{c} (n={class_counts[c]})" for c in sorted_classes]
    rewards_avg = [avg_rewards[c] for c in sorted_classes]
    
    bars = plt.bar(classes, rewards_avg, color='skyblue')
    

    for bar, reward in zip(bars, rewards_avg):
        plt.text(bar.get_x() + bar.get_width()/2, reward + 0.01, 
                f'{reward:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Average Reward by State Category (Using {aggregation_method} aggregation)')
    plt.ylabel('Average Reward')
    plt.xlabel('State Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    

    plt.savefig(save_path)
    plt.close()

def run_interactive_visualization(args, env, visualizer):
    print("\n=== 进入交互模式 ===")
    print("按 'a' 执行随机动作，'c' 查看当前状态分类，'r' 重置环境，'q' 退出")
    
    state = env.reset()
    done = False
    
    while True:
        cv2.imshow('Interactive Mode', visualizer.rgb_from_state(state))

        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):  
            break
        elif key == ord('a'):  
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(f"执行动作后的奖励: {reward:.4f}")
            
            if done:
                print("环境结束，自动重置")
                state = env.reset()
                done = False
        elif key == ord('c'):  
            class_idx, results = visualizer.classify_state(state, visualizer.temperature, args.aggregation_method)
            
            print("\n当前状态分类:")
            
            if hasattr(visualizer, 'display_descriptions'):
                top_class = visualizer.display_descriptions[class_idx] if class_idx < len(visualizer.display_descriptions) else f"Unknown Class {class_idx}"
                print(f"最可能的类别: {top_class}")
                
                print("\n所有类别的置信度:")
                
                if hasattr(visualizer, 'class_map') and isinstance(visualizer.descriptions[0], list):
                    for cls_idx, conf in results[:5]: 
                        class_name = visualizer.display_descriptions[cls_idx] if cls_idx < len(visualizer.display_descriptions) else f"Unknown Class {cls_idx}"
                        print(f"- {class_name}: {conf:.4f}")
                else:
                    for i, (desc_idx, conf) in enumerate(results[:5]):
                        if isinstance(desc_idx, int) and desc_idx < len(visualizer.display_descriptions):
                            class_name = visualizer.display_descriptions[desc_idx]
                        else:
                            class_name = str(desc_idx)
                        print(f"- {class_name}: {conf:.4f}")
            else:
                print(f"最可能的类别索引: {class_idx}")
                print("\n所有置信度:")
                for i, (desc, conf) in enumerate(results[:5]):
                    print(f"- 类别 {desc}: {conf:.4f}")
                    
            visualizer.visualize_classification(state, aggregation_method=args.aggregation_method)
        elif key == ord('r'):  
            print("重置环境")
            state = env.reset()
            done = False
    
    cv2.destroyAllWindows()
    print("交互模式已退出")
    
    
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
            # 使用frame_step作为采样间隔
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

def run_episode_visualization(
    env, 
    visualizer, 
    num_episodes=1, 
    frames_per_segment=1000, 
    frame_step=1, 
    model=None, 
    save_video=False, 
    save_dir=None, 
    fps=30,
    frame_indices=None,  
    aggregation_method='max',
    save_examples=False  
):

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    all_states = []
    all_rewards = []
    

    for episode in range(num_episodes):
        print(f"\n处理第 {episode+1}/{num_episodes} episode:")
        

        states = []
        rewards = []
        
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        

        if model:
            while not done:

                with torch.no_grad():
                    if hasattr(model, 'act'):
                        # SVEA, DRQV2 模型
                        action = model.act(state, step, eval_mode=True)
                    elif hasattr(model, 'select_action'):

                        action = model.select_action(state)
                    else:

                        try:
                            # 提取最后3个通道（RGB图像）
                            if isinstance(state, np.ndarray) and state.shape[0] > 3:
                                rgb_state = state[-3:]
                            else:
                                rgb_state = state
                                
                            obs = torch.FloatTensor(rgb_state).to(model.device)
                            if len(obs.shape) == 3:  # 确保有批次维度
                                obs = obs.unsqueeze(0)
                                
                            action = model(obs, sample=False)
                            if isinstance(action, tuple):
                                action = action[0]
                            action = action.cpu().numpy()
                        except Exception as e:
                            print(f"动作预测出错: {e}")

                            action = env.action_space.sample()
                

                next_state, reward, done, _ = env.step(action)
                

                if frame_indices is None and step % frame_step == 0:
                    states.append(next_state)  
                    rewards.append(reward)
                elif frame_indices is not None and step in frame_indices:
                    states.append(next_state)  
                    rewards.append(reward)
                

                state = next_state
                total_reward += reward
                step += 1
                
 
                if step >= frames_per_segment:
                    break
                

                if step % 100 == 0:
                    print(f"  步骤 {step}/{frames_per_segment}，当前奖励：{total_reward:.2f}")
        else:

            while not done:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                

                if frame_indices is None and step % frame_step == 0:
                    states.append(next_state)  
                    rewards.append(reward)
                elif frame_indices is not None and step in frame_indices:
                    states.append(next_state)  
                    rewards.append(reward)
                
                state = next_state
                total_reward += reward
                step += 1
                
                if step >= frames_per_segment:
                    break
                
                if step % 100 == 0:
                    print(f"  步骤 {step}/{frames_per_segment}，当前奖励：{total_reward:.2f}")
        
        print(f"  episode {episode+1} 完成，总奖励：{total_reward:.2f}，收集了 {len(states)} 个状态")
        
        all_states.extend(states)
        all_rewards.extend(rewards)
    
    print(f"\n总共收集了 {len(all_states)} 个状态")
    

    if not all_states:
        print("没有收集到状态，无法进行可视化")
        return
    

    classified_states = []
    for i, state in enumerate(all_states):
        try:
            class_idx, results = visualizer.classify_state(state, visualizer.temperature, aggregation_method)
            
            if hasattr(visualizer, 'display_descriptions'):
                if class_idx < len(visualizer.display_descriptions):
                    top_class = visualizer.display_descriptions[class_idx]
                else:
                    top_class = f"Unknown Class {class_idx}"
            else:
                top_class = f"Class {class_idx}"
                
            confidence = results[0][1]  # 获取最高置信度
            classified_states.append((state, class_idx, top_class, confidence))
            
            if i % 10 == 0 or i == len(all_states) - 1:
                print(f"  分类进度: {i+1}/{len(all_states)}")
        except Exception as e:
            print(f"  分类状态 {i} 时出错: {e}")
    

    class_counts = defaultdict(int)
    for _, _, class_name, _ in classified_states:
        class_counts[class_name] += 1
    
    print("\n类别分布:")
    if classified_states:
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} 个状态 ({count/len(classified_states)*100:.1f}%)")
        

        avg_confidence = sum(conf for _, _, _, conf in classified_states) / len(classified_states)
        print(f"\n平均置信度: {avg_confidence:.4f}")
    else:
        print("  警告: 没有成功分类任何状态!")
    

    if save_dir and classified_states:

        reward_plot_path = os.path.join(save_dir, "reward_by_class.png")
        create_reward_class_plot(all_states, all_rewards, visualizer, reward_plot_path, aggregation_method)
        print(f"奖励-类别关系图已保存到: {reward_plot_path}")
        

        class_summary_path = os.path.join(save_dir, "class_examples.png")
        create_class_summary(all_states, visualizer, class_summary_path, aggregation_method)
        print(f"类别示例汇总已保存到: {class_summary_path}")
    

    tsne_path = os.path.join(save_dir, "tsne_visualization.png")
    try:
        visualizer.visualize_tsne(all_states, tsne_path, aggregation_method)
        print(f"t-SNE可视化已保存到: {tsne_path}")
    except Exception as e:
        print(f"创建t-SNE可视化时出错: {e}")
    

    class_dirs = {}
    if hasattr(visualizer, 'display_descriptions'):
        for i in range(len(visualizer.display_descriptions)):
            class_dir = os.path.join(save_dir, f"class_{i}")
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[i] = class_dir
            

            with open(os.path.join(class_dir, "descriptions.txt"), "w") as f:
                if hasattr(visualizer, 'descriptions') and isinstance(visualizer.descriptions[0], list):
                    for j, desc in enumerate(visualizer.descriptions[i]):
                        f.write(f"{j+1}. {desc}\n")
                else:
                    f.write(visualizer.display_descriptions[i])
    

    for i, (state, class_idx, class_name, confidence) in enumerate(classified_states):
        if class_idx in class_dirs:
            state_path = os.path.join(class_dirs[class_idx], f"state_{i:03d}_conf_{confidence:.3f}.png")
            try:
                visualizer.visualize_classification(state, state_path, aggregation_method)
            except Exception as e:
                print(f"  保存状态 {i} 到类别 {class_idx} 时出错: {e}")
    
    print(f"分类状态已保存到各个类别目录中")
    
    if save_examples and save_dir and classified_states:
        examples_dir = os.path.join(save_dir, "examples")
        if not os.path.exists(examples_dir):
            os.makedirs(examples_dir)
            

        class_examples = defaultdict(list)
        for state, class_idx, class_name, confidence in classified_states:
            if len(class_examples[class_name]) < 5:  
                class_examples[class_name].append((state, confidence))
        
        for class_name, examples in class_examples.items():
            class_dir = os.path.join(examples_dir, class_name.replace(" ", "_"))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                
            for i, (state, confidence) in enumerate(examples):
                example_path = os.path.join(class_dir, f"example_{i+1}_conf_{confidence:.3f}.png")
                visualizer.visualize_classification(state, example_path, aggregation_method)
        
        print(f"类别示例已保存到: {examples_dir}")
    

    if save_video and classified_states:
        video_path = os.path.join(save_dir, "episode_visualization.mp4")
        

        frames = []
        for i, (state, _, _, _) in enumerate(classified_states):
            vis_result_path = os.path.join(save_dir, f"temp_frame_{i:04d}.png")
            visualizer.visualize_classification(state, vis_result_path, aggregation_method)
            

            frame = cv2.imread(vis_result_path)
            frames.append(frame)
            

            os.remove(vis_result_path)
        

        if frames:
            height, width, _ = frames[0].shape
            

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            

            for frame in frames:
                video_writer.write(frame)
            

            video_writer.release()
            
            print(f"可视化视频已保存到: {video_path}")
        else:
            print("没有帧可用于创建视频")
            
    return all_states, all_rewards, classified_states

def create_class_summary(states, visualizer, save_path, aggregation_method='max'):
    classified_states = []
    for state in states:
        try:
            class_idx, _ = visualizer.classify_state(state, visualizer.temperature, aggregation_method)
            classified_states.append((state, class_idx))
        except Exception as e:
            print(f"  分类状态时出错: {e}")
    
    if not classified_states:
        print("  警告: 没有成功分类任何状态，无法创建类别示例汇总")
        return
    

    class_examples = {}
    for state, class_idx in classified_states:
        if hasattr(visualizer, 'display_descriptions') and class_idx < len(visualizer.display_descriptions):
            class_name = visualizer.display_descriptions[class_idx]
        else:
            class_name = f"Class {class_idx}"
            
        if class_name not in class_examples:
            class_examples[class_name] = state
    

    n_classes = len(class_examples)
    if n_classes == 0:
        print("  警告: 没有找到类别示例，无法创建类别示例汇总")
        return  
    
    try:
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        
        plt.figure(figsize=(cols * 4, rows * 4))
        
        for i, (class_name, state) in enumerate(class_examples.items()):
            plt.subplot(rows, cols, i + 1)
            
            try:

                state_rgb = visualizer.rgb_from_state(state)
                plt.imshow(state_rgb)
                plt.title(class_name)
                plt.axis('off')
            except Exception as e:
                print(f"  显示类别 '{class_name}' 的图像时出错: {e}")
                plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)
                plt.title(f"{class_name} (Error)")
                plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Examples of Each Category (Using {aggregation_method} aggregation)', fontsize=16, y=1.02)
        

        plt.savefig(save_path)
        plt.close()
        print(f"  类别示例汇总已保存到: {save_path}")
    except Exception as e:
        print(f"  创建类别示例汇总时出错: {e}")

        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error creating class summary: {str(e)}", ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def load_model(model_type, env, device, model_path):

    if not model_path or not os.path.exists(model_path):
        return None
        
    print(f"Loading trained model from: {model_path}")
    try:

        agent = torch.load(model_path, map_location=device)
        

        if hasattr(agent, 'eval'):
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