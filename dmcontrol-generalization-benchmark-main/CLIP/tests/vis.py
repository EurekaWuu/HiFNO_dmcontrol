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
                        aggregated_similarities[cls] = max(conf_list)
                    elif aggregation_method == 'mean':
                        aggregated_similarities[cls] = sum(conf_list) / len(conf_list)
                    elif aggregation_method == 'sum':
                        aggregated_similarities[cls] = sum(conf_list)
                    else:
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
                            aggregated_similarities[cls] = max(conf_list)
                        elif aggregation_method == 'mean':
                            aggregated_similarities[cls] = sum(conf_list) / len(conf_list)
                        elif aggregation_method == 'sum':
                            aggregated_similarities[cls] = sum(conf_list)
                        else:
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
        if hasattr(state, '__array__'):
            state_np = np.array(state)
        elif isinstance(state, torch.Tensor):
            state_np = state.numpy()
        else:
            state_np = state
            
        if state_np.shape[0] > 3:
            display_state = state_np[-3:]
        else:
            display_state = state_np
            
        if display_state.shape[0] == 3:
            display_state = np.transpose(display_state, (1, 2, 0))
            
        if display_state.max() <= 1.0:
            display_state = display_state * 255
            
        display_state = display_state.astype(np.uint8)
            
        return display_state

    def cluster_states(self, states, n_clusters=None, use_text_descriptions=True, 
                       algorithm='kmeans', save_path=None, visualize=True):
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        import numpy as np
        
        print("处理状态并提取CLIP特征...")
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
            features_np = features.cpu().numpy()
        
        if algorithm.lower() == 'kmeans':
            if n_clusters is None:
                print("自动确定最佳聚类数量...")
                max_clusters = min(10, len(states) // 5)
                if max_clusters < 2:
                    max_clusters = 2
                
                inertias = []
                silhouette_scores = []
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(features_np)
                    inertias.append(kmeans.inertia_)
                    
                    if k > 1:
                        score = silhouette_score(features_np, kmeans.labels_)
                        silhouette_scores.append(score)
                        print(f"  k={k}, 轮廓系数: {score:.4f}, 惯性: {kmeans.inertia_:.4f}")
                
                if silhouette_scores:
                    best_k = np.argmax(silhouette_scores) + 2
                    print(f"基于轮廓系数选择的最佳聚类数量: {best_k}")
                else:
                    best_k = 3
                    print(f"使用默认聚类数量: {best_k}")
                    
                n_clusters = best_k
            
            print(f"使用K-means执行聚类，k={n_clusters}...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_np)
            cluster_centers = kmeans.cluster_centers_
            
        elif algorithm.lower() == 'dbscan':
            from sklearn.neighbors import NearestNeighbors
            
            print("计算最佳DBSCAN参数...")
            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(features_np)
            distances, _ = nn.kneighbors(features_np)
            
            distances = np.sort(distances[:, 4], axis=0)
            eps = np.mean(distances) * 1.5
            
            min_samples = 5
            if len(features_np) < 50:
                min_samples = 3
                
            print(f"使用DBSCAN执行聚类，eps={eps:.4f}, min_samples={min_samples}...")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(features_np)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"DBSCAN发现了{n_clusters}个簇和{np.sum(cluster_labels == -1)}个噪声点")
            
            cluster_centers = []
            for i in range(max(cluster_labels) + 1):
                if i in cluster_labels:
                    mask = cluster_labels == i
                    center = features_np[mask].mean(axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers)
        
        else:
            raise ValueError(f"不支持的聚类算法: {algorithm}，请使用'kmeans'或'dbscan'")
        
        unique_clusters = sorted(np.unique(cluster_labels))
        cluster_counts = {c: np.sum(cluster_labels == c) for c in unique_clusters}
        total_states = len(cluster_labels)
        
        print("\n聚类分布:")
        for cluster in unique_clusters:
            count = cluster_counts[cluster]
            percentage = (count / total_states) * 100
            if cluster == -1:
                print(f"  噪声点: {count} 个状态 ({percentage:.1f}%)")
            else:
                print(f"  簇 {cluster}: {count} 个状态 ({percentage:.1f}%)")
        
        cluster_descriptions = {}
        if use_text_descriptions and hasattr(self, 'text_features'):
            print("为每个聚类寻找最匹配的文本描述...")
            centers_tensor = torch.FloatTensor(cluster_centers).to(self.device)
            centers_tensor = centers_tensor / centers_tensor.norm(dim=1, keepdim=True)
            
            centers_tensor = centers_tensor.to(self.text_features.dtype)
            
            with torch.no_grad():
                similarities = centers_tensor @ self.text_features.T
            
            all_descriptions = []
            if isinstance(self.descriptions[0], list):
                for class_idx, desc_list in enumerate(self.descriptions):
                    for desc in desc_list:
                        all_descriptions.append((class_idx, desc))
            else:
                for idx, desc in enumerate(self.descriptions):
                    all_descriptions.append((idx, desc))
            
            for cluster_idx in range(len(cluster_centers)):
                if algorithm.lower() == 'dbscan' and cluster_idx >= similarities.shape[0]:
                    continue
                    
                sim_scores = similarities[cluster_idx].cpu().numpy()
                top_matches = np.argsort(-sim_scores)[:3]
                
                cluster_descriptions[cluster_idx] = []
                for match_idx in top_matches:
                    if match_idx < len(all_descriptions):
                        class_idx, description = all_descriptions[match_idx]
                        similarity = sim_scores[match_idx]
                        cluster_descriptions[cluster_idx].append({
                            'class': class_idx,
                            'description': description,
                            'similarity': float(similarity)
                        })
        
        if visualize:
            self.visualize_clustering(states, features_np, cluster_labels, 
                                     cluster_descriptions, algorithm, save_path)
        
        return features_np, cluster_labels, cluster_descriptions

    def visualize_clustering(self, states, features, cluster_labels, cluster_descriptions=None,
                             algorithm='kmeans', save_path=None):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from collections import defaultdict
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        unique_clusters = sorted(set(cluster_labels))
        n_clusters = len(unique_clusters)
        
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:
                plt.scatter(features_2d[cluster_labels == cluster, 0], 
                           features_2d[cluster_labels == cluster, 1],
                           c='black', marker='x', label='噪声', alpha=0.7)
            else:
                color_idx = i if -1 not in unique_clusters else i-1
                plt.scatter(features_2d[cluster_labels == cluster, 0], 
                          features_2d[cluster_labels == cluster, 1],
                          c=[colors[color_idx]], label=f'簇 {cluster}', alpha=0.7)
        
        plt.title(f"使用{algorithm.upper()}的聚类结果 - PCA可视化")
        plt.xlabel("主成分1")
        plt.ylabel("主成分2")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        
        cluster_stats = defaultdict(int)
        for label in cluster_labels:
            cluster_stats[label] += 1
        
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1], reverse=True)
        
        y_pos = np.arange(len(sorted_clusters))
        cluster_sizes = [count for _, count in sorted_clusters]
        cluster_labels_sorted = [f'簇 {cluster}' if cluster != -1 else '噪声' 
                               for cluster, _ in sorted_clusters]
        
        bars = plt.barh(y_pos, cluster_sizes)
        
        for i, (cluster, _) in enumerate(sorted_clusters):
            if cluster == -1:
                bars[i].set_color('black')
            else:
                color_idx = unique_clusters.index(cluster)
                if -1 in unique_clusters:
                    color_idx = color_idx if color_idx == 0 else color_idx-1
                bars[i].set_color(colors[color_idx])
        
        plt.yticks(y_pos, cluster_labels_sorted)
        plt.xlabel('样本数量')
        plt.title('各簇样本分布')
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{cluster_sizes[i]}', va='center')
        
        plt.subplot(2, 2, 3)
        plt.axis('off')
        
        n_examples = min(9, n_clusters)
        cols = 3
        rows = (n_examples + cols - 1) // cols
        
        max_label_len = 0
        cluster_examples = {}
        
        for cluster in unique_clusters:
            if cluster == -1:
                continue
            
            mask = cluster_labels == cluster
            cluster_states = [states[i] for i in range(len(states)) if mask[i]]
            
            if cluster_states:
                cluster_features = features[mask]
                center = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = np.argmin(distances)
                
                cluster_examples[cluster] = cluster_states[closest_idx]
        
        for i, cluster in enumerate(sorted([c for c in unique_clusters if c != -1])[:n_examples]):
            if cluster in cluster_examples:
                plt.subplot(rows, cols, i+1)
                try:
                    state_rgb = self.rgb_from_state(cluster_examples[cluster])
                    plt.imshow(state_rgb)
                    
                    title = f"簇 {cluster}"
                    if cluster_descriptions and cluster in cluster_descriptions:
                        top_desc = cluster_descriptions[cluster][0]
                        desc_text = top_desc['description']
                        similarity = top_desc['similarity']
                        
                        if len(desc_text) > 50:
                            desc_text = desc_text[:47] + "..."
                            
                        title += f"\n相似: {similarity:.2f}"
                        plt.figtext(0.5, 0.01 + i*0.03, f"簇 {cluster}: {desc_text}", 
                                  ha="center", fontsize=8, 
                                  bbox={"facecolor":"white", "alpha":0.8, "pad":2})
                    
                    plt.title(title)
                    plt.axis('off')
                except Exception as e:
                    plt.text(0.5, 0.5, f"错误: {str(e)}", ha='center', va='center', 
                           transform=plt.gca().transAxes)
                    plt.title(f"簇 {cluster} (错误)")
                    plt.axis('off')
        
        plt.suptitle(f"使用{algorithm.upper()}的聚类分析\n总样本: {len(states)}, 簇数量: {n_clusters}", 
                    fontsize=16, y=0.98)
        
        if cluster_descriptions:
            desc_text = "簇与文本描述匹配:\n"
            
            for cluster in sorted([c for c in unique_clusters if c != -1]):
                if cluster in cluster_descriptions:
                    top_match = cluster_descriptions[cluster][0]
                    desc = top_match['description']
                    if len(desc) > 60:
                        desc = desc[:57] + "..."
                    
                    desc_text += f"簇 {cluster}: {desc} ({top_match['similarity']:.2f})\n"
            
            plt.figtext(0.7, 0.25, desc_text, fontsize=8, 
                       bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"聚类可视化已保存到: {save_path}")
            plt.close()
        else:
            plt.show()

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

def get_hifno_descriptions():
    return [
        # 类别0: 单腿支撑姿势 (包括行走和站立时的单腿状态)
        [
            "Robot with ONLY ONE leg touching the ground while other leg is completely off ground.",
            "Single leg stance where exactly one foot contacts surface and the other is elevated.",
            "Robot balancing on one leg only, with second leg clearly lifted from the ground.",
            "Side view of robot with only one leg providing support, other leg raised in air.",
            "Dynamic or static pose with single-leg support and zero contact from second leg.",
            "Robot with one leg vertically raised upward while standing on the other leg.",
            "Single leg balance posture with one leg extended upward at significant angle.",
            "Robot performing leg raise with one foot planted firmly on ground for stability.",
            "Asymmetrical leg position with one leg providing complete support and other elevated.",
            "Robot in single-leg stance with vertical alignment and one leg lifted from ground.",
            "Robot with exactly ONE point of leg contact with the ground surface.",
            "Precise single leg balancing with other leg held in controlled raised position.",
            "Robot demonstrating unipedal stance with complete elevation of non-support leg."
        ],
        
        # 类别1: 双腿支撑姿势 (必须两腿都接触地面)
        [
            "Robot with BOTH legs simultaneously contacting the ground surface.",
            "Standing position with TWO feet firmly planted on ground with weight distribution.",
            "Bipedal stance with both legs providing simultaneous support and stability.",
            "Stationary balanced pose with two legs making complete ground contact.",
            "Robot supported by two legs with both feet touching the floor simultaneously.",
            "Two-leg support configuration with dual ground contact points.",
            "Robot with weight distributed across both legs in contact with ground.",
            "Stable standing posture requiring both feet to be touching surface.",
            "Symmetrical leg positioning with two contact points on the ground."
        ],
        
        # 类别2: 不稳定姿势 (任何失衡状态，包括行走中的不稳定)
        [
            "Robot in UNBALANCED POSITION with visible tilt angle, on verge of falling or already fallen.",
            "Unstable robot pose with torso leaning at extreme angle relative to ground plane.",
            "Robot's body oriented horizontally or diagonally instead of upright, showing loss of balance.",
            "Fallen walker with limbs splayed and torso making contact with ground surface.",
            "Walking robot with upper body tilted forward beyond normal gait posture.",
            "Robot with forward lean indicating beginning of balance loss during movement.",
            "Unbalanced state with torso pitched forward and compromised stability.",
            "Robot showing signs of stumbling with abnormal leg position and body tilt.",
            "Biped in transition between standing and falling, with excessive momentum.",
            "Robot with body tilted more than 20 degrees from vertical during locomotion.",
            "Unstable stride with visible upper body sway and compromised balance.",
            "Robot performing unstable movements with compensatory leg adjustments to prevent falling."
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

def run_interactive_visualization(env, visualizer, aggregation_method='max'):
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
            class_idx, results = visualizer.classify_state(state, visualizer.temperature, aggregation_method)
            
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
                    
            visualizer.visualize_classification(state, aggregation_method=aggregation_method)
        elif key == ord('r'):  
            print("重置环境")
            state = env.reset()
            done = False
    
    cv2.destroyAllWindows()
    print("交互模式已退出")

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
    save_examples=False,
    use_clustering=False,
    clustering_algorithm='kmeans',
    n_clusters=None,
    use_text_descriptions=True
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
                        action = model.act(state, step, eval_mode=True)
                    elif hasattr(model, 'select_action'):
                        action = model.select_action(state)
                    else:
                        try:
                            if isinstance(state, np.ndarray) and state.shape[0] > 3:
                                rgb_state = state[-3:]
                            else:
                                rgb_state = state
                                
                            obs = torch.FloatTensor(rgb_state).to(model.device)
                            if len(obs.shape) == 3:
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
    
    if use_clustering:
        print("\n执行无监督聚类分析...")
        clustering_save_path = os.path.join(save_dir, f"clustering_{clustering_algorithm}.png")
        features, cluster_labels, cluster_descriptions = visualizer.cluster_states(
            all_states, 
            n_clusters=n_clusters,
            use_text_descriptions=use_text_descriptions,
            algorithm=clustering_algorithm,
            save_path=clustering_save_path
        )
        
        print(f"聚类分析完成，结果已保存到: {clustering_save_path}")
        
        cluster_dirs = {}
        unique_clusters = sorted(set(cluster_labels))
        for cluster_idx in unique_clusters:
            if cluster_idx == -1:
                cluster_dir = os.path.join(save_dir, "cluster_noise")
            else:
                cluster_dir = os.path.join(save_dir, f"cluster_{cluster_idx}")
            os.makedirs(cluster_dir, exist_ok=True)
            cluster_dirs[cluster_idx] = cluster_dir
            
            if cluster_descriptions and cluster_idx in cluster_descriptions:
                with open(os.path.join(cluster_dir, "descriptions.txt"), "w") as f:
                    for i, desc_info in enumerate(cluster_descriptions[cluster_idx]):
                        desc = desc_info['description']
                        similarity = desc_info['similarity']
                        class_idx = desc_info['class']
                        f.write(f"{i+1}. 描述: {desc}\n")
                        f.write(f"   相似度: {similarity:.4f}, 原始类别: {class_idx}\n\n")
        
        print("\n各簇样本分布:")
        cluster_counts = defaultdict(int)
        for cluster_idx in cluster_labels:
            cluster_counts[cluster_idx] += 1
        
        for i, (state, cluster_idx) in enumerate(zip(all_states, cluster_labels)):
            if cluster_idx in cluster_dirs:
                state_path = os.path.join(cluster_dirs[cluster_idx], f"state_{i:03d}.png")
                try:
                    state_rgb = visualizer.rgb_from_state(state)
                    plt.figure(figsize=(5, 5))
                    plt.imshow(state_rgb)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(state_path)
                    plt.close()
                except Exception as e:
                    print(f"  保存状态 {i} 到聚类 {cluster_idx} 时出错: {e}")
        
        total_samples = len(all_states)
        for cluster_idx in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_idx]
            percentage = (count / total_samples) * 100
            if cluster_idx == -1:
                print(f"  噪声点: {count} 个样本 ({percentage:.1f}%)")
            else:
                print(f"  簇 {cluster_idx}: {count} 个样本 ({percentage:.1f}%)")
        
        print(f"聚类样本已保存到各个聚类目录中")
    
    else:
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
                    
                confidence = results[0][1]
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
    
    if use_clustering:
        return all_states, all_rewards, cluster_labels
    else:
        return all_states, all_rewards, classified_states
