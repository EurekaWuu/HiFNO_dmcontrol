import os
import sys
import torch
import numpy as np
import clip
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from einops import rearrange
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import json
from datetime import datetime
import argparse
import csv

# 路径添加，确保可以导入项目中的其他模块
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')

try:
    from env.wrappers import make_env
except ImportError:
    print("警告: 无法导入环境模块，请确保路径正确")
    make_env = None

from bisimulation_loss_1 import BisimulationLoss
from algorithms.models.HiFNO_multigpu import HierarchicalFNO, ConvResFourierLayer

# 修改：使用本地gpt2-small模型路径
GPT2_MODEL_PATH = "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/gpt2-small"

# 全局CLIP模型实例
_CLIP_MODEL = None
_CLIP_PREPROCESS = None

def get_clip_model(device=None):
    """获取共享的CLIP模型实例"""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if _CLIP_MODEL is None:
        print("正在加载全局CLIP模型: ViT-B/32")
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        _CLIP_MODEL = _CLIP_MODEL.float()
        
    return _CLIP_MODEL, _CLIP_PREPROCESS

def load_model(model_type, env, device, model_path):
    """加载训练好的模型"""
    if not model_path or not os.path.exists(model_path):
        return None
        
    print(f"正在加载训练模型: {model_path}")
    try:
        agent = torch.load(model_path, map_location=device)
        
        if hasattr(agent, 'eval'):
            agent.eval()
            
        return agent
    except Exception as e:
        print(f"加载模型错误: {e}")
        return None

def get_save_dir(args):
    """获取结果保存目录"""
    base_dir = './concept_results'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    sub_dir = f"{args.domain_name}-{args.task_name}-{args.seed}-{timestamp}"
    
    save_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"结果将保存到: {save_dir}")
    return save_dir

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='概念自适应与演化测试工具')
    parser.add_argument('--domain_name', type=str, default='walker',
                        help='环境域名')
    parser.add_argument('--task_name', type=str, default='walk',
                        help='任务名称')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='收集的episode数量')
    parser.add_argument('--steps_per_episode', type=int, default=1000,
                        help='每个episode的步数')

    parser.add_argument('--model_path', type=str, default=None,
                        help='训练好的模型路径')
    parser.add_argument('--model_type', type=str, default='svea',
                        choices=['svea', 'drq', 'hifno', 'hifno_bisim'],
                        help='模型类型')

    parser.add_argument('--episode_length', type=int, default=1000,
                      help='每个episode的长度')
    parser.add_argument('--action_repeat', type=int, default=4,
                      help='动作重复次数')
    parser.add_argument('--image_size', type=int, default=84,
                      help='观察图像大小')
    
    parser.add_argument('--n_concepts', type=int, default=6,
                      help='生成的概念数量')
    parser.add_argument('--n_variants', type=int, default=3,
                      help='每个概念的变体数量')
    parser.add_argument('--evolution_iterations', type=int, default=3,
                      help='概念演化迭代次数')
    parser.add_argument('--mutation_rate', type=float, default=0.3,
                      help='概念突变率')
    
    parser.add_argument('--save_concepts', action='store_true',
                      help='保存生成的概念')
    parser.add_argument('--transfer_concepts', action='store_true',
                      help='应用概念迁移')
                      
    parser.add_argument('--use_fourier_encoder', action='store_true',
                      help='使用傅里叶编码器处理状态')
    
    return parser.parse_args()

# =============== 1. 概念自适应与演化 ===============

class ConceptGenerator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # 初始化文本生成模型，使用本地gpt2-small
        print(f"正在加载GPT-2 Small模型: {GPT2_MODEL_PATH}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
        self.model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(device)
        
        # 使用共享的CLIP模型，而不是单独加载
        self.clip_model, self.clip_preprocess = get_clip_model(device)
        
        # 保存概念的历史演化记录
        self.concept_history = defaultdict(list)
        # 概念适应度分数
        self.concept_fitness = {}
        
    def collect_state_representations(self, states):
        """收集状态的视觉表示"""
        processed_images = []
        for state in states:
            # 确保图像是3通道的
            if state.shape[0] > 3:
                state = state[-3:]
            
            processed_image = self.clip_preprocess(Image.fromarray(
                (state.transpose(1, 2, 0) * 255).astype(np.uint8)
            )).unsqueeze(0)
            processed_images.append(processed_image)
        
        processed_images = torch.cat(processed_images).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_images)
            normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return normalized_features
    
    def generate_initial_concepts(self, states, task_name, n_concepts=6, n_variants=3):
        """根据任务特性和状态样本生成初始概念描述"""
        # 收集状态表示
        state_features = self.collect_state_representations(states)
        
        # 使用KMeans聚类找到状态的主要类别
        kmeans = KMeans(n_clusters=n_concepts, random_state=42)
        clusters = kmeans.fit_predict(state_features.cpu().numpy())
        
        # 为每个聚类中心生成描述
        concepts = []
        
        # 任务相关提示词
        task_prompts = {
            "walk": "a humanoid robot walking with",
            "run": "a humanoid robot running with", 
            "stand": "a humanoid robot standing with",
            "jump": "a humanoid robot jumping with"
        }
        
        base_prompt = task_prompts.get(task_name, f"a robot performing {task_name} with")
        
        for i in range(n_concepts):
            cluster_states = [states[j] for j in range(len(states)) if clusters[j] == i]
            if not cluster_states:
                continue
                
            # 为每个概念生成多个变体
            concept_variants = []
            for _ in range(n_variants):
                prompt = f"Describe {base_prompt} specific body posture and movement in one detailed sentence:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    inputs["input_ids"], 
                    max_length=100, 
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )
                
                description = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # 提取生成的描述部分
                if ":" in description:
                    description = description.split(":", 1)[1].strip()
                
                concept_variants.append(description)
            
            # 记录概念演化历史
            concept_id = f"concept_{i}"
            self.concept_history[concept_id].append(concept_variants)
            concepts.append(concept_variants)
            
        return concepts
    
    def evaluate_concept_fitness(self, concepts, states, actions):
        """评估概念的适应度，基于CLIP相似度和状态-动作一致性"""
        # 将概念扁平化为一个列表
        flat_concepts = []
        concept_indices = []
        for i, variants in enumerate(concepts):
            for variant in variants:
                flat_concepts.append(variant)
                concept_indices.append(i)
        
        # 编码所有文本描述
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in flat_concepts]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=1)
        
        # 收集状态表示
        state_features = self.collect_state_representations(states)
        
        # 计算每个状态与每个概念的相似度
        similarities = torch.mm(state_features, text_features.T)
        
        # 为每个概念计算适应度分数
        fitness_scores = {}
        for concept_idx in set(concept_indices):
            # 获取该概念的所有变体索引
            variant_indices = [i for i, idx in enumerate(concept_indices) if idx == concept_idx]
            
            # 该概念的最大相似度
            concept_similarities = similarities[:, variant_indices].max(dim=1)[0]
            
            # 计算该概念的状态分组
            concept_states = []
            concept_actions = []
            for i in range(len(states)):
                if torch.argmax(similarities[i]) in variant_indices:
                    concept_states.append(state_features[i])
                    concept_actions.append(actions[i])
            
            if not concept_states:
                fitness_scores[f"concept_{concept_idx}"] = 0.0
                continue
                
            # 将列表转换为张量
            concept_states = torch.stack(concept_states)
            concept_actions = torch.tensor(concept_actions).to(self.device)
            
            # 计算状态内部一致性 - 状态之间的平均相似度
            if len(concept_states) > 1:
                state_sim_matrix = torch.mm(concept_states, concept_states.T)
                state_consistency = (state_sim_matrix.sum() - state_sim_matrix.trace()) / (len(concept_states) * (len(concept_states) - 1))
            else:
                state_consistency = torch.tensor(1.0)
            
            # 计算动作内部一致性 - 动作之间的平均相似度
            if len(concept_actions) > 1:
                action_sim_matrix = F.cosine_similarity(concept_actions.unsqueeze(1), concept_actions.unsqueeze(0), dim=2)
                action_consistency = (action_sim_matrix.sum() - action_sim_matrix.trace()) / (len(concept_actions) * (len(concept_actions) - 1))
            else:
                action_consistency = torch.tensor(1.0)
            
            # 综合得分 = 平均相似度 * 状态一致性 * 动作一致性
            avg_similarity = concept_similarities.mean()
            fitness = avg_similarity * state_consistency * action_consistency
            
            fitness_scores[f"concept_{concept_idx}"] = fitness.item()
            
        self.concept_fitness.update(fitness_scores)
        return fitness_scores
    
    def evolve_concepts(self, concepts, fitness_scores, states, mutation_rate=0.3):
        """基于适应度分数，演化概念描述"""
        new_concepts = []
        
        # 获取每个概念的最佳变体
        best_variants = {}
        flat_concepts = []
        concept_indices = []
        
        for i, variants in enumerate(concepts):
            for variant in variants:
                flat_concepts.append(variant)
                concept_indices.append(i)
        
        # 编码所有文本描述
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in flat_concepts]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=1)
        
        # 收集状态表示
        state_features = self.collect_state_representations(states)
        
        # 计算每个状态与每个概念的相似度
        similarities = torch.mm(state_features, text_features.T)
        
        for concept_idx in range(len(concepts)):
            concept_id = f"concept_{concept_idx}"
            
            # 获取该概念的所有变体索引
            variant_indices = [i for i, idx in enumerate(concept_indices) if idx == concept_idx]
            
            # 计算每个变体的平均相似度
            variant_scores = []
            for i, vidx in enumerate(variant_indices):
                score = similarities[:, vidx].mean().item()
                variant_scores.append((score, flat_concepts[vidx]))
            
            # 保留适应度最高的变体
            variant_scores.sort(reverse=True)
            best_variants[concept_id] = variant_scores[0][1]
        
        for concept_idx in range(len(concepts)):
            concept_id = f"concept_{concept_idx}"
            fitness = fitness_scores.get(concept_id, 0)
            
            # 如果适应度较高，保留原始概念
            if fitness > 0.5:
                new_variants = concepts[concept_idx].copy()
            else:
                # 否则，生成新变体替换低适应度变体
                best_variant = best_variants[concept_id]
                new_variants = [best_variant]
                
                # 根据突变率决定是否生成新变体
                if np.random.random() < mutation_rate:
                    prompt = f"Rewrite and improve this description of a robot pose: '{best_variant}'. Be more precise and detailed:"
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    output = self.model.generate(
                        inputs["input_ids"], 
                        max_length=100, 
                        num_return_sequences=2,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    for out in output:
                        description = self.tokenizer.decode(out, skip_special_tokens=True)
                        if ":" in description:
                            description = description.split(":", 1)[1].strip()
                        if description != best_variant and description not in new_variants:
                            new_variants.append(description)
                
                # 确保每个概念有至少3个变体
                while len(new_variants) < 3:
                    new_variants.append(best_variant)
            
            new_concepts.append(new_variants)
            # 记录概念演化历史
            self.concept_history[concept_id].append(new_variants)
            
        return new_concepts
    
    def save_concepts(self, concepts, file_path):
        """保存概念到文件"""
        with open(file_path, 'w') as f:
            json.dump(concepts, f, indent=2)
    
    def load_concepts(self, file_path):
        """从文件加载概念"""
        with open(file_path, 'r') as f:
            return json.load(f)


# =============== 2. 频域与语义空间融合优化 ===============

class SemanticGuidedFourierLayer(ConvResFourierLayer):
    """语义引导的傅里叶层，融合频域特征与CLIP语义空间"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, truncation_size=16, 
                 clip_dim=512, semantic_weight=0.5, device=None):
        super().__init__(in_channels, out_channels, kernel_size, truncation_size)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.semantic_weight = semantic_weight
        
        # 语义引导注意力机制
        self.semantic_attention = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, truncation_size * truncation_size)
        )
        
        # 频域特征映射到语义空间
        self.freq_to_semantic = nn.Sequential(
            nn.Linear(truncation_size * truncation_size, 256),
            nn.ReLU(),
            nn.Linear(256, clip_dim)
        )
        
        # 使用共享的CLIP模型
        self.clip_model, _ = get_clip_model(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def compute_semantic_attention(self, concept_features, x_fft):
        """根据概念特征计算语义注意力权重"""
        # 将频域特征映射到向量
        B = x_fft.size(0)
        x_fft_flat = x_fft.view(B, self.in_channels, -1)
        
        # 将频域特征映射到语义空间
        freq_semantic = self.freq_to_semantic(x_fft_flat.mean(dim=1))
        
        # 计算频域特征与概念特征的相似度
        similarity = F.cosine_similarity(freq_semantic.unsqueeze(1), 
                                        concept_features.unsqueeze(0), dim=2)
        
        # 选择最相似的概念
        best_concept_idx = similarity.max(dim=1)[1]
        selected_concepts = concept_features[best_concept_idx]
        
        # 生成注意力权重
        attention_weights = self.semantic_attention(selected_concepts)
        attention_weights = attention_weights.view(B, 1, self.truncation_size, self.truncation_size)
        attention_weights = torch.sigmoid(attention_weights)
        
        return attention_weights
    
    def forward(self, x, concept_features=None):
        """前向传播，加入语义引导的注意力机制"""
        self._update_conv(x)
        conv_out = self.conv(x)
        
        # 计算傅里叶变换
        x_fft = torch.fft.fftn(x, dim=tuple(range(2, x.dim())))
        x_fft = x_fft[..., :self.truncation_size, :self.truncation_size]
        
        # 如果提供了概念特征，则应用语义引导
        if concept_features is not None:
            # 计算语义注意力权重
            attention_weights = self.compute_semantic_attention(concept_features, x_fft)
            
            # 应用注意力权重到频域特征
            x_fft = x_fft * (1 + self.semantic_weight * attention_weights)
        
        # 应用卷积核的傅里叶变换
        pad_sizes = []
        for dim_size in x.shape[2:]:
            pad_sizes.extend([0, dim_size - self.conv.weight.shape[2]])
        weight_padded = F.pad(self.conv.weight, pad_sizes[::-1])
        
        kernel_fft = torch.fft.fftn(weight_padded, dim=tuple(range(2, weight_padded.dim())))
        kernel_fft = kernel_fft[..., :self.truncation_size, :self.truncation_size]
        
        # 调整维度以进行批量卷积
        B = x_fft.size(0)
        kernel_fft = kernel_fft.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # 在频域中应用卷积
        out_fft = torch.einsum('bcxy,bkcxy->bkxy', x_fft, kernel_fft)
        
        # 逆傅里叶变换
        out = torch.fft.ifftn(out_fft, s=x.shape[2:], dim=tuple(range(2, out_fft.dim())))
        out = out.real
        
        return out + conv_out + self.bias


class SemanticFourierEncoder(nn.Module):
    """融合语义信息的傅里叶编码器"""
    
    def __init__(self, img_size, in_channels, embed_dim, clip_dim=512, 
                 num_fourier_layers=3, truncation_sizes=None, device=None):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.clip_dim = clip_dim
        
        if truncation_sizes is None:
            truncation_sizes = [16, 12, 8]
        
        # 初始卷积层
        self.init_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # 语义引导的傅里叶层
        self.fourier_layers = nn.ModuleList([
            SemanticGuidedFourierLayer(
                embed_dim, embed_dim, 
                truncation_size=truncation_sizes[i % len(truncation_sizes)],
                clip_dim=clip_dim,
                device=device
            )
            for i in range(num_fourier_layers)
        ])
        
        # 输出映射
        self.output_map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # 使用共享的CLIP模型
        self.clip_model, _ = get_clip_model(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def encode_concepts(self, concepts):
        """将概念编码为特征向量"""
        # 将概念扁平化为一个列表
        flat_concepts = []
        for variants in concepts:
            flat_concepts.extend(variants)
        
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in flat_concepts]).to(self.device)
            concept_features = self.clip_model.encode_text(text_inputs)
            concept_features = F.normalize(concept_features, dim=1)
        
        return concept_features
    
    def forward(self, x, concepts=None):
        """前向传播，融合语义信息"""
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 预处理输入
        if len(x.shape) == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # 添加batch维度
        
        # 编码概念（如果提供）
        concept_features = None
        if concepts is not None:
            concept_features = self.encode_concepts(concepts)
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 应用傅里叶层
        for layer in self.fourier_layers:
            x = layer(x, concept_features)
        
        # 输出映射
        x = self.output_map(x)
        
        return x


# =============== 3. 跨任务概念迁移 ===============

class ConceptTransferModule:
    """跨任务概念迁移模块"""
    
    def __init__(self, base_path="./concept_library", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # 创建概念库目录
        self.library_path = os.path.join(base_path, "concepts")
        self.metadata_path = os.path.join(base_path, "metadata.json")
        os.makedirs(self.library_path, exist_ok=True)
        
        # 初始化元数据
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "tasks": {},
                "concepts": {},
                "transfer_history": []
            }
        
        # 使用共享的CLIP模型
        self.clip_model, _ = get_clip_model(device)
    
    def save_task_concepts(self, task_name, domain_name, concepts, fitness_scores=None):
        """保存任务相关的概念到库中"""
        # 创建任务目录
        task_path = os.path.join(self.library_path, f"{domain_name}_{task_name}")
        os.makedirs(task_path, exist_ok=True)
        
        # 保存概念文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        concept_file = os.path.join(task_path, f"concepts_{timestamp}.json")
        
        concept_data = {
            "concepts": concepts,
            "fitness": fitness_scores if fitness_scores else {},
            "timestamp": timestamp
        }
        
        with open(concept_file, 'w') as f:
            json.dump(concept_data, f, indent=2)
        
        # 更新元数据
        if task_name not in self.metadata["tasks"]:
            self.metadata["tasks"][task_name] = {
                "domain": domain_name,
                "concept_files": []
            }
        
        self.metadata["tasks"][task_name]["concept_files"].append(concept_file)
        
        # 将概念添加到全局概念库
        for i, variants in enumerate(concepts):
            concept_id = f"{domain_name}_{task_name}_concept_{i}"
            self.metadata["concepts"][concept_id] = {
                "task": task_name,
                "domain": domain_name,
                "variants": variants,
                "fitness": fitness_scores.get(f"concept_{i}", 0) if fitness_scores else 0
            }
        
        # 保存更新后的元数据
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def find_similar_concepts(self, task_name, query_concepts, top_k=3):
        """查找与查询概念最相似的概念"""
        # 如果任务库为空，返回原始概念
        if not self.metadata["concepts"]:
            return query_concepts
        
        # 编码查询概念
        query_texts = []
        for variants in query_concepts:
            query_texts.extend(variants)
        
        with torch.no_grad():
            query_inputs = torch.cat([clip.tokenize(text) for text in query_texts]).to(self.device)
            query_features = self.clip_model.encode_text(query_inputs)
            query_features = F.normalize(query_features, dim=1)
        
        # 从库中获取所有概念
        library_concepts = []
        concept_ids = []
        concept_tasks = []
        
        for concept_id, concept_data in self.metadata["concepts"].items():
            # 跳过来自相同任务的概念
            if concept_data["task"] == task_name:
                continue
                
            for variant in concept_data["variants"]:
                library_concepts.append(variant)
                concept_ids.append(concept_id)
                concept_tasks.append(concept_data["task"])
        
        if not library_concepts:
            return query_concepts
        
        # 编码库中的概念
        with torch.no_grad():
            library_inputs = torch.cat([clip.tokenize(text) for text in library_concepts]).to(self.device)
            library_features = self.clip_model.encode_text(library_inputs)
            library_features = F.normalize(library_features, dim=1)
        
        # 计算查询概念与库中概念的相似度
        similarities = torch.mm(query_features, library_features.T)
        
        # 为每个查询概念找到最相似的库中概念
        similar_concepts = []
        transfer_history = []
        
        for i, variants in enumerate(query_concepts):
            variant_indices = list(range(i * len(variants), (i + 1) * len(variants)))
            variant_features = query_features[variant_indices]
            
            # 计算平均相似度
            avg_similarities = similarities[variant_indices].mean(dim=0)
            
            # 找到top_k个最相似的概念
            top_indices = avg_similarities.topk(min(top_k, len(library_concepts))).indices.cpu().numpy()
            selected_concepts = [library_concepts[idx] for idx in top_indices]
            selected_ids = [concept_ids[idx] for idx in top_indices]
            selected_tasks = [concept_tasks[idx] for idx in top_indices]
            
            # 合并原始变体和相似概念
            merged_variants = variants.copy()
            for concept in selected_concepts:
                if concept not in merged_variants:
                    merged_variants.append(concept)
            
            similar_concepts.append(merged_variants)
            
            # 记录迁移历史
            transfer_history.append({
                "query_concept": f"{task_name}_concept_{i}",
                "similar_concepts": [{"id": selected_ids[j], "task": selected_tasks[j]} 
                                    for j in range(len(selected_ids))],
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
            })
        
        # 更新元数据中的迁移历史
        self.metadata["transfer_history"].extend(transfer_history)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return similar_concepts
    
    def get_progressive_concepts(self, source_task, target_task, adaptation_level=0.5):
        """渐进式概念迁移，根据适应级别混合源任务和目标任务的概念"""
        # 获取源任务概念
        source_concepts = []
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] == source_task:
                source_concepts.append(concept_data["variants"])
        
        # 获取目标任务概念
        target_concepts = []
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] == target_task:
                target_concepts.append(concept_data["variants"])
        
        # 如果源任务或目标任务没有概念，返回空列表
        if not source_concepts or not target_concepts:
            return []
        
        # 匹配源任务和目标任务的概念
        matched_pairs = []
        
        # 编码所有源概念
        source_texts = []
        for variants in source_concepts:
            source_texts.extend(variants)
        
        with torch.no_grad():
            source_inputs = torch.cat([clip.tokenize(text) for text in source_texts]).to(self.device)
            source_features = self.clip_model.encode_text(source_inputs)
            source_features = F.normalize(source_features, dim=1)
        
        # 编码所有目标概念
        target_texts = []
        for variants in target_concepts:
            target_texts.extend(variants)
        
        with torch.no_grad():
            target_inputs = torch.cat([clip.tokenize(text) for text in target_texts]).to(self.device)
            target_features = self.clip_model.encode_text(target_inputs)
            target_features = F.normalize(target_features, dim=1)
        
        # 计算源概念和目标概念之间的相似度
        source_variant_counts = [len(variants) for variants in source_concepts]
        target_variant_counts = [len(variants) for variants in target_concepts]
        
        source_offsets = [0] + list(np.cumsum(source_variant_counts[:-1]))
        target_offsets = [0] + list(np.cumsum(target_variant_counts[:-1]))
        
        # 计算每个源概念和目标概念的平均相似度
        for i, source_variants in enumerate(source_concepts):
            source_indices = list(range(source_offsets[i], source_offsets[i] + len(source_variants)))
            source_concept_features = source_features[source_indices].mean(dim=0, keepdim=True)
            
            best_similarity = -1
            best_target_idx = -1
            
            for j, target_variants in enumerate(target_concepts):
                target_indices = list(range(target_offsets[j], target_offsets[j] + len(target_variants)))
                target_concept_features = target_features[target_indices].mean(dim=0, keepdim=True)
                
                similarity = F.cosine_similarity(source_concept_features, target_concept_features).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target_idx = j
            
            if best_target_idx >= 0:
                matched_pairs.append((i, best_target_idx, best_similarity))
        
        # 根据适应级别生成渐进式概念
        progressive_concepts = []
        for source_idx, target_idx, similarity in matched_pairs:
            source_variants = source_concepts[source_idx]
            target_variants = target_concepts[target_idx]
            
            # 根据适应级别混合源概念和目标概念
            num_source = max(1, int(len(source_variants) * (1 - adaptation_level)))
            num_target = max(1, int(len(target_variants) * adaptation_level))
            
            mixed_variants = source_variants[:num_source] + target_variants[:num_target]
            progressive_concepts.append(mixed_variants)
        
        return progressive_concepts
    
    def generate_hybrid_concepts(self, task_name, concepts, num_variants=3):
        """生成混合概念，融合给定概念和库中相似概念的特性"""
        # 如果概念库为空，返回原始概念
        if not self.metadata["concepts"]:
            return concepts
        
        # 找到与给定任务相关的所有概念
        task_related_concepts = []
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] != task_name:  # 排除相同任务的概念
                task_related_concepts.extend(concept_data["variants"])
        
        if not task_related_concepts:
            return concepts
            
        # 初始化文本生成器，使用本地gpt2-small
        print(f"为混合概念生成加载GPT-2 Small模型: {GPT2_MODEL_PATH}")
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(self.device)
        
        hybrid_concepts = []
        for variants in concepts:
            # 随机选择一个原始变体作为基础
            base_variant = np.random.choice(variants)
            
            # 随机选择一些库中的概念进行混合
            num_to_mix = min(3, len(task_related_concepts))
            concepts_to_mix = np.random.choice(task_related_concepts, num_to_mix, replace=False)
            
            # 生成混合概念
            new_variants = [base_variant]
            
            for _ in range(num_variants - 1):
                # 随机选择一个库中的概念
                lib_concept = np.random.choice(concepts_to_mix)
                
                prompt = f"Combine these two descriptions of robot poses into one detailed sentence:\n1. {base_variant}\n2. {lib_concept}\nCombined description:"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                output = model.generate(
                    inputs["input_ids"], 
                    max_length=100, 
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )
                
                hybrid = tokenizer.decode(output[0], skip_special_tokens=True)
                if "Combined description:" in hybrid:
                    hybrid = hybrid.split("Combined description:", 1)[1].strip()
                
                if hybrid not in new_variants:
                    new_variants.append(hybrid)
            
            # 确保有足够的变体
            while len(new_variants) < num_variants:
                new_variants.append(base_variant)
                
            hybrid_concepts.append(new_variants)
        
        return hybrid_concepts


# =============== 使用示例 ===============

def save_concepts_to_csv(concepts, fitness_scores, iteration, file_path):
    """将概念集合保存为CSV格式"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # 写入模式：如果文件不存在则创建并写入标题，否则直接追加
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果是新文件，写入标题行
        if not file_exists:
            writer.writerow(["迭代", "概念ID", "变体ID", "变体描述", "适应度"])
        
        # 写入概念数据
        for i, variants in enumerate(concepts):
            concept_id = f"concept_{i}"
            fitness = fitness_scores.get(concept_id, 0.0) if fitness_scores else 0.0
            
            for j, variant in enumerate(variants):
                writer.writerow([iteration, concept_id, j, variant, f"{fitness:.4f}"])

def example_usage(args):
    """展示如何使用上述实现的功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 预先加载全局CLIP模型
    clip_model, clip_preprocess = get_clip_model(device)
    print("已加载全局CLIP模型，将在各组件中共享使用")
    
    # 创建环境
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
    print(f"已创建环境: {args.domain_name}_{args.task_name}")
    
    # 加载训练好的模型
    model = None
    if args.model_path and args.model_type:
        model = load_model(args.model_type, env, device, args.model_path)
        print(f"已加载 {args.model_type} 模型: {args.model_path}")
    
    # 创建保存目录
    save_dir = get_save_dir(args)
    
    # 收集状态和动作数据
    print(f"\n收集数据中，计划采集 {args.num_episodes} 个episode，每个 {args.steps_per_episode} 步...")
    all_states = []
    all_actions = []
    
    for episode in range(args.num_episodes):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        episode_states = []
        episode_actions = []
        
        if model:  # 使用训练好的模型
            while not done and step < args.steps_per_episode:
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
                                
                            obs = torch.FloatTensor(rgb_state).to(device)
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
                
                episode_states.append(next_state)
                episode_actions.append(action)
                
                state = next_state
                total_reward += reward
                step += 1
                
                if step % 100 == 0:
                    print(f"  步骤 {step}/{args.steps_per_episode}，当前奖励: {total_reward:.2f}")
        else:  # 使用随机动作
            while not done and step < args.steps_per_episode:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                
                episode_states.append(next_state)
                episode_actions.append(action)
                
                state = next_state
                total_reward += reward
                step += 1
                
                if step % 100 == 0:
                    print(f"  步骤 {step}/{args.steps_per_episode}，当前奖励: {total_reward:.2f}")
        
        print(f"  Episode {episode+1} 完成，总奖励: {total_reward:.2f}，收集了 {len(episode_states)} 个状态")
        
        # 采样状态以减少数据量
        if len(episode_states) > 50:
            indices = np.linspace(0, len(episode_states) - 1, 50, dtype=int)
            episode_states = [episode_states[i] for i in indices]
            episode_actions = [episode_actions[i] for i in indices]
        
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
    
    print(f"总共收集了 {len(all_states)} 个状态-动作对")
    
    # 1. 概念自适应与演化
    print("\n===== 1. 概念自适应与演化 =====")
    concept_generator = ConceptGenerator(device=device)
    
    # 创建CSV文件路径
    csv_file = f"{args.domain_name}_{args.task_name}_concept_evolution.csv"
    print(f"概念演化记录将保存到: {csv_file}")
    
    # 如果文件已存在，先删除
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    # 生成初始概念
    print("生成初始概念...")
    initial_concepts = concept_generator.generate_initial_concepts(
        states=all_states,
        task_name=args.task_name,
        n_concepts=args.n_concepts,
        n_variants=args.n_variants
    )
    
    # 打印初始概念
    print("\n初始概念:")
    for i, variants in enumerate(initial_concepts):
        print(f"概念 {i+1}:")
        for j, variant in enumerate(variants):
            print(f"  变体 {j+1}: {variant}")
    
    # 评估概念适应度
    print("\n评估概念适应度...")
    fitness_scores = concept_generator.evaluate_concept_fitness(initial_concepts, all_states, all_actions)
    
    print("适应度分数:")
    for concept_id, score in fitness_scores.items():
        print(f"{concept_id}: {score:.4f}")
    
    # 保存初始概念到CSV
    save_concepts_to_csv(initial_concepts, fitness_scores, 0, csv_file)
    
    # 多次演化概念
    concepts = initial_concepts
    for i in range(args.evolution_iterations):
        print(f"\n执行第 {i+1}/{args.evolution_iterations} 次概念演化...")
        concepts = concept_generator.evolve_concepts(
            concepts=concepts,
            fitness_scores=fitness_scores,
            states=all_states,
            mutation_rate=args.mutation_rate
        )
        
        # 再次评估适应度
        fitness_scores = concept_generator.evaluate_concept_fitness(concepts, all_states, all_actions)
        
        # 保存当前迭代的概念到CSV
        save_concepts_to_csv(concepts, fitness_scores, i+1, csv_file)
        
        print(f"第 {i+1} 次演化后的适应度分数:")
        for concept_id, score in fitness_scores.items():
            print(f"{concept_id}: {score:.4f}")
    
    # 打印最终概念
    print("\n最终概念:")
    for i, variants in enumerate(concepts):
        print(f"概念 {i+1}:")
        for j, variant in enumerate(variants):
            print(f"  变体 {j+1}: {variant}")
    
    print(f"\n概念演化记录已保存到: {csv_file}")
    
    # 保存概念
    if args.save_concepts:
        concepts_file = os.path.join(save_dir, f"{args.domain_name}_{args.task_name}_concepts.json")
        concept_generator.save_concepts(concepts, concepts_file)
        print(f"概念已保存到: {concepts_file}")
    
    # 2. 概念迁移 (如果启用)
    if args.transfer_concepts:
        print("\n===== 2. 跨任务概念迁移 =====")
        transfer_module = ConceptTransferModule(base_path=os.path.join(save_dir, "concept_library"), device=device)
        
        # 保存当前任务的概念到库
        transfer_module.save_task_concepts(args.task_name, args.domain_name, concepts, fitness_scores)
        
        # 查找相似概念
        print("\n查找相似概念...")
        similar_concepts = transfer_module.find_similar_concepts(args.task_name, concepts, top_k=3)
        
        # 打印相似概念
        print("\n相似概念:")
        for i, variants in enumerate(similar_concepts):
            print(f"概念 {i+1}:")
            for j, variant in enumerate(variants):
                print(f"  变体 {j+1}: {variant}")
        
        # 生成混合概念
        print("\n生成混合概念...")
        hybrid_concepts = transfer_module.generate_hybrid_concepts(args.task_name, concepts, num_variants=args.n_variants)
        
        # 打印混合概念
        print("\n混合概念:")
        for i, variants in enumerate(hybrid_concepts):
            print(f"概念 {i+1}:")
            for j, variant in enumerate(variants):
                print(f"  变体 {j+1}: {variant}")
    
    # 3. 使用频域与语义空间融合 (如果启用)
    if args.use_fourier_encoder:
        print("\n===== 3. 使用频域与语义空间融合 =====")
        
        # 创建编码器
        encoder = SemanticFourierEncoder(
            img_size=args.image_size,
            in_channels=all_states[0].shape[0],
            embed_dim=256,
            clip_dim=512,
            num_fourier_layers=3,
            device=device
        )
        
        # 编码状态
        print("使用语义引导的傅里叶编码器处理状态...")
        
        # 采样部分状态以减少计算量
        sample_indices = np.random.choice(len(all_states), min(10, len(all_states)), replace=False)
        sample_states = [all_states[i] for i in sample_indices]
        
        # 不使用概念的编码
        print("不使用概念引导的编码:")
        encoded_states = [encoder(state).cpu().detach().numpy() for state in sample_states]
        print(f"编码形状: {encoded_states[0].shape}")
        
        # 使用概念的编码
        print("使用概念引导的编码:")
        encoded_states_with_concepts = [encoder(state, concepts).cpu().detach().numpy() for state in sample_states]
        print(f"编码形状: {encoded_states_with_concepts[0].shape}")
        
        # 比较编码差异
        diffs = [np.mean(np.abs(e1 - e2)) for e1, e2 in zip(encoded_states, encoded_states_with_concepts)]
        avg_diff = np.mean(diffs)
        print(f"概念引导带来的平均编码差异: {avg_diff:.6f}")
    
    print("\n演示完成!")

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    example_usage(args)

if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=1 python gen_concepts.py \
    --domain_name walker \
    --task_name walk \
    --seed 42 \
    --model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --model_type svea \
    --num_episodes 2 \
    --steps_per_episode 1000 \
    --n_concepts 6 \
    --n_variants 3 \
    --evolution_iterations 3 \
    --save_concepts

"""