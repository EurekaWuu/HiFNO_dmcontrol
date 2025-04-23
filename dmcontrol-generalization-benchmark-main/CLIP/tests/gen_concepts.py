import os
import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')
import torch
import numpy as np
import clip
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from einops import rearrange
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import json
from datetime import datetime
import argparse
import csv
import re
import random



try:
    from env.wrappers import make_env
except ImportError:
    print("警告: 无法导入环境模块，请确保路径正确")
    make_env = None

from bisimulation_loss_1 import BisimulationLoss
from algorithms.models.HiFNO_multigpu import HierarchicalFNO, ConvResFourierLayer

GPT2_MODEL_PATH = "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/gpt2-small"

_CLIP_MODEL = None
_CLIP_PREPROCESS = None

def get_clip_model(device=None):
    global _CLIP_MODEL, _CLIP_PREPROCESS
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if _CLIP_MODEL is None:
        print("正在加载全局CLIP模型: ViT-B/32")
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        _CLIP_MODEL = _CLIP_MODEL.float()
        
    return _CLIP_MODEL, _CLIP_PREPROCESS

def load_model(model_type, env, device, model_path):
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
    base_dir = './concept_results'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    sub_dir = f"{args.domain_name}-{args.task_name}-{args.seed}-{timestamp}"
    
    save_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"结果将保存到: {save_dir}")
    return save_dir

def parse_args():
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

# =============== 概念自适应演化 ===============

class ConceptGenerator:
    def __init__(self, model_path, device=None, img_feat_ext=None, prompt_type='text'):
        if model_path:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("已设置tokenizer.pad_token = tokenizer.eos_token")
        
        if device is not None:
            self.device = device
            if hasattr(self, 'model'):
                self.model.to(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self, 'model'):
                self.model.to(self.device)
        self.img_feat_ext = img_feat_ext
        self.prompt_type = prompt_type
        
        self.clip_model, self.clip_preprocess = get_clip_model(device)
        
        self.concept_history = defaultdict(list)
        self.concept_fitness = {}
        
    def collect_state_representations(self, states):
        processed_images = []
        for state in states:
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
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
    
    def generate_initial_concepts(self, n_concepts, n_variations=3):
        concepts = {}
        
        robot_prompts = [
            "Describe a humanoid robot's walking posture, focusing on joint positions and leg movements: ",
            "Write a short description of robot walking stance, detailing leg positions and joint angles: ",
            "Explain how the robot's legs and joints are positioned during walking (max 15 words): ",
            "Describe the specific leg posture of a walking humanoid robot: ",
            "Detail the joint angles and leg positions of a humanoid robot in walking motion: "
        ]
        
        backup_descriptions = [
            "Robot with only one leg touching the ground while other leg is suspended in mid-stride motion.",
            "Humanoid with left leg planted firmly while right leg swings forward with bent knee.",
            "Robot balancing on one leg only, with second leg clearly lifted from the ground.",
            "Humanoid with left knee raised high and right leg extended backward in walking motion.",
            "Robot with one foot firmly planted while other leg is mid-stride with bent knee joint.",
            "Bipedal robot in mid-walking pose with single leg support and other leg in swing phase.",
            "Robot maintaining balance on right leg while left leg executes forward stepping motion.",
            "Humanoid with knees bent at optimal walking angle and one leg lifted from ground.",
            "Robot demonstrating controlled gait with precise joint angles in single-leg stance.",
            "Robot with both legs simultaneously contacting the ground in double-support phase.",
            "Humanoid robot in stable standing posture with weight distributed evenly on both feet.",
            "Robot with both knees slightly bent and feet parallel on ground in ready position."
        ]
        
        posture_keywords = ["leg", "knee", "joint", "foot", "feet", "stance", "posture", "gait", 
                           "balance", "walking", "step", "stride", "support", "motion", "angle"]
        robot_keywords = ["robot", "humanoid", "bipedal", "mechanical", "machine"]
        
        try:
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print(f"加载本地GPT-2模型: {GPT2_MODEL_PATH}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if not hasattr(self, 'gpt2') or self.gpt2 is None:
                self.gpt2 = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(self.device)
        except Exception as e:
            print(f"无法加载GPT-2模型: {e}")
            print("将使用备用描述...")
            
            print("生成初始概念变体...")
            for i in range(n_concepts):
                concept_id = f"concept_{i}"
                concepts[concept_id] = []
                
                for j in range(n_variations):
                    description = random.choice(backup_descriptions)
                    concepts[concept_id].append(description)
                    print(f"  概念 {i+1}, 变体 {j+1}: {description}")
            
            return concepts
        
        print("生成初始概念变体...")
        for i in range(n_concepts):
            concept_id = f"concept_{i}"
            concepts[concept_id] = []
            
            for j in range(n_variations):
                prompt = random.choice(robot_prompts)
                
                try:
                    input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                    attention_mask = torch.ones(input_ids.shape, device=self.device)
                    
                    output = self.gpt2.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                        temperature=1.2,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=2.0,
                        do_sample=True,
                    num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    description = generated_text[len(prompt):].strip()
                    
                    description = re.sub(r'["\']+', '', description)
                    description = re.sub(r'^\s*[-:]+\s*', '', description)
                    
                    if '.' in description:
                        description = description.split('.')[0].strip() + '.'
                    elif '!' in description:
                        description = description.split('!')[0].strip() + '!'
                    elif '?' in description:
                        description = description.split('?')[0].strip() + '?'
                    
                    for template in robot_prompts:
                        description = description.replace(template, "")
                    
                    words = description.split()
                    if len(words) < 5:
                        description = random.choice(backup_descriptions)
                        print(f"  使用备用描述: {description[:30]}...")
                    else:
                        has_posture_keyword = any(keyword in description.lower() for keyword in posture_keywords)
                        has_robot_keyword = any(keyword in description.lower() for keyword in robot_keywords)
                        
                        if not (has_posture_keyword and has_robot_keyword):
                            description = random.choice(backup_descriptions)
                            print(f"  使用备用描述(缺少关键词): {description[:30]}...")
                    
                    words = description.split()
                    if len(words) > 20:
                        description = " ".join(words[:20]) + "."
                    
                    try:
                        clip_tokens = clip.tokenize([description])
                        if clip_tokens.shape[1] > 77:
                            while len(words) > 15 and clip.tokenize([" ".join(words)]).shape[1] > 77:
                                words = words[:-1]
                            description = " ".join(words) + "."
                    except Exception as e:
                        description = random.choice(backup_descriptions)
                        print(f"  CLIP处理失败，使用备用描述: {description[:30]}...")
                    
                    if self._similarity_too_high(description, [var for var in concepts[concept_id]]):
                        backup = random.choice(backup_descriptions)
                        while self._similarity_too_high(backup, [var for var in concepts[concept_id]]):
                            backup = random.choice(backup_descriptions)
                        description = backup
                        print(f"  使用备用描述(避免重复): {description[:30]}...")
                    
                    concepts[concept_id].append(description)
                    print(f"  概念 {i+1}, 变体 {j+1}: {description}")
                    
                except Exception as e:
                    print(f"  生成变体时出错: {e}")
                    backup = random.choice(backup_descriptions)
                    while self._similarity_too_high(backup, [var for var in concepts[concept_id]]):
                        backup = random.choice(backup_descriptions)
                    concepts[concept_id].append(backup)
                    print(f"  使用备用描述(异常处理): {backup[:30]}...")
            
        return concepts
    
    def _similarity_too_high(self, new_var, existing_vars, threshold=0.8):
        new_words = set(new_var.lower().split())
        
        for var in existing_vars:
            existing_words = set(var.lower().split())
            
            if len(new_words) > 0 and len(existing_words) > 0:
                overlap = len(new_words.intersection(existing_words))
                union = len(new_words.union(existing_words))
                similarity = overlap / union
                
                if similarity > threshold:
                    return True
        
        return False
    
    def evaluate_concept_fitness(self, concepts, states, actions):
        flat_concepts = []
        concept_indices = []
        for concept_id, variants in concepts.items():
            for variant in variants:
                flat_concepts.append(variant)
                idx = int(concept_id.split('_')[1])
                concept_indices.append(idx)
        
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in flat_concepts]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=1)
        
        state_features = self.collect_state_representations(states)
        
        similarities = torch.mm(state_features, text_features.T)
        
        fitness_scores = {}
        for concept_idx in set(concept_indices):
            variant_indices = [i for i, idx in enumerate(concept_indices) if idx == concept_idx]
            
            concept_similarities = similarities[:, variant_indices].max(dim=1)[0]
            
            concept_states = []
            concept_actions = []
            for i in range(len(states)):
                if torch.argmax(similarities[i]) in variant_indices:
                    concept_states.append(state_features[i])
                    concept_actions.append(actions[i])
            
            if not concept_states:
                fitness_scores[f"concept_{concept_idx}"] = 0.0
                continue
                
            concept_states = torch.stack(concept_states)
            concept_actions = torch.tensor(concept_actions).to(self.device)
            
            if len(concept_states) > 1:
                state_sim_matrix = torch.mm(concept_states, concept_states.T)
                state_consistency = (state_sim_matrix.sum() - state_sim_matrix.trace()) / (len(concept_states) * (len(concept_states) - 1))
            else:
                state_consistency = torch.tensor(1.0)
            
            if len(concept_actions) > 1:
                action_sim_matrix = F.cosine_similarity(concept_actions.unsqueeze(1), concept_actions.unsqueeze(0), dim=2)
                action_consistency = (action_sim_matrix.sum() - action_sim_matrix.trace()) / (len(concept_actions) * (len(concept_actions) - 1))
            else:
                action_consistency = torch.tensor(1.0)
            
            avg_similarity = concept_similarities.mean()
            fitness = avg_similarity * state_consistency * action_consistency
            
            fitness_scores[f"concept_{concept_idx}"] = fitness.item()
            
        self.concept_fitness.update(fitness_scores)
        return fitness_scores
    
    def evolve_concepts(self, concepts, fitness_scores, n_variants=3):
        evolved_concepts = {}
        
        robot_prompts = [
            "Describe a humanoid robot's walking posture, focusing on joint positions and leg movements: ",
            "Write a short description of robot walking stance, detailing leg positions and joint angles: ",
            "Explain how the robot's legs and joints are positioned during walking (max 15 words): ",
            "Describe the specific leg posture of a walking humanoid robot: ",
            "Detail the joint angles and leg positions of a humanoid robot in walking motion: "
        ]
        
        backup_descriptions = [
            "Robot with only one leg touching the ground while other leg is suspended in mid-stride motion.",
            "Humanoid with left leg planted firmly while right leg swings forward with bent knee.",
            "Robot balancing on one leg only, with second leg clearly lifted from the ground.",
            "Humanoid with left knee raised high and right leg extended backward in walking motion.",
            "Robot with one foot firmly planted while other leg is mid-stride with bent knee joint.",
            "Bipedal robot in mid-walking pose with single leg support and other leg in swing phase.",
            "Robot maintaining balance on right leg while left leg executes forward stepping motion.",
            "Robot with precise joint angles showing single-leg stance during walking sequence.",
            "Humanoid with double-leg support phase transitioning to single-leg stance in gait cycle.",
            "Robot with legs positioned at optimal joint angles for stable walking posture."
        ]
        
        posture_keywords = ["leg", "knee", "joint", "foot", "feet", "stance", "posture", "gait", 
                           "balance", "walking", "step", "stride", "support", "motion", "angle"]
        robot_keywords = ["robot", "humanoid", "bipedal", "mechanical", "machine"]
        
        try:
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print(f"加载本地GPT-2模型: {GPT2_MODEL_PATH}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if not hasattr(self, 'gpt2') or self.gpt2 is None:
                self.gpt2 = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(self.device)
        except Exception as e:
            print(f"无法加载GPT-2模型: {e}")
            print("将使用备用描述...")
            
            print("演化概念变体...")
            for concept_id, variants in concepts.items():
                evolved_concepts[concept_id] = []
                for _ in range(n_variants):
                    description = random.choice(backup_descriptions)
                    evolved_concepts[concept_id].append(description)
                    print(f"  使用备用描述: {description[:30]}...")
            
            return evolved_concepts
        
        print("演化概念变体...")
        for concept_id, variants in concepts.items():
            best_variant = None
            if concept_id in fitness_scores and fitness_scores[concept_id] > 0:
                best_variant = variants[0] if variants else None
            
            if best_variant is None and variants:
                best_variant = random.choice(variants)
            elif best_variant is None and not variants:
                best_variant = random.choice(backup_descriptions)
            
            evolved_concepts[concept_id] = []
            
            if best_variant:
                evolved_concepts[concept_id].append(best_variant)
            
            remaining_variants = n_variants - 1 if best_variant else n_variants
            
            for _ in range(remaining_variants):
                try:
                    prompt = f"Rewrite this robot movement description with more joint details (max 15 words): '{best_variant}'"
                    
                    input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                    attention_mask = torch.ones(input_ids.shape, device=self.device)
                    
                    output = self.gpt2.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=60,
                        temperature=1.0,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=2.0,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    description = generated_text.replace(prompt, "").strip()
                    
                    description = re.sub(r'["\']+', '', description)
                    description = re.sub(r'^\s*[-:]+\s*', '', description)
                    
                    if '.' in description:
                        description = description.split('.')[0].strip() + '.'
                    elif '!' in description:
                        description = description.split('!')[0].strip() + '!'
                    elif '?' in description:
                        description = description.split('?')[0].strip() + '?'
                    
                    for template in robot_prompts:
                        description = description.replace(template, "")
                    
                    words = description.split()
                    if len(words) < 5:
                        description = random.choice(backup_descriptions)
                        print(f"  使用备用描述: {description[:30]}...")
                    else:
                        has_posture_keyword = any(keyword in description.lower() for keyword in posture_keywords)
                        has_robot_keyword = any(keyword in description.lower() for keyword in robot_keywords)
                        
                        if not (has_posture_keyword and has_robot_keyword):
                            description = random.choice(backup_descriptions)
                            print(f"  使用备用描述(缺少关键词): {description[:30]}...")
                    
                    words = description.split()
                    if len(words) > 20:
                        description = " ".join(words[:20]) + "."
                    
                    try:
                        clip_tokens = clip.tokenize([description])
                        if clip_tokens.shape[1] > 77:
                            while len(words) > 15 and clip.tokenize([" ".join(words)]).shape[1] > 77:
                                words = words[:-1]
                            description = " ".join(words) + "."
                    except Exception as e:
                        description = random.choice(backup_descriptions)
                        print(f"  CLIP处理失败，使用备用描述: {description[:30]}...")
                    
                    if self._similarity_too_high(description, [var for var in evolved_concepts[concept_id]]):
                        backup = random.choice(backup_descriptions)
                        while self._similarity_too_high(backup, [var for var in evolved_concepts[concept_id]]):
                            backup = random.choice(backup_descriptions)
                        description = backup
                        print(f"  使用备用描述(避免重复): {description[:30]}...")
                    
                    evolved_concepts[concept_id].append(description)
                    
                except Exception as e:
                    print(f"  演化变体时出错: {e}")
                    description = random.choice(backup_descriptions)
                    while self._similarity_too_high(description, [var for var in evolved_concepts[concept_id]]):
                        description = random.choice(backup_descriptions)
                    evolved_concepts[concept_id].append(description)
                    print(f"  使用备用描述(异常处理): {description[:30]}...")
        
        return evolved_concepts
    
    def save_concepts(self, concepts, file_path):
        with open(file_path, 'w') as f:
            json.dump(concepts, f, indent=2)
    
    def load_concepts(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)


# =============== 频域与语义空间融合优化 ===============

class SemanticGuidedFourierLayer(ConvResFourierLayer):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, truncation_size=16, 
                 clip_dim=512, semantic_weight=0.5, device=None):
        super().__init__(in_channels, out_channels, kernel_size, truncation_size)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.semantic_weight = semantic_weight
        
        self.semantic_attention = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, truncation_size * truncation_size)
        )
        
        self.freq_to_semantic = nn.Sequential(
            nn.Linear(truncation_size * truncation_size, 256),
            nn.ReLU(),
            nn.Linear(256, clip_dim)
        )
        
        self.clip_model, _ = get_clip_model(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def compute_semantic_attention(self, concept_features, x_fft):
        B = x_fft.size(0)
        x_fft_flat = x_fft.view(B, self.in_channels, -1)
        
        freq_semantic = self.freq_to_semantic(x_fft_flat.mean(dim=1))
        
        similarity = F.cosine_similarity(freq_semantic.unsqueeze(1), 
                                        concept_features.unsqueeze(0), dim=2)
        
        best_concept_idx = similarity.max(dim=1)[1]
        selected_concepts = concept_features[best_concept_idx]
        
        attention_weights = self.semantic_attention(selected_concepts)
        attention_weights = attention_weights.view(B, 1, self.truncation_size, self.truncation_size)
        attention_weights = torch.sigmoid(attention_weights)
        
        return attention_weights
    
    def forward(self, x, concept_features=None):
        self._update_conv(x)
        conv_out = self.conv(x)
        
        x_fft = torch.fft.fftn(x, dim=tuple(range(2, x.dim())))
        x_fft = x_fft[..., :self.truncation_size, :self.truncation_size]
        
        if concept_features is not None:
            attention_weights = self.compute_semantic_attention(concept_features, x_fft)
            
            x_fft = x_fft * (1 + self.semantic_weight * attention_weights)
        
        pad_sizes = []
        for dim_size in x.shape[2:]:
            pad_sizes.extend([0, dim_size - self.conv.weight.shape[2]])
        weight_padded = F.pad(self.conv.weight, pad_sizes[::-1])
        
        kernel_fft = torch.fft.fftn(weight_padded, dim=tuple(range(2, weight_padded.dim())))
        kernel_fft = kernel_fft[..., :self.truncation_size, :self.truncation_size]
        
        B = x_fft.size(0)
        kernel_fft = kernel_fft.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        out_fft = torch.einsum('bcxy,bkcxy->bkxy', x_fft, kernel_fft)
        
        out = torch.fft.ifftn(out_fft, s=x.shape[2:], dim=tuple(range(2, out_fft.dim())))
        out = out.real
        
        return out + conv_out + self.bias


class SemanticFourierEncoder(nn.Module):
    
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
        
        self.init_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        self.fourier_layers = nn.ModuleList([
            SemanticGuidedFourierLayer(
                embed_dim, embed_dim, 
                truncation_size=truncation_sizes[i % len(truncation_sizes)],
                clip_dim=clip_dim,
                device=device
            )
            for i in range(num_fourier_layers)
        ])
        
        self.output_map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        self.clip_model, _ = get_clip_model(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def encode_concepts(self, concepts):
        flat_concepts = []
        for variants in concepts:
            flat_concepts.extend(variants)
        
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in flat_concepts]).to(self.device)
            concept_features = self.clip_model.encode_text(text_inputs)
            concept_features = F.normalize(concept_features, dim=1)
        
        return concept_features
    
    def forward(self, x, concepts=None):
        x = x.to(self.device)
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        concept_features = None
        if concepts is not None:
            concept_features = self.encode_concepts(concepts)
        
        x = self.init_conv(x)
        
        for layer in self.fourier_layers:
            x = layer(x, concept_features)
        
        x = self.output_map(x)
        
        return x


# =============== 跨任务概念迁移 ===============

class ConceptTransferModule:
    
    def __init__(self, base_path="./concept_library", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        self.library_path = os.path.join(base_path, "concepts")
        self.metadata_path = os.path.join(base_path, "metadata.json")
        os.makedirs(self.library_path, exist_ok=True)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "tasks": {},
                "concepts": {},
                "transfer_history": []
            }
        
        self.clip_model, _ = get_clip_model(device)
    
    def save_task_concepts(self, task_name, domain_name, concepts, fitness_scores=None):
        task_path = os.path.join(self.library_path, f"{domain_name}_{task_name}")
        os.makedirs(task_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        concept_file = os.path.join(task_path, f"concepts_{timestamp}.json")
        
        concept_data = {
            "concepts": concepts,
            "fitness": fitness_scores if fitness_scores else {},
            "timestamp": timestamp
        }
        
        with open(concept_file, 'w') as f:
            json.dump(concept_data, f, indent=2)
        
        if task_name not in self.metadata["tasks"]:
            self.metadata["tasks"][task_name] = {
                "domain": domain_name,
                "concept_files": []
            }
        
        self.metadata["tasks"][task_name]["concept_files"].append(concept_file)
        
        for i, variants in enumerate(concepts):
            concept_id = f"{domain_name}_{task_name}_concept_{i}"
            self.metadata["concepts"][concept_id] = {
                "task": task_name,
                "domain": domain_name,
                "variants": variants,
                "fitness": fitness_scores.get(f"concept_{i}", 0) if fitness_scores else 0
            }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def find_similar_concepts(self, task_name, query_concepts, top_k=3):
        if not self.metadata["concepts"]:
            return query_concepts
        
        query_texts = []
        for variants in query_concepts:
            query_texts.extend(variants)
        
        with torch.no_grad():
            query_inputs = torch.cat([clip.tokenize(text) for text in query_texts]).to(self.device)
            query_features = self.clip_model.encode_text(query_inputs)
            query_features = F.normalize(query_features, dim=1)
        
        library_concepts = []
        concept_ids = []
        concept_tasks = []
        
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] == task_name:
                continue
                
            for variant in concept_data["variants"]:
                library_concepts.append(variant)
                concept_ids.append(concept_id)
                concept_tasks.append(concept_data["task"])
        
        if not library_concepts:
            return query_concepts
        
        with torch.no_grad():
            library_inputs = torch.cat([clip.tokenize(text) for text in library_concepts]).to(self.device)
            library_features = self.clip_model.encode_text(library_inputs)
            library_features = F.normalize(library_features, dim=1)
        
        similarities = torch.mm(query_features, library_features.T)
        
        similar_concepts = []
        transfer_history = []
        
        for i, variants in enumerate(query_concepts):
            variant_indices = list(range(i * len(variants), (i + 1) * len(variants)))
            variant_features = query_features[variant_indices]
            
            avg_similarities = similarities[variant_indices].mean(dim=0)
            
            top_indices = avg_similarities.topk(min(top_k, len(library_concepts))).indices.cpu().numpy()
            selected_concepts = [library_concepts[idx] for idx in top_indices]
            selected_ids = [concept_ids[idx] for idx in top_indices]
            selected_tasks = [concept_tasks[idx] for idx in top_indices]
            
            merged_variants = variants.copy()
            for concept in selected_concepts:
                if concept not in merged_variants:
                    merged_variants.append(concept)
            
            similar_concepts.append(merged_variants)
            
            transfer_history.append({
                "query_concept": f"{task_name}_concept_{i}",
                "similar_concepts": [{"id": selected_ids[j], "task": selected_tasks[j]} 
                                    for j in range(len(selected_ids))],
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
            })
        
        self.metadata["transfer_history"].extend(transfer_history)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return similar_concepts
    
    def get_progressive_concepts(self, source_task, target_task, adaptation_level=0.5):
        source_concepts = []
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] == source_task:
                source_concepts.append(concept_data["variants"])
        
        target_concepts = []
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] == target_task:
                target_concepts.append(concept_data["variants"])
        
        if not source_concepts or not target_concepts:
            return []
        
        matched_pairs = []
        
        source_texts = []
        for variants in source_concepts:
            source_texts.extend(variants)
        
        with torch.no_grad():
            source_inputs = torch.cat([clip.tokenize(text) for text in source_texts]).to(self.device)
            source_features = self.clip_model.encode_text(source_inputs)
            source_features = F.normalize(source_features, dim=1)
        
        target_texts = []
        for variants in target_concepts:
            target_texts.extend(variants)
        
        with torch.no_grad():
            target_inputs = torch.cat([clip.tokenize(text) for text in target_texts]).to(self.device)
            target_features = self.clip_model.encode_text(target_inputs)
            target_features = F.normalize(target_features, dim=1)
        
        source_variant_counts = [len(variants) for variants in source_concepts]
        target_variant_counts = [len(variants) for variants in target_concepts]
        
        source_offsets = [0] + list(np.cumsum(source_variant_counts[:-1]))
        target_offsets = [0] + list(np.cumsum(target_variant_counts[:-1]))
        
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
        
        progressive_concepts = []
        for source_idx, target_idx, similarity in matched_pairs:
            source_variants = source_concepts[source_idx]
            target_variants = target_concepts[target_idx]
            
            num_source = max(1, int(len(source_variants) * (1 - adaptation_level)))
            num_target = max(1, int(len(target_variants) * adaptation_level))
            
            mixed_variants = source_variants[:num_source] + target_variants[:num_target]
            progressive_concepts.append(mixed_variants)
        
        return progressive_concepts
    
    def generate_hybrid_concepts(self, task_name, concepts, num_variants=3):
        if not self.metadata["concepts"]:
            return concepts
        
        task_related_concepts = []
        for concept_id, concept_data in self.metadata["concepts"].items():
            if concept_data["task"] != task_name:
                task_related_concepts.extend(concept_data["variants"])
        
        if not task_related_concepts:
            return concepts
            
        print(f"为混合概念生成加载GPT-2 Small模型: {GPT2_MODEL_PATH}")
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(self.device)
        
        hybrid_concepts = []
        for variants in concepts:
            base_variant = np.random.choice(variants)
            
            num_to_mix = min(3, len(task_related_concepts))
            concepts_to_mix = np.random.choice(task_related_concepts, num_to_mix, replace=False)
            
            new_variants = [base_variant]
            
            for _ in range(num_variants - 1):
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
            
            while len(new_variants) < num_variants:
                new_variants.append(base_variant)
                
            hybrid_concepts.append(new_variants)
        
        return hybrid_concepts


def save_concepts(concepts, fitness_scores, iteration, file_path):
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["迭代", "概念ID", "变体ID", "变体描述", "适应度"])
        
        for concept_id, variants in concepts.items():
            fitness = fitness_scores.get(concept_id, 0.0) if fitness_scores else 0.0
            
            for j, variant in enumerate(variants):
                writer.writerow([iteration, concept_id, j, variant, f"{fitness:.4f}"])

def example_usage(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    clip_model, clip_preprocess = get_clip_model(device)
    print("已加载全局CLIP模型，将在各组件中共享使用")
    
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
    
    model = None
    if args.model_path and args.model_type:
        model = load_model(args.model_type, env, device, args.model_path)
        print(f"已加载 {args.model_type} 模型: {args.model_path}")
    
    save_dir = get_save_dir(args)
    
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
        
        if model:
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
        else:
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
        
        if len(episode_states) > 50:
            indices = np.linspace(0, len(episode_states) - 1, 50, dtype=int)
            episode_states = [episode_states[i] for i in indices]
            episode_actions = [episode_actions[i] for i in indices]
        
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
    
    print(f"总共收集了 {len(all_states)} 个状态-动作对")
    
    print("\n===== 1. 概念自适应与演化 =====")
    concept_generator = ConceptGenerator(model_path=GPT2_MODEL_PATH, device=device)
    
    csv_file = os.path.join(save_dir, f"{args.domain_name}_{args.task_name}_concept.csv")
    print(f"概念演化记录将保存到: {csv_file}")
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    print("生成初始概念...")
    initial_concepts = concept_generator.generate_initial_concepts(
        n_concepts=args.n_concepts,
        n_variations=args.n_variants
    )
    
    print("\n初始概念:")
    for concept_id, variants in initial_concepts.items():
        print(f"{concept_id}:")
        for j, variant in enumerate(variants):
            print(f"  变体 {j+1}: {variant}")
    
    print("\n评估概念适应度...")
    fitness_scores = concept_generator.evaluate_concept_fitness(initial_concepts, all_states, all_actions)
    
    print("适应度分数:")
    for concept_id, score in fitness_scores.items():
        print(f"{concept_id}: {score:.4f}")
    
    save_concepts(initial_concepts, fitness_scores, 0, csv_file)
    
    concepts = initial_concepts
    for i in range(args.evolution_iterations):
        print(f"\n执行第 {i+1}/{args.evolution_iterations} 次概念演化...")
        concepts = concept_generator.evolve_concepts(
            concepts=concepts,
            fitness_scores=fitness_scores,
            n_variants=args.n_variants
        )
        
        fitness_scores = concept_generator.evaluate_concept_fitness(concepts, all_states, all_actions)
        
        save_concepts(concepts, fitness_scores, i+1, csv_file)
        
        print(f"第 {i+1} 次演化后的适应度分数:")
        for concept_id, score in fitness_scores.items():
            print(f"{concept_id}: {score:.4f}")
    
    print("\n===== 最终概念及其适应度 =====")
    for concept_id, variants in concepts.items():
        fitness = fitness_scores.get(concept_id, 0.0)
        print(f"概念 {concept_id} (适应度: {fitness:.4f}):")
        for j, variant in enumerate(variants):
            print(f"  变体 {j+1}: \"{variant}\"")
        print("-" * 50)
    
    print(f"\n概念演化记录已保存到: {csv_file}")
    
    if args.save_concepts:
        concepts_file = os.path.join(save_dir, f"{args.domain_name}_{args.task_name}_concepts.json")
        concept_generator.save_concepts(concepts, concepts_file)
        print(f"概念已保存到: {concepts_file}")
    
    if args.transfer_concepts:
        print("\n===== 2. 跨任务概念迁移 =====")
        transfer_module = ConceptTransferModule(base_path=os.path.join(save_dir, "concept_library"), device=device)
        
        transfer_module.save_task_concepts(args.task_name, args.domain_name, concepts, fitness_scores)
        
        print("\n查找相似概念...")
        similar_concepts = transfer_module.find_similar_concepts(args.task_name, concepts, top_k=3)
        
        print("\n相似概念:")
        for i, variants in enumerate(similar_concepts):
            print(f"概念 {i+1}:")
            for j, variant in enumerate(variants):
                print(f"  变体 {j+1}: {variant}")
        
        print("\n生成混合概念...")
        hybrid_concepts = transfer_module.generate_hybrid_concepts(args.task_name, concepts, num_variants=args.n_variants)
        
        print("\n混合概念:")
        for i, variants in enumerate(hybrid_concepts):
            print(f"概念 {i+1}:")
            for j, variant in enumerate(variants):
                print(f"  变体 {j+1}: {variant}")
    
    if args.use_fourier_encoder:
        print("\n===== 3. 使用频域与语义空间融合 =====")
        
        encoder = SemanticFourierEncoder(
            img_size=args.image_size,
            in_channels=all_states[0].shape[0],
            embed_dim=256,
            clip_dim=512,
            num_fourier_layers=3,
            device=device
        )
        
        print("使用语义引导的傅里叶编码器处理状态...")
        
        sample_indices = np.random.choice(len(all_states), min(10, len(all_states)), replace=False)
        sample_states = [all_states[i] for i in sample_indices]
        
        print("不使用概念引导的编码:")
        encoded_states = [encoder(state).cpu().detach().numpy() for state in sample_states]
        print(f"编码形状: {encoded_states[0].shape}")
        
        print("使用概念引导的编码:")
        encoded_states_with_concepts = [encoder(state, concepts).cpu().detach().numpy() for state in sample_states]
        print(f"编码形状: {encoded_states_with_concepts[0].shape}")
        
        diffs = [np.mean(np.abs(e1 - e2)) for e1, e2 in zip(encoded_states, encoded_states_with_concepts)]
        avg_diff = np.mean(diffs)
        print(f"概念引导带来的平均编码差异: {avg_diff:.6f}")
    
    print("\n演示完成!")

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    example_usage(args)

if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=6 python gen_concepts.py \
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