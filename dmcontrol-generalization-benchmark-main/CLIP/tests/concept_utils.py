import os
import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')

import torch
import numpy as np
import clip
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import csv
import re
import random


try:
    from env.wrappers import make_env
except ImportError:
    print("警告: 无法导入环境模块，请确保路径正确")
    make_env = None


GPT2_MODEL_PATH = "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/gpt2-small"


MODEL_PATHS = {
    "gpt2-small": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/gpt2-small",
    "gpt2-medium": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/GPT2_pretrain/gpt2_medium/",
    "gpt2-large": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/gpt2_large/",
    "gpt2-xl": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/gpt2-xl/",
    "llama-2-7b": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/Llama-2-7b/",
    "llama-2-7b-hf": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/Llama-2-7b-hf/"
}

_CLIP_MODEL = None
_CLIP_PREPROCESS = None

def get_clip_model(device=None):
    global _CLIP_MODEL, _CLIP_PREPROCESS
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if _CLIP_MODEL is None:
        print("正在加载全局CLIP模型: ViT-L/14")
        try:
            _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-L/14", device=device)
            _CLIP_MODEL = _CLIP_MODEL.float()
            print("成功加载CLIP ViT-L/14模型")
        except Exception as e:
            print(f"加载ViT-L/14失败: {e}, 回退到ViT-B/32")
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        _CLIP_MODEL = _CLIP_MODEL.float()
        
    return _CLIP_MODEL, _CLIP_PREPROCESS

def save_concepts(concepts, file_path, extra_data=None):
    data = {
        "concepts": concepts,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    if extra_data:
        data.update(extra_data)
        
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_concepts(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("concepts", {})

def save_concepts_to_csv(concepts, fitness_scores, iteration, file_path):
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

def encode_concepts_with_clip(concepts, clip_model, device, cached_features=None):
    if isinstance(concepts, dict):
        flat_concepts = []
        for variants in concepts.values():
            flat_concepts.extend(variants)
    elif isinstance(concepts, list) and concepts and isinstance(concepts[0], list):
        flat_concepts = []
        for variants in concepts:
            flat_concepts.extend(variants)
    else:
        flat_concepts = concepts
    
    if cached_features is not None:
        to_encode = []
        to_encode_idx = []
        cached_results = []
        
        for i, desc in enumerate(flat_concepts):
            if desc in cached_features:
                cached_results.append(cached_features[desc])
            else:
                to_encode.append(desc)
                to_encode_idx.append(i)
        
        if to_encode:
            with torch.no_grad():
                try:
                    tokens = clip.tokenize(to_encode).to(device)
                    features = clip_model.encode_text(tokens)
                    features = F.normalize(features, dim=1)
                    
                    for i, idx in enumerate(to_encode_idx):
                        cached_features[to_encode[i]] = features[i]
                        cached_results.insert(idx, features[i])
                except RuntimeError as e:
                    print(f"CLIP编码错误: {e}")
                    print("尝试单独编码每个描述...")
                    
                    for i, desc in enumerate(to_encode):
                        try:
                            clean_desc = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', desc)
                            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                            
                            words = clean_desc.split()
                            if len(words) > 20:
                                clean_desc = ' '.join(words[:20])
                                
                            token = clip.tokenize([clean_desc]).to(device)
                            feature = clip_model.encode_text(token)
                            feature = F.normalize(feature, dim=1)[0]
                            
                            cached_features[to_encode[i]] = feature
                            cached_results.insert(to_encode_idx[i], feature)
                        except Exception as e2:
                            print(f"单独编码失败 '{desc[:20]}...': {e2}")
                            zero_feature = torch.zeros(clip_model.token_embedding.weight.shape[1], device=device)
                            cached_features[to_encode[i]] = zero_feature
                            cached_results.insert(to_encode_idx[i], zero_feature)
        
        if len(cached_results) == len(flat_concepts):
            return torch.stack(cached_results)
        else:
            raise ValueError(f"特征数量({len(cached_results)})与描述数量({len(flat_concepts)})不匹配")
    else:
        with torch.no_grad():
            try:
                text_inputs = torch.cat([clip.tokenize([desc]) for desc in flat_concepts]).to(device)
                concept_features = clip_model.encode_text(text_inputs)
                concept_features = F.normalize(concept_features, dim=1)
                return concept_features
            except RuntimeError as e:
                print(f"CLIP批量编码错误: {e}")
                print("回退到单独编码模式...")
                
                results = []
                dim = clip_model.token_embedding.weight.shape[1]
                
                for desc in flat_concepts:
                    try:
                        clean_desc = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', desc)
                        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                        
                        words = clean_desc.split()
                        if len(words) > 20:
                            clean_desc = ' '.join(words[:20])
                            
                        token = clip.tokenize([clean_desc]).to(device)
                        feature = clip_model.encode_text(token)
                        feature = F.normalize(feature, dim=1)[0]
                        results.append(feature)
                    except Exception as e2:
                        print(f"单独编码失败 '{desc[:20]}...': {e2}")
                        results.append(torch.zeros(dim, device=device))
                
                return torch.stack(results)

def clean_description(description, robot_keywords=None, posture_keywords=None, keywords_filter=True, strict_format=False, word_limit=None):
    if robot_keywords is None:
        robot_keywords = ["robot", "humanoid", "bipedal", "mechanical", "machine"]
        
    if posture_keywords is None:
        posture_keywords = ["leg", "knee", "joint", "foot", "feet", "stance", "posture", "gait",
                          "balance", "walking", "step", "stride", "support", "motion", "angle"]
    
    if word_limit is None:
        word_limit = (10, 30)
    
    if description is None:
        print("  描述为空（None值）")
        return None
    
    description = ''.join(c for c in description if ord(c) < 128)
    
    description = re.sub(r'http[s]?://\S+', '', description)
    
    description = re.sub(r'\[.*?\]|\{.*?\}|\(.*?\)', '', description)
    
    code_patterns = [
        r'```[\s\S]*?```',     
        r'`[^`]*?`',           
        r'<[^>]*?>',           
        r'#\w+',               
        r'@\w+',               
        r'\*\*.*?\*\*',        
        r'\*.*?\*',            
        r'\_\_.*?\_\_',        
        r'\_.*?\_',            
    ]
    for pattern in code_patterns:
        description = re.sub(pattern, ' ', description)
    
    description = re.sub(r'^[~\-!_\[\]\{\}\(\)#*\.]{1,}', '', description)
    description = re.sub(r'[\"\"\"\'\`\—\*\#\_\-\=\+]{2,}', '', description)
    
    description = re.sub(r'^\s*[-:_*~!]+\s*', '', description)
    
    description = re.sub(r'(第\d+[节章段]|Section \d+|Chapter \d+|PART \d+)', '', description)
    
    description = re.sub(r'START STATEMENT|BEGIN|END|NOTE:|IMPORTANT:|【.*?】|Step Description', '', description)
    
    description = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', description)
    
    description = re.sub(r'\s+', ' ', description).strip()
    
    if not description:
        print("  描述清理后为空字符串")
        return None
    
    words = description.split()
        
    if word_limit:
        min_words, max_words = word_limit
        if len(words) < min_words or len(words) > max_words:
            print(f"  描述词数不符合要求 ({len(words)}个词，要求{min_words}-{max_words}个词): '{description}'")
            return None
    
    description = description[0].upper() + description[1:]
    
    if not description[-1] in ['.', '!', '?']:
        description += '.'
        
    max_words = word_limit[1] if word_limit else 30
    if len(words) > max_words * 2:
        description = ' '.join(words[:max_words * 2])
        print(f"  描述过长，已截断至{max_words * 2}个单词: '{description}'")
    
    has_robot_keyword = any(kw.lower() in description.lower() for kw in robot_keywords)
    has_posture_keyword = any(kw.lower() in description.lower() for kw in posture_keywords)
    
    if keywords_filter and not (has_robot_keyword or has_posture_keyword):
        print(f"  描述不包含机器人或姿势关键词: '{description}'")
        return None
    
    if strict_format:
        posture_keywords_count = sum(1 for kw in posture_keywords if kw.lower() in description.lower())
        robot_keywords_count = sum(1 for kw in robot_keywords if kw.lower() in description.lower())
        
        if posture_keywords_count < 2:
            print(f"  描述姿态关键词不足 (只有{posture_keywords_count}个): '{description}'")
            return None
        
        if robot_keywords_count < 1:
            print(f"  描述机器人关键词不足 (只有{robot_keywords_count}个): '{description}'")
            return None
        
        format_check = False
        
        if " with " in description.lower() and " in " in description.lower():
            parts = description.lower().split(" with ")
            if len(parts) >= 2 and any(kw in parts[0] for kw in robot_keywords):
                second_parts = parts[1].split(" in ")
                if len(second_parts) >= 2 and any(kw in second_parts[0] for kw in posture_keywords):
                    format_check = True
        
        if not format_check:
            print(f"  描述不符合格式要求 '[机器人类型] with [腿部位置] in [动作/平衡状态]': '{description}'")
            return None
    
    try:
        tokens = clip.tokenize([description])
        token_count = len(tokens[0])
        if token_count > 75:  
            words = description.split()
            max_words = min(40, len(words) - (token_count - 70) // 2)
            description = ' '.join(words[:max_words])
            print(f"  描述token数量过多({token_count})，已截断至约70 tokens: '{description}'")
    except Exception as e:
        print(f"  描述CLIP编码测试失败: {e}，进一步清理文本")
        description = re.sub(r'[^a-zA-Z0-9\s,.!?;:\-\'"]', ' ', description)
        description = re.sub(r'\s+', ' ', description).strip()
        words = description.split()
        if len(words) > 20:
            description = ' '.join(words[:20])
    
    return description

def process_states_for_clip(states, clip_preprocess, device):
    processed_images = []
    for state in states:
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        if state.shape[0] > 3:
            state = state[-3:]
            
        processed_image = clip_preprocess(Image.fromarray(
            (state.transpose(1, 2, 0) * 255).astype(np.uint8)
        )).unsqueeze(0)
        processed_images.append(processed_image)
        
    return torch.cat(processed_images).to(device)

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

def visualize_evolution(evolution_history, fitness_history, save_path=None):
    try:
        import pandas as pd
        
        data = []
        for i, (concept_id, history) in enumerate(evolution_history.items()):
            for gen, desc in enumerate(history):
                fitness = fitness_history.get(concept_id, {}).get(gen, 0)
                data.append({
                    'concept_id': concept_id,
                    'generation': gen,
                    'description': desc[:20] + "...",  
                    'fitness': fitness
                })
                
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        for concept_id, group in df.groupby('concept_id'):
            plt.plot(group['generation'], group['fitness'], marker='o', label=concept_id)
            
        plt.title('Concept Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            print(f"演化历史图表已保存至: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"生成可视化图表失败: {e}")

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
    
    parser.add_argument('--clip_threshold', type=float, default=0.5,
                      help='CLIP语义相似度阈值，低于此值的候选将被过滤')
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                      help='描述相似度阈值，高于此值被视为过于相似')
    parser.add_argument('--temperature', type=float, default=1.2,
                      help='GPT-2生成温度，越高多样性越大')
    parser.add_argument('--num_candidates', type=int, default=5,
                      help='每次生成候选数量')
    parser.add_argument('--semantic_score_type', type=str, default='raw',
                      choices=['raw', 'shifted'],
                      help='语义得分计算方式: raw=直接使用，shifted=偏移增强')
    parser.add_argument('--keywords_filter', type=bool, default=True,
                      help='是否使用关键词过滤')
    
    parser.add_argument('--use_avg_similarity', type=bool, default=True,
                      help='使用平均相似度计算适应度')
    parser.add_argument('--use_state_consistency', type=bool, default=True,
                      help='使用状态一致性计算适应度')
    parser.add_argument('--use_action_consistency', type=bool, default=True,
                      help='使用动作一致性计算适应度')
    
    parser.add_argument('--avg_similarity_weight', type=float, default=0.4,
                      help='平均相似度权重')
    parser.add_argument('--state_consistency_weight', type=float, default=0.3,
                      help='状态一致性权重')
    parser.add_argument('--action_consistency_weight', type=float, default=0.3,
                      help='动作一致性权重')
    
    parser.add_argument('--save_concepts', action='store_true',
                      help='保存生成的概念')
    parser.add_argument('--transfer_concepts', action='store_true',
                      help='应用概念迁移')
                      
    parser.add_argument('--use_fourier_encoder', action='store_true',
                      help='使用傅里叶编码器处理状态')
    
    parser.add_argument('--gpt2_max_length', type=int, default=100,
                      help='GPT-2生成的最大token长度')
    parser.add_argument('--gpt2_top_k', type=int, default=20,
                      help='GPT-2生成时使用的top_k值，0表示禁用')
    parser.add_argument('--gpt2_top_p', type=float, default=0.9,
                      help='GPT-2生成时使用的top_p值，取值范围(0,1]')
    parser.add_argument('--gpt2_repetition_penalty', type=float, default=1.2,
                      help='GPT-2生成时的重复惩罚系数')
    
    parser.add_argument('--model_name', type=str, default='gpt2-small',
                      choices=list(MODEL_PATHS.keys()),
                      help='使用的预训练语言模型')
    
    parser.add_argument('--lm_model_type', type=str, default='auto',
                      choices=['auto', 'gpt2', 'llama'],
                      help='语言模型类型，auto表示自动检测')
    
    return parser.parse_args()

def collect_state_representations(states, num_samples=None):
    if num_samples is not None and num_samples < len(states):
        indices = np.linspace(0, len(states) - 1, num_samples, dtype=int)
        sampled_states = [states[i] for i in indices]
        return sampled_states
    return states

def encode_batch_text(descriptions, clip_model, device):
    with torch.no_grad():
        try:
            text_inputs = clip.tokenize(descriptions).to(device)
            features = clip_model.encode_text(text_inputs)
            return F.normalize(features, dim=1)
        except Exception as e:
            print(f"批量编码出错: {e}")
            results = []
            for desc in descriptions:
                try:
                    clean_desc = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', desc)
                    clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                    
                    token = clip.tokenize([clean_desc]).to(device)
                    feature = clip_model.encode_text(token)
                    feature = F.normalize(feature, dim=1)[0]
                    results.append(feature)
                except Exception as e2:
                    print(f"单独编码失败 '{desc[:20]}...': {e2}")
                    results.append(torch.zeros(clip_model.text_projection.weight.shape[1], device=device))
            
            if results:
                return torch.stack(results)
            raise ValueError("所有描述编码均失败")

def similarity_too_high(new_desc, existing_descs, clip_model=None, device=None, threshold=None):
    if not existing_descs:
        return False
        
    if threshold is None:
        threshold = 0.85  
    
    for desc in existing_descs:
        if desc == new_desc:
            return True
            
        longer = max(len(desc), len(new_desc))
        if longer > 0:
            common_chars = sum(c1 == c2 for c1, c2 in zip(desc.lower(), new_desc.lower()))
            if common_chars / longer > threshold:
                return True
    
    if clip_model is not None and device is not None:
        try:
            new_embedding = encode_batch_text([new_desc], clip_model, device)
            existing_embeddings = encode_batch_text(existing_descs, clip_model, device)
            
            similarities = F.cosine_similarity(new_embedding, existing_embeddings)
            max_similarity = similarities.max().item()
            
            if max_similarity > threshold:
                return True
        except Exception as e:
            print(f"计算语义相似度时出错: {e}")
    
    return False

def _encode_batch_text(descriptions, clip_model, device, cached_features=None):
    return encode_concepts_with_clip(descriptions, clip_model, device, cached_features)

def collect_state_representations_with_clip(states, clip_model, clip_preprocess, device):
    processed_images = process_states_for_clip(states, clip_preprocess, device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(processed_images)
        normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
    return normalized_features

def _similarity_too_high(new_desc, existing_descs, clip_model=None, device=None, threshold=None, cached_features=None):
    if not existing_descs:
        return False

    if threshold is None:
        threshold = 0.85

    all_descs = [new_desc] + existing_descs
    all_features = _encode_batch_text(all_descs, clip_model, device, cached_features)

    new_features = all_features[0].unsqueeze(0)
    existing_features = all_features[1:]

    similarities = (new_features @ existing_features.T).squeeze(0)
    max_similarity = similarities.max().item() if similarities.numel() > 0 else 0

    return max_similarity > threshold
