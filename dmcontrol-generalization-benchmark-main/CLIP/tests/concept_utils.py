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
import tempfile
import subprocess
import socket
import base64
import io


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
    "llama-2-7b-hf": "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/Llama-2-7b-hf/",
    "llava-1.5-7b": "/mnt/lustre/GPU4/home/wuhanpeng/models/llava-v1.5-7b" 
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
    # 展平嵌套列表
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
    
    # 处理缓存
    if cached_features is not None:
        # 检查缓存中是否已有特征
        to_encode = []
        to_encode_idx = []
        cached_results = []
        
        for i, desc in enumerate(flat_concepts):
            if desc in cached_features:
                cached_results.append(cached_features[desc])
            else:
                to_encode.append(desc)
                to_encode_idx.append(i)
        
        # 如果有需要编码的文本
        if to_encode:
            with torch.no_grad():
                try:
                    tokens = clip.tokenize(to_encode).to(device)
                    features = clip_model.encode_text(tokens)
                    features = F.normalize(features, dim=1)
                    
                    # 缓存新编码的特征
                    for i, idx in enumerate(to_encode_idx):
                        cached_features[to_encode[i]] = features[i]
                        cached_results.insert(idx, features[i])
                except RuntimeError as e:
                    print(f"CLIP编码错误: {e}")
                    print("尝试单独编码每个描述...")
                    
                    for i, desc in enumerate(to_encode):
                        try:
                            # 清理特殊字符
                            clean_desc = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', desc)
                            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                            
                            # 截断长度
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
                            # 使用零向量作为回退
                            zero_feature = torch.zeros(clip_model.token_embedding.weight.shape[1], device=device)
                            cached_features[to_encode[i]] = zero_feature
                            cached_results.insert(to_encode_idx[i], zero_feature)
        
        if len(cached_results) == len(flat_concepts):
            return torch.stack(cached_results)
        else:
            raise ValueError(f"特征数量({len(cached_results)})与描述数量({len(flat_concepts)})不匹配")
    else:
        # 不使用缓存
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
                        # 清理特殊字符
                        clean_desc = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', desc)
                        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                        
                        # 截断长度
                        words = clean_desc.split()
                        if len(words) > 20:
                            clean_desc = ' '.join(words[:20])
                            
                        token = clip.tokenize([clean_desc]).to(device)
                        feature = clip_model.encode_text(token)
                        feature = F.normalize(feature, dim=1)[0]
                        results.append(feature)
                    except Exception as e2:
                        print(f"单独编码失败 '{desc[:20]}...': {e2}")
                        # 使用零向量
                        results.append(torch.zeros(dim, device=device))
                
                return torch.stack(results)

def clean_description(description, robot_keywords=None, posture_keywords=None, keywords_filter=True, strict_format=False, word_limit=None):
    # 如果已经是错误信息，直接返回
    if description and description.startswith("[ERROR]"):
        print(f"  检测到错误描述，跳过处理: '{description}'")
        return description
        
    # 检查是否为无意义文本
    if description and (description.isdigit() or re.match(r'^\d+[^\w]*$', description)):
        print(f"  检测到无意义数字文本: '{description}'")
        return f"[ERROR] 无意义数字文本: {description}"
    
    if description and (len(description.strip()) < 3 or 
                       all(c in '.,;:!? ' for c in description) or
                       description.count(' ') < 2):
        print(f"  描述内容无效: '{description}'")
        return f"[ERROR] 描述内容无效: {description}"
    
    # 默认关键词列表
    if robot_keywords is None:
        robot_keywords = ["robot", "humanoid", "bipedal", "mechanical", "machine"]
        
    if posture_keywords is None:
        posture_keywords = ["leg", "knee", "joint", "foot", "feet", "stance", "posture", "gait",
                          "balance", "walking", "step", "stride", "support", "motion", "angle"]
    
    if word_limit is None:
        word_limit = (5, 30)
    
    if description is None:
        print("  描述为空（None值）")
        return f"[ERROR] 描述为空（None值）"
    
    # 过滤所有非ASCII字符
    description = ''.join(c for c in description if ord(c) < 128)
    
    # 去除URL链接
    description = re.sub(r'http[s]?://\S+', '', description)
    
    # 去除各种括号及其内容
    description = re.sub(r'\[.*?\]|\{.*?\}|\(.*?\)', '', description)
    
    # 去除代码块、HTML标签和特殊标记
    code_patterns = [
        r'```[\s\S]*?```',     # 代码块
        r'`[^`]*?`',           # 行内代码
        r'<[^>]*?>',           # HTML标签
        r'#\w+',               # 井号标签
        r'@\w+',               # @提及
        r'\*\*.*?\*\*',        # 加粗
        r'\*.*?\*',            # 斜体
        r'\_\_.*?\_\_',        # 加粗
        r'\_.*?\_',            # 斜体
    ]
    for pattern in code_patterns:
        description = re.sub(pattern, ' ', description)
    
    # 去除特殊前缀和格式字符
    description = re.sub(r'^[~\-!_\[\]\{\}\(\)#*\.]{1,}', '', description)
    description = re.sub(r'[\"\"\"\'\`\—\*\#\_\-\=\+]{2,}', '', description)
    
    # 去除开头的格式标记和分隔符
    description = re.sub(r'^\s*[-:_*~!]+\s*', '', description)
    
    # 去掉段落标记和编号
    description = re.sub(r'(第\d+[节章段]|Section \d+|Chapter \d+|PART \d+)', '', description)
    
    # 去掉指令性语句标记
    description = re.sub(r'START STATEMENT|BEGIN|END|NOTE:|IMPORTANT:|【.*?】|Step Description', '', description)
    
    # 只保留基本英文字母、数字和常见标点
    description = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', description)
    
    # 压缩空白字符
    description = re.sub(r'\s+', ' ', description).strip()
    
    # 检查清理后的描述是否为空
    if not description:
        print("  描述清理后为空字符串")
        return f"[ERROR] 描述清理后为空字符串"
    
    # 获取单词列表
    words = description.split()
        
    # 检查描述长度是否在限制范围内
    if word_limit:
        min_words, max_words = word_limit
        if len(words) < min_words:
            print(f"  描述词数不足 ({len(words)}个词，需要{min_words}个词): '{description}'")
            # 如果词数太少，自动扩充描述
            if len(words) >= 2:  
                # 添加机器人类型
                if not any(kw in description.lower() for kw in robot_keywords):
                    robot_type = random.choice(["Humanoid robot", "Bipedal robot", "Mechanical robot"])
                    # 检查描述是否以机器人类型开头
                    if not description.lower().startswith(robot_type.lower()):
                        # 如果不是，将其添加到开头
                        description = f"{robot_type} with {description}"
                        words = description.split()

                # 添加姿势/动作描述
                if len(words) < min_words and not any(kw in description.lower() for kw in posture_keywords):
                    posture_desc = random.choice(["balanced stance", "coordinated movement", "stable walking motion", "controlled gait"])
                    if "in" not in description.lower():
                        description = f"{description} in {posture_desc}"
                    else:
                        description = f"{description} with {posture_desc}"
                    words = description.split()

                # 如果还是不够词数，添加更多描述
                if len(words) < min_words:
                    additional = random.choice([
                        "maintaining perfect balance",
                        "showing precise joint control",
                        "demonstrating fluid motion",
                        "with careful weight distribution",
                        "exhibiting stable gait pattern"
                    ])
                    description = f"{description}, {additional}"
            else:
                # 词数太少无法扩充，使用空字符串
                description = f"[ERROR] 描述词数不足: {len(words)}"
            
            # 重新获取单词列表
            words = description.split()
            
        elif len(words) > max_words:
            print(f"  描述词数过多 ({len(words)}个词，要求最多{max_words}个词): '{description}'")
            
            # 保留包含关键词的部分
            keywords_combined = robot_keywords + posture_keywords
            keyword_positions = [i for i, word in enumerate(words) 
                                if any(kw.lower() in word.lower() for kw in keywords_combined)]
            
            if keyword_positions and max(keyword_positions) < max_words * 1.5:
                # 保留到最后一个关键词后的几个词
                end_pos = min(max(keyword_positions) + 3, len(words))
                end_pos = min(end_pos, max_words)  # 不超过最大词数限制
                description = ' '.join(words[:end_pos])
            else:
                # 简单截断
                description = ' '.join(words[:max_words])
            
            print(f"  截断后: '{description}'")
    
    # 强制首字母大写
    description = description[0].upper() + description[1:]
    
    # 句子以标点符号结束
    if not description[-1] in ['.', '!', '?']:
        description += '.'
        
    # 关键词检查（宽松）
    if keywords_filter:
        has_robot_keyword = any(kw.lower() in description.lower() for kw in robot_keywords)
        has_posture_keyword = any(kw.lower() in description.lower() for kw in posture_keywords)
        
        # 如果缺少关键词，添加一些
        if not has_robot_keyword:
            prefix = random.choice(["Robot", "Humanoid", "Bipedal robot"])
            if not description.lower().startswith(prefix.lower()):
                description = f"{prefix} {description}"
            has_robot_keyword = True
            
        if not has_posture_keyword:
            posture_term = random.choice(["walking", "with balanced stance", "with coordinated leg movement", "maintaining stable gait"])
            if "in" in description:
                parts = description.split(" in ")
                description = f"{parts[0]} in {posture_term}"
                if len(parts) > 1:
                    description += f" and {parts[1]}"
            else:
                description = f"{description}, {posture_term}"
            has_posture_keyword = True
    
    # 格式检查和修复（宽松版）
    if strict_format:
        # 检查是否包含 "with" 和 "in"
        if " with " not in description.lower():
            parts = description.split()
            if len(parts) > 4:  
                insert_pos = min(2, len(parts)-3)  
                parts.insert(insert_pos, "with")
                description = " ".join(parts)
                
        if " in " not in description.lower():
            parts = description.split()
            if len(parts) > 3:  
                with_pos = -1
                for i, word in enumerate(parts):
                    if word.lower() == "with":
                        with_pos = i
                        break
                
                if with_pos >= 0 and with_pos < len(parts) - 3:
                    insert_pos = with_pos + min(3, len(parts) - with_pos - 1)
                    parts.insert(insert_pos, "in")
                    description = " ".join(parts)
                else:
                    insert_pos = max(len(parts) - 3, 2)
                    parts.insert(insert_pos, "in")
                    description = " ".join(parts)
    
    # 尝试通过CLIP tokenizer验证，但增加允许的token数量
    try:
        tokens = clip.tokenize([description])
        token_count = len(tokens[0])
 
        if token_count > 120:  
            words = description.split()
            # 保留更多的前半部分和后半部分
            if len(words) > 40:
                # 保留前30个和后10个词，中间用"..."连接
                description = ' '.join(words[:30]) + " ... " + ' '.join(words[-10:])
            else:
                # 如果词不多，保留前30个词
                description = ' '.join(words[:40])
            print(f"  描述token数量过多({token_count})，已智能截断: '{description}'")
            # 重新检查token数量
            tokens = clip.tokenize([description])
            token_count = len(tokens[0])
            print(f"  截断后token数量: {token_count}")
    except Exception as e:
        print(f"  描述CLIP编码测试失败: {e}，进一步清理文本")
        description = re.sub(r'[^a-zA-Z0-9\s,.!?;:\-\'"]', ' ', description)
        description = re.sub(r'\s+', ' ', description).strip()
        words = description.split()
        if len(words) > 40:  
            description = ' '.join(words[:40])
    
    # 17) 如果描述长度在4个词以下，返回错误信息
    words = description.split()
    if len(words) <= 4:
        print(f"  描述过短 ({len(words)}个词): '{description}'")
        return f"[ERROR] 描述过短（{len(words)}个词）"
    
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

def load_model(algorithm_name, env, device, model_path):
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
    """统一的结果保存目录获取函数"""
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
        
        # 创建数据框
        data = []
        for i, (concept_id, history) in enumerate(evolution_history.items()):
            for gen, desc in enumerate(history):
                fitness = fitness_history.get(concept_id, {}).get(gen, 0)
                data.append({
                    'concept_id': concept_id,
                    'generation': gen,
                    'description': desc[:20] + "...",  # 截断显示
                    'fitness': fitness
                })
                
        df = pd.DataFrame(data)
        
        # 绘制适应度变化曲线
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
    
    # 添加自定义布尔转换函数
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    parser.add_argument('--states_per_episode', type=int, default=20,
                        help='每个episode采样的状态数量，设置为0表示不采样保留所有状态')

    parser.add_argument('--rl_model_path', type=str, default=None,
                        help='训练好的强化学习模型路径')
    parser.add_argument('--algorithm_name', type=str, default='svea',
                        choices=['svea', 'drq', 'hifno', 'hifno_bisim'],
                        help='强化学习算法类型')

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
                      help='语言模型生成温度，越高多样性越大')
    parser.add_argument('--num_candidates', type=int, default=5,
                      help='每次生成候选数量')
    parser.add_argument('--semantic_score_type', type=str, default='raw',
                      choices=['raw', 'shifted'],
                      help='语义得分计算方式: raw=直接使用，shifted=偏移增强')
    parser.add_argument('--keywords_filter', type=str2bool, default=True,
                      help='是否使用关键词过滤（True/False）')
    
    parser.add_argument('--use_avg_similarity', type=str2bool, default=True,
                      help='使用平均相似度计算适应度（True/False）')
    parser.add_argument('--use_state_consistency', type=str2bool, default=True,
                      help='使用状态一致性计算适应度（True/False）')
    parser.add_argument('--use_action_consistency', type=str2bool, default=True,
                      help='使用动作一致性计算适应度（True/False）')
    
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
    
    parser.add_argument('--language_model_path', type=str, default=None,
                       help='预训练语言模型路径，如果为None则根据model_name自动选择')
    parser.add_argument('--model_name', type=str, default='gpt2-medium',
                       help='使用的语言模型类型（gpt2-medium, llama-2-7b, llava-1.5-7b等）')
    parser.add_argument('--gpt2_max_length', type=int, default=100,
                      help='语言模型生成的最大token长度')
    parser.add_argument('--gpt2_top_k', type=int, default=20,
                      help='语言模型生成时使用的top_k值，0表示禁用')
    parser.add_argument('--gpt2_top_p', type=float, default=0.9,
                      help='语言模型生成时使用的top_p值，取值范围(0,1]')
    parser.add_argument('--gpt2_repetition_penalty', type=float, default=1.2,
                      help='语言模型生成时的重复惩罚系数')
    
    parser.add_argument('--use_keyword_guided_generation', type=str2bool, default=True,
                      help='是否使用关键词引导生成（True/False）')
    parser.add_argument('--keyword_groups_focus', type=str, nargs='+', default=None,
                      help='要关注的关键词组列表，如"机器人类型 关节结构 运动特征"')
    parser.add_argument('--min_keywords', type=int, default=3,
                      help='最少使用的关键词数量')
    parser.add_argument('--max_keywords', type=int, default=7,
                      help='最多使用的关键词数量')
    
    parser.add_argument('--llava_env_path', type=str, default="/mnt/lustre/GPU4/home/wuhanpeng/anaconda3/envs/llava/bin/python",
                       help='LLaVA环境的Python解释器路径，用于子进程调用')
    parser.add_argument('--llava_prompt', type=str, 
                       default="Describe what you see in this image, focusing on the robot's posture and leg positions.",
                       help='发送给LLaVA的图像描述提示')
    parser.add_argument('--use_clean_description', type=str2bool, default=True,
                      help='是否对生成的描述使用clean_description清洗（True/False）')
    return parser.parse_args()

def collect_state_representations(states, num_samples=None):
    if num_samples is not None and num_samples < len(states):
        # 均匀采样以确保覆盖整个轨迹
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
            # 尝试单独编码
            results = []
            for desc in descriptions:
                try:
                    # 清理特殊字符
                    clean_desc = re.sub(r'[^\w\s,.!?;:\-\'"]', ' ', desc)
                    clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                    
                    token = clip.tokenize([clean_desc]).to(device)
                    feature = clip_model.encode_text(token)
                    feature = F.normalize(feature, dim=1)[0]
                    results.append(feature)
                except Exception as e2:
                    print(f"单独编码失败 '{desc[:20]}...': {e2}")
                    # 使用零向量作为回退
                    results.append(torch.zeros(clip_model.text_projection.weight.shape[1], device=device))
            
            if results:
                return torch.stack(results)
            raise ValueError("所有描述编码均失败")

def similarity_too_high(new_desc, existing_descs, clip_model=None, device=None, threshold=None):
    if not existing_descs:
        return False
        
    if threshold is None:
        threshold = 0.85  # 默认相似度阈值
    
    # 文本相似度比较
    # 检查基于字符重叠的相似度
    for desc in existing_descs:
        if desc == new_desc:
            return True
            
        # 计算最长公共子序列比例
        longer = max(len(desc), len(new_desc))
        if longer > 0:
            common_chars = sum(c1 == c2 for c1, c2 in zip(desc.lower(), new_desc.lower()))
            if common_chars / longer > threshold:
                return True
    
    # 如果提供了CLIP模型，进行语义相似度比较
    if clip_model is not None and device is not None:
        try:
            # 编码新描述和已有描述
            new_embedding = encode_batch_text([new_desc], clip_model, device)
            existing_embeddings = encode_batch_text(existing_descs, clip_model, device)
            
            # 计算余弦相似度
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
    # 禁用相似度过滤功能，始终返回False
    return False

    # 以下是原始代码，已被注释掉
    """
    if not existing_descs:
        return False

    # 使用默认相似度阈值
    if threshold is None:
        threshold = 0.85

    # 批量编码所有描述
    all_descs = [new_desc] + existing_descs
    all_features = _encode_batch_text(all_descs, clip_model, device, cached_features)

    # 新描述的特征是第一个
    new_features = all_features[0].unsqueeze(0)
    # 已有描述的特征是剩余的
    existing_features = all_features[1:]

    # 计算相似度并检查是否超过阈值
    similarities = (new_features @ existing_features.T).squeeze(0)
    max_similarity = similarities.max().item() if similarities.numel() > 0 else 0

    return max_similarity > threshold
    """

def init_keyword_groups():
    keyword_groups = {
        "机器人类型": ["robot", "humanoid", "bipedal", "mechanical", "machine", "android", "automaton"],
        "关节结构": ["leg", "knee", "joint", "foot", "feet", "ankle", "thigh", "calf", "shin", "limb"],
        "姿态状态": ["stance", "posture", "gait", "balance", "equilibrium", "position", "alignment"],
        "运动特征": ["walking", "step", "stride", "motion", "movement", "locomotion", "gait cycle"],
        "平衡状态": ["balance", "stability", "equilibrium", "centered", "poised", "steady"],
        "动作描述": ["bending", "extending", "flexing", "swinging", "lifting", "supporting", "pushing"]
    }
    return keyword_groups

def select_keywords_from_groups(group_dict, selected_groups=None, keywords_per_group=1, min_total=5, max_total=10):
    if selected_groups is None:
        # 随机选择2-4个组
        num_groups = min(random.randint(2, 4), len(group_dict))
        selected_groups = random.sample(list(group_dict.keys()), num_groups)
    
    selected_keywords = []
    # "机器人类型"和"姿态状态"组始终有
    essential_groups = ["机器人类型", "姿态状态"]
    for group in essential_groups:
        if group in group_dict and group not in selected_groups:
            selected_groups.append(group)
    
    
    for group in selected_groups:
        if group in group_dict:
            keywords = group_dict[group]
            num_to_select = min(keywords_per_group, len(keywords))
            selected_keywords.extend(random.sample(keywords, num_to_select))
    
    # 确保关键词总数在指定范围内
    if len(selected_keywords) < min_total:
        # 从所有关键词中随机添加，直到达到最小数量
        all_keywords = []
        for keywords in group_dict.values():
            all_keywords.extend(keywords)
        # 移除已选择的关键词
        remaining_keywords = [k for k in all_keywords if k not in selected_keywords]
        # 随机选择更多关键词
        if remaining_keywords:
            additional = random.sample(remaining_keywords, min(min_total - len(selected_keywords), len(remaining_keywords)))
            selected_keywords.extend(additional)
    
    # 如果超出最大数量，随机移除一些
    if len(selected_keywords) > max_total:
        selected_keywords = random.sample(selected_keywords, max_total)
    
    return selected_keywords

# BLIP图像描述模型相关函数
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _HAS_BLIP = True
except ImportError:
    _HAS_BLIP = False
    print("未能导入BLIP模型")

_BLIP_MODEL = None
_BLIP_PROCESSOR = None

def get_blip_model(device=None):
    global _BLIP_MODEL, _BLIP_PROCESSOR
    
    if not _HAS_BLIP:
        print("BLIP模型不可用，请安装transformers库")
        return None, None
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if _BLIP_MODEL is None:
        print("正在加载全局BLIP图像描述模型...")
        try:
            _BLIP_PROCESSOR = BlipProcessor.from_pretrained("/mnt/lustre/GPU4/home/wuhanpeng/models/blip/")
            _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("/mnt/lustre/GPU4/home/wuhanpeng/models/blip/").to(device)
            print("成功加载BLIP图像描述模型")
        except Exception as e:
            print(f"加载BLIP模型失败: {e}")
            _BLIP_PROCESSOR = None
            _BLIP_MODEL = None
            print("BLIP模型加载失败，将无法使用图像描述功能")
            
    return _BLIP_PROCESSOR, _BLIP_MODEL

def generate_image_captions(states, device=None, max_captions=3):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    processor, model = get_blip_model(device)
    
    if model is None or processor is None:
        print("BLIP模型未初始化，无法生成图像描述")
        return []
        
    captions = []
    if len(states) > max_captions:
        indices = np.linspace(0, len(states) - 1, max_captions, dtype=int)
        sample_states = [states[i] for i in indices]
    else:
        sample_states = states
    
    print(f"为{len(sample_states)}个环境状态生成描述...")
    
    for state in sample_states:
        # 处理状态数据为PIL图像
        if hasattr(state, '_frames') or type(state).__name__ == 'LazyFrames':
            arr = np.array(state)
            if arr.shape[0] > 3:
                arr = arr[-3:]
            img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
        elif isinstance(state, np.ndarray):
            arr = state
            if arr.shape[0] > 3:
                arr = arr[-3:]
            img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
        elif isinstance(state, Image.Image):
            img = state.convert('RGB')
        else:
            print(f"[BLIP] 不支持的状态类型: {type(state)}")
            captions.append(f"[ERROR] 不支持的状态类型: {type(state)}")
            continue
        
        # 使用BLIP生成描述
        try:
            image_resized = img.resize((224, 224), Image.BICUBIC)
            
            
            inputs = processor(image_resized, return_tensors="pt", do_resize=False).to(device)
            
            with torch.no_grad():
                
                out = model.generate(**inputs, 
                                    max_length=30, 
                                    num_return_sequences=1, 
                                    do_sample=True, 
                                    temperature=0.7)
                caption = processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
                print(f"  生成描述: {caption}")
        except Exception as e:
            print(f"  生成描述时出错: {e}")
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_resized = img.resize((224, 224), Image.BICUBIC)
                
                image_array = np.array(image_resized).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                
                with torch.no_grad():
                    outputs = model(pixel_values=image_tensor)
                    out = model.generate(pixel_values=image_tensor, 
                                        max_length=30, 
                                        num_return_sequences=1, 
                                        do_sample=True, 
                                        temperature=0.7)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    captions.append(caption)
                    print(f"  生成描述 (备选方法): {caption}")
            except Exception as e2:
                print(f"  备选方法也失败: {e2}")
                captions.append(f"[ERROR] 图像处理失败: {e2}")
                print(f"  返回错误信息: 图像处理失败: {e2}")
    
    return captions

_HAS_LLAVA = False


_LLAVA_MODEL = None
_LLAVA_PROCESSOR = None
_LLAVA_MODEL_PATH = "/mnt/lustre/GPU4/home/wuhanpeng/models/llava-v1.5-7b"  # 硬编码LLaVA模型路径

def process_image_with_llava_subprocess(image, prompt=None, model_path=None, llava_env_path=None):
    try:
        if prompt is None:
            prompt = "Describe what you see in this image. Focus on the robot's posture, leg positions, and body orientation. Be specific and detailed about the walking motion."
        
        if len(prompt) > 1000:
            # 如果提示过长，截取合理长度
            prompt = prompt[:1000] + "... [提示已截断]"
        
        # 处理不同类型的图像输入
        # 检查是否为LazyFrames对象 (OpenAI Gym的延迟加载帧)
        if hasattr(image, '_frames') or str(type(image).__name__) == 'LazyFrames':
            # 将LazyFrames转换为numpy数组
            image_array = np.array(image)
            if image_array.shape[0] > 3:
                image_array = image_array[-3:]  # 只保留RGB通道
            # 转换为PIL图像
            image = Image.fromarray((image_array.transpose(1, 2, 0) * 255).astype(np.uint8))
        # 处理numpy数组
        elif isinstance(image, np.ndarray):
            if image.shape[0] > 3:
                image = image[-3:]
            image = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
        # 确保图像是PIL.Image类型
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的图像类型: {type(image)}，需要PIL.Image, numpy.ndarray或LazyFrames")
        
        # 创建临时文件保存图像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            img_path = tmp_img_file.name
            image.save(img_path)
            
        # 创建临时文件保存输出
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_out_file:
            out_path = tmp_out_file.name
        
        script_path = "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP/tests/llava_bridge.py"
        
        if not os.path.exists(script_path):
            print(f"错误: LLaVA脚本文件不存在: {script_path}")
            return f"[ERROR] LLaVA脚本文件不存在: {script_path}"
            
        if not llava_env_path or not os.path.exists(llava_env_path):
            print(f"错误: LLaVA环境路径无效: {llava_env_path}")
            return f"[ERROR] LLaVA环境路径无效: {llava_env_path}"
            
        # 构建子进程
        cmd = [
            llava_env_path,
            script_path,
            '--image_path', img_path,
            '--prompt', prompt,
            '--output_file', out_path
        ]
        
        if model_path:
            cmd.extend(['--model_path', model_path])
        
        try:
            # 执行子进程
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(script_path)
            )
            
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.strip() if stderr else "未知错误"
                print(f"[LLaVA错误] 子进程返回错误代码 {proc.returncode}: {error_msg}")
                return f"[ERROR] LLaVA处理失败: {error_msg}"

            try:
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    with open(out_path, 'r') as f:
                        result_json = json.load(f)
                    description = result_json.get('description', '')
                    
                    if not description and stdout:
                        try:
                            # 只处理最后一行作为JSON输出
                            json_lines = [line for line in stdout.strip().split('\n') if line.strip().startswith('{')]
                            if json_lines:
                                stdout_json = json.loads(json_lines[-1])  # 取最后一行JSON
                                description = stdout_json.get('description', '')
                                print(f"  从JSON输出中提取描述: '{description}'")
                            else:
                                print(f"  未在stdout中找到JSON格式输出")
                                
                                if "error" in stdout.lower() or "exception" in stdout.lower():
                                    description = f"[ERROR] {stdout.strip()}"
                                else:
                                    description = stdout.strip()
                        except json.JSONDecodeError:
                            print(f"  无法解析stdout为JSON: '{stdout}'")
                            if "error" in stdout.lower() or "exception" in stdout.lower():
                                description = f"[ERROR] {stdout.strip()}"
                            else:
                                
                                stdout_content = stdout.strip()
                                # 过滤掉可能的前缀
                                if "LLaVA输出:" in stdout_content:
                                    
                                    description = stdout_content.split("LLaVA输出:", 1)[1].strip()
                                else:
                                    description = stdout_content
                                print(f"  直接使用stdout作为描述: '{description}'")
                else:
                    # 如果输出文件不存在或为空，尝试从stdout中提取描述
                    print(f"  输出文件不存在或为空，尝试从stdout提取: '{stdout[:100]}...'")
                    
                    # 尝试查找JSON输出
                    json_lines = [line for line in stdout.strip().split('\n') if line.strip().startswith('{')]
                    if json_lines:
                        try:
                            stdout_json = json.loads(json_lines[-1])  # 取最后一行JSON
                            description = stdout_json.get('description', '')
                            print(f"  从stdout的JSON中提取描述: '{description}'")
                        except json.JSONDecodeError:
                            description = stdout.strip()
                    else:
                        # 如果没有JSON格式的行，直接使用stdout
                        stdout_content = stdout.strip()
                        # 过滤掉可能的前缀
                        if "LLaVA输出:" in stdout_content:
                            description = stdout_content.split("LLaVA输出:", 1)[1].strip()
                        else:
                            description = stdout_content
                        print(f"  直接使用stdout内容: '{description}'")
                    
                # 如果没有得到有效描述，返回错误信息
                if not description:
                    raw_output = f"stdout: {stdout[:200]}..." if len(stdout) > 200 else stdout
                    print(f"没有从LLaVA获取到有效描述，原始输出: {raw_output}")
                    description = f"[ERROR] LLaVA没有生成有效描述"
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"处理LLaVA输出时出错: {e}")
                print(f"错误堆栈: {error_trace}")
                description = f"[ERROR] 处理LLaVA输出时异常: {str(e)}"
            
            return description
        finally:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception as e:
                print(f"清理临时文件时出错: {e}")
    
    except Exception as e:
        print(f"LLaVA处理图像出错: {e}")
        return f"[ERROR] LLaVA处理图像出错: {e}"

def process_states_with_llava(states, model_path=None, llava_env_path=None, max_samples=3, prompt=None, host='localhost', port=7788):
    """
    使用 socket 客户端调用 LLaVA socket 服务端，批量处理 states，返回描述列表。
    states: list of numpy.ndarray or PIL.Image
    prompt: str
    host, port: socket 服务端地址
    """
    if not states:
        print("警告: 没有提供状态数据")
        return ["[ERROR] 没有提供状态数据"]
    # if not prompt:
    # prompt = "Question: Describe the robot's posture in this image.\nAnswer:"  
    # prompt = "Describe the state of the robot's limbs and joints as it moves in this figure"  
    prompt = "Describe in a complete sentence the state of the robot's limbs and joints as it moves in this figure."

    captions = []
    for i, state in enumerate(states):
        print(f"[LLaVA-Socket] 处理图像 {i+1}/{len(states)}...")
        # 转为 PIL.Image
        if hasattr(state, '_frames') or type(state).__name__ == 'LazyFrames':
            arr = np.array(state)
            if arr.shape[0] > 3:
                arr = arr[-3:]
            img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
        elif isinstance(state, np.ndarray):
            arr = state
            if arr.shape[0] > 3:
                arr = arr[-3:]
            img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
        elif isinstance(state, Image.Image):
            img = state.convert('RGB')
        else:
            print(f"[LLaVA-Socket] 不支持的状态类型: {type(state)}")
            captions.append(f"[ERROR] 不支持的状态类型: {type(state)}")
            continue
        # 编码为 base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        payload = {
            "prompt": prompt,
            "image_base64": img_base64
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(30)  # 设置30秒超时
                s.connect((host, port))
                s.sendall(json.dumps(payload).encode())
                s.shutdown(socket.SHUT_WR)
                
                # 增加接收缓冲区大小，确保完整接收数据
                chunks = []
                bytes_received = 0
                while True:
                    try:
                        chunk = s.recv(8192)  # 增加单次接收缓冲区大小
                        if not chunk:
                            break
                        bytes_received += len(chunk)
                        chunks.append(chunk)
                        if bytes_received > 1024*1024:  # 设置1MB的接收上限，防止无限接收
                            break
                    except socket.timeout:
                        print("[LLaVA-Socket] 接收超时，可能已接收到全部数据")
                        break
                
                response = b''.join(chunks)
                
                resp_str = response.decode()
                try:
                    resp_json = json.loads(resp_str)
                    desc = resp_json.get('description', '')
                    # 确保描述不为空
                    if not desc.strip():
                        desc = "[ERROR] 服务器返回空描述"
                except Exception as e:
                    print(f"[LLaVA-Socket] 响应解析失败: {e}, 原始: {resp_str[:200]}...")
                    desc = f"[ERROR] 响应解析失败: {e}"
                captions.append(desc)
                # 打印完整的描述，不截断
                print(f"[LLaVA-Socket] 获取描述 ({len(desc)}字符): {desc}")
        except Exception as e:
            print(f"[LLaVA-Socket] socket 通信异常: {e}")
            captions.append(f"[ERROR] socket 通信异常: {e}")
    print(f"[LLaVA-Socket] 批量处理完成，共生成{len(captions)}个描述")
    return captions
