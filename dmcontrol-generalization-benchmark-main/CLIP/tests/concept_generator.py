import os
import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')
from algorithms.models.HiFNO_multigpu import HierarchicalFNO, ConvResFourierLayer
from bisimulation_loss_1 import BisimulationLoss
import random
import re
import csv
import json
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import transformers
from einops import rearrange
import torch.nn.functional as F
from torch import nn
import clip
import numpy as np
import torch
import argparse


try:
    from spellchecker import SpellChecker
    HAS_SPELLCHECKER = True
except ImportError:
    HAS_SPELLCHECKER = False
    print("警告: 未找到拼写检查器")

from concept_utils import (get_clip_model, load_model, get_save_dir, parse_args, 
                  GPT2_MODEL_PATH, MODEL_PATHS, make_env, save_concepts, load_concepts,
                  encode_concepts_with_clip, process_states_for_clip, save_concepts_to_csv, visualize_evolution, clean_description,
                  collect_state_representations_with_clip, _encode_batch_text, _similarity_too_high,
                  init_keyword_groups, select_keywords_from_groups, process_states_with_llava)


# =============== 概念自适应演化 ===============


class ConceptGenerator:
    def __init__(self, language_model_path=None, device=None, img_feat_ext=None, prompt_type='text', args=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args or parse_args()  # 使用传入的参数或默认
        self.language_model_path = language_model_path or self.args.language_model_path
        
        model_name = self.args.model_name if hasattr(self.args, 'model_name') else "未指定"
        
        if self.language_model_path is None:
            self.language_model_path = MODEL_PATHS.get(model_name, model_name)
        
        print(f"使用预训练语言模型: {model_name} ({self.language_model_path})")
        
        if "llava" in model_name.lower() or "llava" in (self.language_model_path or "").lower():  
            self.model_type = "llava"
            print(f"检测到模型类型: LLaVA")
        elif "llama" in model_name.lower() or "llama" in (self.language_model_path or "").lower():
            self.model_type = "llama"
            print(f"检测到模型类型: Llama")
        else:
            self.model_type = "gpt2"
            print(f"检测到模型类型: GPT-2")

        self.robot_keywords = [
            "robot",
            "humanoid",
            "bipedal",
            "mechanical",
            "machine"
        ]
        self.posture_keywords = ["leg", "knee", "joint", "foot", "feet", "stance", "posture", "gait",
                               "balance", "walking", "step", "stride", "support", "motion", "angle"]

        self.keywords = self._init_keywords()
        
        self.keyword_groups = init_keyword_groups()
        print(f"已初始化{len(self.keyword_groups)}个关键词组")
        
        # 是否使用关键词引导生成
        self.use_keyword_guided = hasattr(self.args, 'use_keyword_guided_generation') and self.args.use_keyword_guided_generation
        print(f"关键词引导生成: {'启用' if self.use_keyword_guided else '禁用'}")
        
        # 加载关键词引导模板
        self.keyword_guided_prompts = []
        self.keyword_evolution_prompts = []
        
        # 添加格式指导模板
        self.format_guides = []
        
        self._init_model()

        self.img_feat_extractor = img_feat_ext
        self.prompt_type = prompt_type

        self.clip_model, self.clip_preprocess = get_clip_model(self.device)

        self.use_llava = self.model_type == "llava"
        if self.use_llava:
            print(f"将使用LLaVA子进程进行图像描述")
            
        self._load_templates_from_csv()

        with torch.no_grad():
            self.ideal_tokens = torch.cat(
                [clip.tokenize(prompt) for prompt in self.ideal_prompts]).to(self.device)
            self.ideal_features = self.clip_model.encode_text(
                self.ideal_tokens)
            self.ideal_features = F.normalize(self.ideal_features, dim=1)

            if self.negative_prompts:
                self.neg_tokens = torch.cat(
                    [clip.tokenize(prompt) for prompt in self.negative_prompts]).to(self.device)
                self.neg_features = self.clip_model.encode_text(
                    self.neg_tokens)
                self.neg_features = F.normalize(self.neg_features, dim=1)

            print(
                f"已初始化{len(self.ideal_prompts)}个理想模板和{len(self.negative_prompts)}个负面模板特征向量")

        self.cached_features = {}

        self.weights = {
            "avg_similarity": self.args.avg_similarity_weight,
            "state_consistency": self.args.state_consistency_weight,
            "action_consistency": self.args.action_consistency_weight
        }

        self.concept_history = defaultdict(list)
        self.concept_fitness = {}
        self.dead_rounds = defaultdict(int)

    def _init_keywords(self):
        return ["robot", "agent", "human", "controller", "control", "environment", "env",
                "task", "physics", "simulation", "domain", "state", "dynamics", "mujoco",
                "algorithm", "policy", "network", "action", "reward", "observation", "training",
                "model", "trajectory", "episode", "position", "rotation", "orientation", "quaternion",
                "angle", "velocity", "acceleration", "force", "torque", "joint", "link", "body",
                "inertia", "mass", "friction", "gravity", "contact", "collision", "constraint",
                "dimension", "space", "coordinate", "system", "frame", "vector", "matrix",
                "balance", "walking", "step", "stride", "support", "motion", "angle"]

    def _init_model(self):
        if self.language_model_path:
            if self.model_type == "llama":
                from transformers import LlamaTokenizer, LlamaForCausalLM
                print(f"[Llama] 正在加载 {self.language_model_path}...")
                
                is_local_path = os.path.exists(self.language_model_path)
                if is_local_path:
                    print(f"检测到本地路径: {self.language_model_path}")
                    
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.language_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    cache_dir=self.language_model_path if is_local_path else None
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.lm_model = LlamaForCausalLM.from_pretrained(
                    self.language_model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=True,
                    cache_dir=self.language_model_path if is_local_path else None
                ).to(self.device)
                print(f"[Llama] 模型已加载到 {self.device}")
            elif self.model_type == "llava":
                # 使用子进程方式运行LLaVA
                print(f"[LLaVA] 将使用子进程模式运行，不加载LLaVA模型到当前环境")
                # 设置空的tokenizer和模型以保持API兼容性
                self.tokenizer = DummyTokenizer()
                self.lm_model = None
            else:
                self._load_gpt2_model(self.language_model_path)
        else:
            raise ValueError("必须指定有效的语言模型路径！")

    def _load_gpt2_model(self, model_path):
        self.model_type = "gpt2"
        if "xl" in model_path.lower():
            model_size = "xl"
        elif "large" in model_path.lower():
            model_size = "large"
        elif "medium" in model_path.lower():
            model_size = "medium"
        else:
            model_size = "small"
        
        print(f"[GPT2-{model_size}] 正在加载 {model_path}...")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("已设置tokenizer.pad_token = tokenizer.eos_token")

            self.lm_model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
            print(f"[GPT2-{model_size}] 模型已加载到 {self.device}")
        except Exception as e:
            print(f"加载GPT-2模型失败: {e}")
            raise

    def _load_templates_from_csv(self, template_file="/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP/tests/concept_templates.csv"):
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"模板文件 {template_file} 不存在！")
        
        print(f"从文件加载文本模板: {template_file}")
        self.ideal_prompts = []
        self.negative_prompts = []
        self.robot_prompts = []
        self.backup_descriptions = []
        self.evolve_backup_descriptions = []
        

        self.generation_prompts = []
        self.evolution_prompts = []
        self.requirements_texts = []
        self.examples_intros = []
        self.instruction_texts = []
        
        self.keyword_guided_prompts = []
        self.keyword_evolution_prompts = []
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category = row['category']
                    text = row['text']
                    
                    if category == 'ideal_prompt':
                        self.ideal_prompts.append(text)
                    elif category == 'negative_prompt':
                        self.negative_prompts.append(text)
                    elif category == 'robot_prompt':
                        self.robot_prompts.append(text)
                    elif category == 'backup_description':
                        self.backup_descriptions.append(text)
                    elif category == 'evolve_backup':
                        self.evolve_backup_descriptions.append(text)
                    elif category == 'generation_prompt':
                        self.generation_prompts.append(text)
                    elif category == 'evolution_prompt':
                        self.evolution_prompts.append(text)
                    elif category == 'requirements':
                        self.requirements_texts.append(text)
                    elif category == 'examples_intro':
                        self.examples_intros.append(text)
                    elif category == 'instruction':
                        self.instruction_texts.append(text)
                    elif category == 'keyword_guided_prompt':
                        self.keyword_guided_prompts.append(text)
                    elif category == 'keyword_evolution_prompt':
                        self.keyword_evolution_prompts.append(text)
                    elif category == 'format_guide':
                        self.format_guides.append(text)
            
            # 检查关键词引导模板是否加载
            if not self.keyword_guided_prompts:
                self.keyword_guided_prompts.append("Generate a description of a robot's walking posture that MUST use these keywords: [KEYWORDS].")
            
            if not self.keyword_evolution_prompts:
                self.keyword_evolution_prompts.append("Evolve this description: \"[BEST_DESCRIPTION]\" into a new description that emphasizes these keywords: [KEYWORDS].")
            
            # 检查必要模板是否加载，如果没有则提供默认值
            if not self.generation_prompts:
                self.generation_prompts.append("Generate a description of a humanoid robot's walking posture, focusing on specific leg movements, joint angles, and balance.")
                
            if not self.evolution_prompts:
                self.evolution_prompts.append("Generate a new walking posture description similar to: \"[BEST_DESCRIPTION]\"")
                
            if not self.requirements_texts:
                self.requirements_texts.append("""- Description MUST be between 10-30 words only
- MUST use exactly the format: "[Type of robot] with [specific leg position] in [movement/balance state]"
- MUST include at least 3 of these keywords: [leg, knee, joint, foot, ankle, posture, stance, balance, gait]
- MUST focus ONLY on walking posture and leg movements
- NO technical terms, URLs, or unnecessary details
- NO emojis or special characters
- NO code syntax or symbols
- Use plain English text with standard punctuation""")
                
            if not self.examples_intros:
                self.examples_intros.append("EXAMPLES of good descriptions:")
                
            if not self.instruction_texts:
                self.instruction_texts.append("Your description MUST follow all requirements above. Violations will be rejected.\n\nDescription: ")
                
            if not self.ideal_prompts or not self.negative_prompts or not self.robot_prompts:
                raise ValueError("CSV模板文件格式错误，必须包含ideal_prompt、negative_prompt和robot_prompt类别的文本")
                
            print(f"模板加载完成：共{len(self.ideal_prompts)}个理想模板")
            
            # 检查格式指导模板是否加载
            if not self.format_guides:
                self.format_guides.append("Use a clear format describing the robot's posture")
            
        except Exception as e:
            raise IOError(f"读取模板文件出错: {e}")

    def collect_state_representations(self, states):
        return collect_state_representations_with_clip(states, self.clip_model, self.clip_preprocess, self.device)

    def _encode_batch_text(self, descriptions):
        return _encode_batch_text(descriptions, self.clip_model, self.device, self.cached_features)

    def _similarity_too_high(self, new_desc, existing_descs, threshold=None):
        # 禁用相似度过滤功能，始终返回False
        return False
        
        # 原始代码
        """
        threshold = threshold or self.args.similarity_threshold
        return _similarity_too_high(new_desc, existing_descs, self.clip_model, self.device, threshold, self.cached_features)
        """

    def evaluate_single_concept(self, description, debug=False):
        fitness = 1.0

        try:
            if description in self.cached_features:
                features = self.cached_features[description].unsqueeze(0)
            else:
                with torch.no_grad():
                    tokens = clip.tokenize([description]).to(self.device)
                    features = self.clip_model.encode_text(tokens)
                    features = F.normalize(features, dim=1)
                    self.cached_features[description] = features.squeeze(0)

            similarity_scores = (features @ self.ideal_features.T).squeeze(0)
            avg_similarity = similarity_scores.mean().item()
            max_similarity = similarity_scores.max().item()

            contrast_score = 0
            if hasattr(self, 'neg_features'):
                neg_similarity = (features @ self.neg_features.T).squeeze(0)
                avg_neg_similarity = neg_similarity.mean().item()
                contrast_score = max(0, max_similarity - avg_neg_similarity)

            if self.args.semantic_score_type == 'raw':
                semantic_score = max_similarity
            else:
                semantic_score = max(0.5, min(1.3, 1.0 + contrast_score))

            fitness *= semantic_score

            if debug:
                print(
                    f"语义相似度得分: {semantic_score:.4f} (avg={avg_similarity:.4f}, max={max_similarity:.4f})")
                if hasattr(self, 'neg_features'):
                    print(f"与负面模板对比分数: {contrast_score:.4f}")
        except Exception as e:
            print(f"  CLIP计算出错: {e}")
            return 0.0

        text = description.lower()

        if len(text) < 20:
            fitness *= 0.8
            if debug:
                print(f"描述过短 ({len(text)}字符)")

        words = text.split()
        if len(words) < 8:
            fitness *= 0.9
            if debug:
                print(f"描述词汇量较少 ({len(words)}个词)")

        word_freq = {}
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1

        repetition_penalty = 0
        for word, count in word_freq.items():
            if count > 1 and len(word) > 3:
                repetition_penalty += (count - 1) * 0.05

        if repetition_penalty > 0:
            fitness *= max(0.7, 1.0 - repetition_penalty)
            if debug and repetition_penalty > 0:
                print(f"重复词惩罚: {repetition_penalty:.2f}")

        keyword_bonus = 0
        for keyword in self.posture_keywords:
            if keyword in text:
                keyword_bonus += 0.02

        fitness *= (1 + min(0.2, keyword_bonus))

        return fitness

    def generate_text_with_llava(self, prompt, num_candidates=1, states=None):
        # 检查是否有环境状态数据可用
        print(f"\n=== LLaVA文本生成开始 ===")
        print(f"请求生成{num_candidates}个候选描述")
        
        if states is None or len(states) == 0:
            print(f"未提供有效的状态数据，将使用备用描述")
            # 从备用描述中随机选择
            if len(self.backup_descriptions) > 0:
                backup_indices = list(range(len(self.backup_descriptions)))
                random.shuffle(backup_indices)
                
                result = []
                for i in range(min(num_candidates, len(self.backup_descriptions))):
                    desc = self.backup_descriptions[backup_indices[i % len(self.backup_descriptions)]]
                    result.append(desc)
                    print(f"使用备用描述 {i+1}: {desc}")
                return result
            else:
                # 如果没有备用描述，返回默认描述
                default_desc = "Humanoid robot with balanced stance in walking motion."
                print(f"无可用备用描述，使用默认描述: {default_desc}")
                return [default_desc] * num_candidates
        
        from concept_utils import process_states_with_llava
        
        llava_prompt = prompt
        print(f"使用提示: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
        
        llava_env_path = self.args.llava_env_path if hasattr(self.args, 'llava_env_path') else None
        
        if llava_env_path and not os.path.exists(llava_env_path):
            print(f"指定的LLaVA环境路径不存在: {llava_env_path}，将使用默认Python")
            llava_env_path = "python"
        
        
        model_path = None
        if self.model_type == "llava":
            # 如果当前模型就是LLaVA，直接使用其路径
            model_path = self.language_model_path
            print(f"使用当前LLaVA模型路径: {model_path}")
        else:
            # 否则从MODEL_PATHS中查找LLaVA模型路径
            model_path = MODEL_PATHS.get("llava-1.5-7b")
            print(f"当前不是LLaVA模型，使用默认LLaVA模型: {model_path}")
        
        # 调用process_states_with_llava函数处理图像并生成描述
        try:
            print(f"开始LLaVA子进程处理图像...")
            descriptions = process_states_with_llava(
                states=states, 
                model_path=model_path,
                llava_env_path=llava_env_path, 
                prompt=llava_prompt
            )
            
            print(f"LLaVA处理完成，获得{len(descriptions)}个描述")
            
            # 如果生成的描述不足，需要补充
            if len(descriptions) < num_candidates:
                if len(descriptions) > 0:
                    # 复制现有描述
                    original_count = len(descriptions)
                    print(f"生成的描述数量({original_count})不足请求数量({num_candidates})，将复制现有描述")
                    while len(descriptions) < num_candidates:
                        idx = len(descriptions) % original_count
                        descriptions.append(descriptions[idx])
                        print(f"复制描述 {idx+1}: {descriptions[idx]}")
                else:
                    # 使用备用描述
                    print(f"未生成任何有效描述，使用备用描述")
                    if len(self.backup_descriptions) > 0:
                        backup_indices = list(range(len(self.backup_descriptions)))
                        random.shuffle(backup_indices)
                        
                        for i in range(num_candidates):
                            idx = i % len(self.backup_descriptions)
                            descriptions.append(self.backup_descriptions[backup_indices[idx]])
                            print(f"使用备用描述 {i+1}: {self.backup_descriptions[backup_indices[idx]]}")
                    else:
                        default_desc = "Humanoid robot with balanced stance in walking motion."
                        print(f"无可用备用描述，使用默认描述: {default_desc}")
                        descriptions = [default_desc] * num_candidates
            
            print(f"=== LLaVA文本生成结束，返回{len(descriptions)}个描述 ===\n")
            return descriptions
            
        except Exception as e:
            import traceback
            print(f"LLaVA处理过程中出错: {e}")
            print(f"错误堆栈: {traceback.format_exc()}")
            print(f"将使用备用描述")
            
            # 使用备用描述
            if len(self.backup_descriptions) > 0:
                backup_indices = list(range(len(self.backup_descriptions)))
                random.shuffle(backup_indices)
                
                result = []
                for i in range(min(num_candidates, len(self.backup_descriptions))):
                    desc = self.backup_descriptions[backup_indices[i % len(self.backup_descriptions)]]
                    result.append(desc)
                    print(f"使用备用描述 {i+1}: {desc}")
                return result
            else:
                default_desc = "Humanoid robot with balanced stance in walking motion."
                print(f"无可用备用描述，使用默认描述: {default_desc}")
                return [default_desc] * num_candidates

    def generate_initial_concepts(self, prompt=None, n=10, n_variations=3, states=None):
        if len(self.backup_descriptions) < n * n_variations:
            print("备用描述数量不足，可能需要重复使用")
        
        concept_set = {}
        
        if not prompt:
            # 使用CSV中的模板构建提示
            # 随机选择一个生成提示
            generation_prompt = random.choice(self.generation_prompts)
            # 随机选择一个要求文本
            requirements = random.choice(self.requirements_texts)
            # 随机选择一个示例介绍
            examples_intro = random.choice(self.examples_intros)
            # 随机选择一个指导文本
            instruction = random.choice(self.instruction_texts)
            
            # 构建示例
            examples = []
            for i, desc in enumerate(self.backup_descriptions[:5]):
                examples.append(f"{i+1}. \"{desc}\"")
            examples_text = "\n".join(examples)
            
            # 随机选择一个格式指导
            format_guide = random.choice(self.format_guides)
            
            # 组合完整提示，不包含图像描述上下文
            prompt = f"{generation_prompt}\n\n{format_guide}\n\nSTRICT REQUIREMENTS:\n{requirements}\n\n{examples_intro}\n{examples_text}\n\n{instruction}"
        
        print("生成初始概念变体...")
        
        prompt_tokens = len(self.tokenizer.encode(prompt))
        max_length = max(prompt_tokens + 100, self.args.gpt2_max_length)
        print(f"  提示长度: {prompt_tokens} tokens, 设置生成最大长度: {max_length} tokens")
        
        for i in range(n):
            concept_id = f"concept_{i}"
            concept_set[concept_id] = []
            
            candidates = []
            
            generation_success = False
            
            if self.model_type == "llava":
                # 使用LLaVA模式特定的生成方法，传递states参数
                candidates = self.generate_text_with_llava(prompt, self.args.num_candidates, states)
                generation_success = len(candidates) > 0
            else:
                try:
                    input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                    attention_mask = torch.ones(input_ids.shape, device=self.device)
                    
                    outputs = self.lm_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=self.args.temperature,
                        top_k=self.args.gpt2_top_k,
                        top_p=self.args.gpt2_top_p,
                        do_sample=True,
                        num_return_sequences=self.args.num_candidates,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=self.args.gpt2_repetition_penalty
                    )
                    
                    for output in outputs:
                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        
                        if "Description:" in generated_text:
                            description = generated_text.split("Description:")[-1].strip()
                        else:
                            for sep in ['\n', '. ', '! ', '? ']:
                                if sep in generated_text:
                                    parts = generated_text.split(sep)
                                    description = parts[-1].strip()
                                    break
                            else:
                                description = generated_text[len(prompt):].strip()
                        
                        description = clean_description(
                            description, 
                            robot_keywords=self.robot_keywords,
                            posture_keywords=self.posture_keywords,
                            keywords_filter=self.args.keywords_filter,
                            strict_format=True,
                            word_limit=(10, 30)
                        )
                        
                        if description:
                            if not any(self._similarity_too_high(description, [c]) for c in candidates):
                                candidates.append(description)
                    
                    generation_success = True
                except Exception as e:
                    print(f"  生成出错: {e}")
                    generation_success = False
            
            # 根据生成结果添加变体
            if len(self.backup_descriptions) == 0:
                concept_set[concept_id] = []
                print(f" 无备用描述可用，概念 {concept_id} 将没有变体")
            else:
                if not generation_success or len(candidates) < n_variations:
                    # 使用备用描述
                    for j in range(min(n_variations, len(self.backup_descriptions))):
                        backup = self.backup_descriptions[j]
                        concept_set[concept_id].append(backup)
                        print(f"  添加变体: {backup}")
                else:
                    # 使用生成的候选
                    new_candidates = []
                    for j in range(min(n_variations, len(candidates))):
                        new_candidates.append(candidates[j])
                        concept_set[concept_id].append(candidates[j])
                        print(f"  添加变体: {candidates[j]}")
                    candidates = new_candidates
        
        return concept_set

    def evaluate_concept_fitness(self, concepts, states, actions,
                           use_avg_similarity=True,
                           use_state_consistency=True,
                           use_action_consistency=True):
        flat_concepts = []
        concept_indices = []
        for concept_id, variants in concepts.items():
            for variant in variants:
                flat_concepts.append(variant)
                idx = int(concept_id.split('_')[1])
                concept_indices.append(idx)

        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc)
                                    for desc in flat_concepts]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=1)

        state_features = self.collect_state_representations(states)

        similarities = torch.mm(state_features, text_features.T)

        fitness_scores = {}
        for concept_idx in set(concept_indices):
            variant_indices = [i for i, idx in enumerate(
                concept_indices) if idx == concept_idx]

            concept_similarities = similarities[:, variant_indices].max(dim=1)[
                                                                        0]

            concept_states = []
            concept_actions = []
            for i in range(len(states)):
                if torch.argmax(similarities[i]) in variant_indices:
                    concept_states.append(state_features[i])
                    concept_actions.append(actions[i])

            if not concept_states:
                avg_sim = concept_similarities.mean().item()
                fitness_scores[f"concept_{concept_idx}"] = max(
                    0.01, avg_sim * 0.2)
                # 禁用概念死亡统计
                # self.dead_rounds[f"concept_{concept_idx}"] += 1
                print(
                    f"概念 {concept_idx} 无匹配状态，分配基线适应度: {fitness_scores[f'concept_{concept_idx}']:.4f}")
                # print(
                #     f"概念 {concept_idx} 已累计 {self.dead_rounds[f'concept_{concept_idx}']} 次无效")
                continue

            concept_states = torch.stack(concept_states)
            concept_actions = torch.from_numpy(
    np.stack(concept_actions)).to(
        self.device)

            # 禁用概念死亡统计
            # self.dead_rounds[f"concept_{concept_idx}"] = 0

            if use_avg_similarity:
                avg_similarity = concept_similarities.mean()
            else:
                avg_similarity = torch.tensor(0.0)

            if use_state_consistency and len(concept_states) > 1:
                state_sim_matrix = torch.mm(concept_states, concept_states.T)
                state_consistency = (state_sim_matrix.sum(
                ) - state_sim_matrix.trace()) / (len(concept_states) * (len(concept_states) - 1))
            else:
                state_consistency = torch.tensor(0.0)

            if use_action_consistency and len(concept_actions) > 1:
                action_sim_matrix = F.cosine_similarity(
    concept_actions.unsqueeze(1), concept_actions.unsqueeze(0), dim=2)
                action_consistency = (action_sim_matrix.sum(
                ) - action_sim_matrix.trace()) / (len(concept_actions) * (len(concept_actions) - 1))
            else:
                action_consistency = torch.tensor(0.0)

            fitness = (self.weights["avg_similarity"] * avg_similarity +
                      self.weights["state_consistency"] * state_consistency +
                      self.weights["action_consistency"] * action_consistency)

            fitness_scores[f"concept_{concept_idx}"] = fitness.item()

        self.concept_fitness.update(fitness_scores)
        return fitness_scores

    def generate_keyword_guided_concepts(self, n=10, n_variations=3, states=None):
        if len(self.backup_descriptions) < n * n_variations:
            print("警告: 备用描述数量不足，可能需要重复使用")
        
        concept_set = {}
        
        # 获取关键词组配置
        min_keywords = self.args.min_keywords if hasattr(self.args, 'min_keywords') else 3
        max_keywords = self.args.max_keywords if hasattr(self.args, 'max_keywords') else 7
        
        # 获取要关注的关键词组
        selected_groups = self.args.keyword_groups_focus if hasattr(self.args, 'keyword_groups_focus') else None
        
        for i in range(n):
            concept_id = f"concept_{i}"
            concept_set[concept_id] = []
            
            # 为每个概念选择关键词
            selected_keywords = select_keywords_from_groups(
                self.keyword_groups,
                selected_groups=selected_groups,
                min_total=min_keywords,
                max_total=max_keywords
            )
            
            # 确保关键词中至少有一个机器人类型和一个姿势类型
            has_robot_type = any(kw in selected_keywords for kw in self.robot_keywords)
            has_posture = any(kw in selected_keywords for kw in self.posture_keywords)
            
            if not has_robot_type:
                selected_keywords.append(random.choice(self.robot_keywords))
            if not has_posture:
                selected_keywords.append(random.choice(self.posture_keywords))
            
            print(f"概念 {concept_id} 选择的关键词: {', '.join(selected_keywords)}")
            
            # 选择提示模板
            prompt_template = random.choice(self.keyword_guided_prompts)
            
            # 插入关键词
            keywords_text = ", ".join(selected_keywords)
            prompt = prompt_template.replace("[KEYWORDS]", keywords_text)
            
            # 添加要求和示例
            requirements = random.choice(self.requirements_texts)
            examples_intro = random.choice(self.examples_intros)
            instruction = random.choice(self.instruction_texts)
            
            # 构建示例部分
            examples = []
            for j, desc in enumerate(self.backup_descriptions[:3]):
                examples.append(f"{j+1}. \"{desc}\"")
            examples_text = "\n".join(examples)
            
            # 组合完整提示
            full_prompt = f"{prompt}\n\nRequirements:\n{requirements}\n\n{examples_intro}\n{examples_text}\n\n{instruction}"
            
            print(f"生成概念 {concept_id} 的变体...")
            prompt_tokens = len(self.tokenizer.encode(full_prompt))
            max_length = max(prompt_tokens + 100, self.args.gpt2_max_length)
            
            candidates = []
            generation_success = False
            
            if self.model_type == "llava":
                # 使用LLaVA模式特定的生成方法，传递states参数
                batch_candidates = self.generate_text_with_llava(full_prompt, self.args.num_candidates, states)
                generation_success = len(batch_candidates) > 0
            else:
                try:
                    input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.device)
                    attention_mask = torch.ones(input_ids.shape, device=self.device)
                    
                    outputs = self.lm_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=self.args.temperature,
                        top_k=self.args.gpt2_top_k,
                        top_p=self.args.gpt2_top_p,
                        do_sample=True,
                        num_return_sequences=self.args.num_candidates,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=self.args.gpt2_repetition_penalty
                    )
                    
                    for output in outputs:
                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        
                        if "Description:" in generated_text:
                            description = generated_text.split("Description:")[-1].strip()
                        else:
                            for sep in ['\n', '. ', '! ', '? ']:
                                if sep in generated_text:
                                    parts = generated_text.split(sep)
                                    description = parts[-1].strip()
                                    break
                            else:
                                description = generated_text[len(full_prompt):].strip()
                        
                        description = clean_description(
                            description, 
                            robot_keywords=self.robot_keywords,
                            posture_keywords=self.posture_keywords,
                            keywords_filter=self.args.keywords_filter,
                            strict_format=True,
                            word_limit=(10, 30)
                        )
                        
                        if description:
                            if not any(self._similarity_too_high(description, [c]) for c in candidates):
                                candidates.append(description)
                    
                    generation_success = True
                except Exception as e:
                    print(f"  生成出错: {e}")
                    generation_success = False
                
                if len(candidates) < n_variations:
                    print(f"  没有合格的候选，评估备用描述")
                    backup_indices = list(range(len(self.backup_descriptions)))
                    random.shuffle(backup_indices)
                    
                    # 如果没有备用描述，则创建空列表
                    if len(self.backup_descriptions) == 0:
                        concept_set[concept_id] = []
                        print(f" 无备用描述可用，概念 {concept_id} 将没有变体")
                    else:
                        for j in range(min(n_variations, len(self.backup_descriptions))):
                            backup = self.backup_descriptions[backup_indices[j % len(self.backup_descriptions)]]
                            concept_set[concept_id].append(backup)
                            print(f"  添加变体: {backup}")
                else:
                    # 使用生成的候选
                    new_candidates = []
                    for j in range(min(n_variations, len(candidates))):
                        new_candidates.append(candidates[j])
                        concept_set[concept_id].append(candidates[j])
                        print(f"  添加变体: {candidates[j]}")
                    candidates = new_candidates
        
        return concept_set

    def evolve_concepts(self, concepts, fitness_scores, num_variants=3, states=None):
        evolved_concepts = {k: v.copy() for k, v in concepts.items()}
        
        backup_descriptions = self.evolve_backup_descriptions
        if not backup_descriptions:
            backup_descriptions = self.backup_descriptions
        
        for concept_id in list(concepts.keys()):
            # 禁用概念死亡判断
            """
            rounds_invalid = self.dead_rounds.get(concept_id, 0)
            
            if rounds_invalid >= 2 and fitness_scores.get(concept_id, 0) < 0.1:
                print(f"跳过概念 {concept_id}: 连续{rounds_invalid}轮适应度接近0")
                continue
            """
            
            variants = concepts[concept_id]
            if not variants:
                print(f"概念 {concept_id} 没有可用的变体，将跳过")
                continue
                
            scored_variants = []
            for var in variants:
                score = self.evaluate_single_concept(var)
                scored_variants.append((var, score))

            scored_variants.sort(key=lambda x: x[1], reverse=True)
            best_var, best_score = scored_variants[0]

            print(f"\n处理概念 {concept_id}:")
            print(f"最佳变体: {best_var} (得分: {best_score:.4f})")
            
            variants_needed = max(1, num_variants - len(variants))
            
            candidates = []

            max_attempts = 3
            for attempt in range(max_attempts):
                if len(candidates) >= variants_needed:
                    break

                print(f"尝试 {attempt+1}/{max_attempts} 生成候选...")

                # 始终使用CSV中的模板构建提示
                # 随机选择一个演化提示，替换[BEST_DESCRIPTION]占位符
                evolution_template = random.choice(self.evolution_prompts)
                evolution_prompt = evolution_template.replace("[BEST_DESCRIPTION]", best_var)
                
                # 随机选择一个要求文本
                requirements = random.choice(self.requirements_texts)
                
                # 随机选择一个示例介绍
                examples_intro = random.choice(self.examples_intros)
                
                # 随机选择一个指导文本
                instruction = random.choice(self.instruction_texts)
                
                # 构建示例部分
                examples = []
                for i, desc in enumerate(self.backup_descriptions[:5]):
                    examples.append(f"{i+1}. \"{desc}\"")
                examples_text = "\n".join(examples)
                
                # 组合完整提示，不包含图像描述上下文
                prompt = f"{evolution_prompt}\n\nSTRICT REQUIREMENTS:\n{requirements}\n\n{examples_intro}\n{examples_text}\n\n{instruction}"

                batch_candidates = []
                generation_success = False
                
                if self.model_type == "llava":
                    batch_candidates = self.generate_text_with_llava(prompt, self.args.num_candidates, states)
                    generation_success = len(batch_candidates) > 0
                else:
                    try:
                        prompt_tokens = len(self.tokenizer.encode(prompt))
                        max_length = max(prompt_tokens + 100, self.args.gpt2_max_length)
                        
                        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                        attention_mask = torch.ones(input_ids.shape, device=self.device)

                        outputs = self.lm_model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            temperature=self.args.temperature,
                            top_k=self.args.gpt2_top_k,
                            top_p=self.args.gpt2_top_p,
                            do_sample=True,
                            num_return_sequences=self.args.num_candidates,
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=self.args.gpt2_repetition_penalty
                        )

                        print(f"  成功生成 {len(outputs)} 个候选")

                        for output in outputs:
                            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)

                            if "Description:" in generated_text:
                                description = generated_text.split("Description:")[-1].strip()
                            else:
                                for sep in ['\n', '. ', '! ', '? ']:
                                    if sep in generated_text:
                                        parts = generated_text.split(sep)
                                        description = parts[-1].strip()
                                        break
                                else:
                                    description = generated_text[len(prompt):].strip()

                            description = clean_description(
                                description, 
                                robot_keywords=self.robot_keywords,
                                posture_keywords=self.posture_keywords,
                                keywords_filter=self.args.keywords_filter,
                                strict_format=True,
                                word_limit=(10, 30)
                            )
                            if description:
                                if not any(self._similarity_too_high(description, [c]) for c in batch_candidates):
                                    batch_candidates.append(description)
                    
                        generation_success = True
                    except Exception as e:
                        print(f"  生成出错: {e}")
                        generation_success = False

                if not generation_success or len(batch_candidates) < variants_needed:
                        print(f"  没有合格的候选，评估备用描述")
                        backup_indices = list(range(len(backup_descriptions)))
                        random.shuffle(backup_indices)
                        
                        for j in range(min(variants_needed, len(backup_descriptions))):
                            backup = backup_descriptions[backup_indices[j % len(backup_descriptions)]]
                            candidates.append(backup)
                            print(f"  添加变体: {backup}")
                else:
                    for j in range(min(variants_needed, len(batch_candidates))):
                        candidates.append(batch_candidates[j])
                        print(f"  添加变体: {batch_candidates[j]}")

            filtered_candidates = []
            for candidate in candidates:
                if not self._similarity_too_high(candidate, evolved_concepts[concept_id]):
                    filtered_candidates.append(candidate)

            print(f"  生成了{len(candidates)}个候选，过滤重复后剩余{len(filtered_candidates)}个")

            if filtered_candidates:
                scored_candidates = []
                for desc in filtered_candidates:
                    score = self.evaluate_single_concept(desc)
                    scored_candidates.append((desc, score))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)

                selected = []
                for desc, score in scored_candidates:
                    if len(selected) >= variants_needed:
                        break

                    if not any(self._similarity_too_high(desc, [s]) for s in selected):
                        selected.append(desc)
                        print(f"  添加变体: {desc} (得分: {score:.4f})")

                evolved_concepts[concept_id].extend(selected)
            else:
                print("  无有效候选，使用备用描述")
                backup_candidates = []
                for desc in backup_descriptions:
                    if not self._similarity_too_high(desc, evolved_concepts[concept_id]):
                        score = self.evaluate_single_concept(desc)
                        backup_candidates.append((desc, score))

                backup_candidates.sort(key=lambda x: x[1], reverse=True)
                for desc, score in backup_candidates[:variants_needed]:
                    evolved_concepts[concept_id].append(desc)
                    print(f"  添加备用变体: {desc} (得分: {score:.4f})")

        return evolved_concepts
        
    def evolve_concepts_with_keywords(self, concepts, fitness_scores, num_variants=3, states=None):
        evolved_concepts = {k: v.copy() for k, v in concepts.items()}
        
        backup_descriptions = self.evolve_backup_descriptions
        if not backup_descriptions:
            backup_descriptions = self.backup_descriptions
            
        # 获取关键词组配置
        min_keywords = self.args.min_keywords if hasattr(self.args, 'min_keywords') else 3
        max_keywords = self.args.max_keywords if hasattr(self.args, 'max_keywords') else 7
        
        # 获取要关注的关键词组
        selected_groups = self.args.keyword_groups_focus if hasattr(self.args, 'keyword_groups_focus') else None
        
        for concept_id in list(concepts.keys()):
            # 禁用概念死亡判断
            """
            rounds_invalid = self.dead_rounds.get(concept_id, 0)
            
            if rounds_invalid >= 2 and fitness_scores.get(concept_id, 0) < 0.1:
                print(f"跳过概念 {concept_id}: 连续{rounds_invalid}轮适应度接近0")
                continue
            """
            
            variants = concepts[concept_id]
            if not variants:
                print(f"概念 {concept_id} 没有可用的变体，将跳过")
                continue
                
            scored_variants = []
            for var in variants:
                score = self.evaluate_single_concept(var)
                scored_variants.append((var, score))

            scored_variants.sort(key=lambda x: x[1], reverse=True)
            best_var, best_score = scored_variants[0]

            print(f"\n处理概念 {concept_id}:")
            print(f"最佳变体: {best_var} (得分: {best_score:.4f})")
            
            # 为这个概念选择新的关键词
            selected_keywords = select_keywords_from_groups(
                self.keyword_groups,
                selected_groups=selected_groups,
                min_total=min_keywords,
                max_total=max_keywords
            )
            
            # 确保关键词中至少有一个机器人类型和一个姿势类型
            has_robot_type = any(kw in selected_keywords for kw in self.robot_keywords)
            has_posture = any(kw in selected_keywords for kw in self.posture_keywords)
            
            if not has_robot_type:
                selected_keywords.append(random.choice(self.robot_keywords))
            if not has_posture:
                selected_keywords.append(random.choice(self.posture_keywords))
                
            print(f"  选择的关键词: {', '.join(selected_keywords)}")
            
            variants_needed = max(1, num_variants - len(variants))
            candidates = []

            max_attempts = 3
            for attempt in range(max_attempts):
                if len(candidates) >= variants_needed:
                    break

                print(f"尝试 {attempt+1}/{max_attempts} 生成候选...")

                # 随机选择一个关键词演化提示模板
                evolution_template = random.choice(self.keyword_evolution_prompts)
                
                # 替换模板中的占位符
                keywords_text = ", ".join(selected_keywords)
                evolution_prompt = evolution_template.replace("[BEST_DESCRIPTION]", best_var)
                evolution_prompt = evolution_prompt.replace("[KEYWORDS]", keywords_text)
                
                # 随机选择一个要求文本
                requirements = random.choice(self.requirements_texts)
                
                # 随机选择一个示例介绍
                examples_intro = random.choice(self.examples_intros)
                
                # 随机选择一个指导文本
                instruction = random.choice(self.instruction_texts)
                
                # 构建示例部分
                examples = []
                for i, desc in enumerate(self.backup_descriptions[:3]):
                    examples.append(f"{i+1}. \"{desc}\"")
                examples_text = "\n".join(examples)
                
                # 组合完整提示，不包含图像描述上下文
                prompt = f"{evolution_prompt}\n\nRequirements:\n{requirements}\n\n{examples_intro}\n{examples_text}\n\n{instruction}"

                batch_candidates = []
                generation_success = False
                
                if self.model_type == "llava":
                    batch_candidates = self.generate_text_with_llava(prompt, self.args.num_candidates, states)
                    generation_success = len(batch_candidates) > 0
                else:
                    try:
                        prompt_tokens = len(self.tokenizer.encode(prompt))
                        max_length = max(prompt_tokens + 100, self.args.gpt2_max_length)
                        
                        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                        attention_mask = torch.ones(input_ids.shape, device=self.device)

                        outputs = self.lm_model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            temperature=self.args.temperature,
                            top_k=self.args.gpt2_top_k,
                            top_p=self.args.gpt2_top_p,
                            do_sample=True,
                            num_return_sequences=self.args.num_candidates,
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=self.args.gpt2_repetition_penalty
                        )

                        print(f"  成功生成 {len(outputs)} 个候选")

                        for output in outputs:
                            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)

                            if "Description:" in generated_text:
                                description = generated_text.split("Description:")[-1].strip()
                            else:
                                for sep in ['\n', '. ', '! ', '? ']:
                                    if sep in generated_text:
                                        parts = generated_text.split(sep)
                                        description = parts[-1].strip()
                                        break
                                else:
                                    description = generated_text[len(prompt):].strip()

                            description = clean_description(
                                description, 
                                robot_keywords=self.robot_keywords,
                                posture_keywords=self.posture_keywords,
                                keywords_filter=self.args.keywords_filter,
                                strict_format=True,
                                word_limit=(10, 30)
                            )
                            
                            if description:
                                if not any(self._similarity_too_high(description, [c]) for c in batch_candidates):
                                    batch_candidates.append(description)
                    
                        generation_success = True
                    except Exception as e:
                        print(f"  生成出错: {e}")
                        generation_success = False

                if not generation_success or len(batch_candidates) < variants_needed:
                    print(f"  没有合格的候选，评估备用描述")
                    backup_indices = list(range(len(backup_descriptions)))
                    random.shuffle(backup_indices)
                    
                    for j in range(min(variants_needed, len(backup_descriptions))):
                        backup = backup_descriptions[backup_indices[j % len(backup_descriptions)]]
                        candidates.append(backup)
                        print(f"  添加变体: {backup}")
                else:
                    for j in range(min(variants_needed, len(batch_candidates))):
                        candidates.append(batch_candidates[j])
                        print(f"  添加变体: {batch_candidates[j]}")

            filtered_candidates = []
            for candidate in candidates:
                if not self._similarity_too_high(candidate, evolved_concepts[concept_id]):
                    filtered_candidates.append(candidate)

            print(f"  生成了{len(candidates)}个候选，过滤重复后剩余{len(filtered_candidates)}个")

            if filtered_candidates:
                scored_candidates = []
                for desc in filtered_candidates:
                    # 评估描述与关键词的匹配度
                    keywords_found = sum(1 for kw in selected_keywords if kw.lower() in desc.lower())
                    keyword_bonus = min(0.2, keywords_found * 0.05)  # 每个关键词增加0.05的分数，最多0.2
                    
                    # 评估描述的质量
                    base_score = self.evaluate_single_concept(desc)
                    
                    # 综合得分
                    final_score = base_score * (1 + keyword_bonus)
                    scored_candidates.append((desc, final_score))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)

                selected = []
                for desc, score in scored_candidates:
                    if len(selected) >= variants_needed:
                        break

                    if not any(self._similarity_too_high(desc, [s]) for s in selected):
                        selected.append(desc)
                        print(f"  添加变体: {desc} (得分: {score:.4f})")

                evolved_concepts[concept_id].extend(selected)
            else:
                print("  无有效候选，使用备用描述")
                backup_candidates = []
                for desc in backup_descriptions:
                    if not self._similarity_too_high(desc, evolved_concepts[concept_id]):
                        score = self.evaluate_single_concept(desc)
                        backup_candidates.append((desc, score))

                backup_candidates.sort(key=lambda x: x[1], reverse=True)
                for desc, score in backup_candidates[:variants_needed]:
                    evolved_concepts[concept_id].append(desc)
                    print(f"  添加备用变体: {desc} (得分: {score:.4f})")

        return evolved_concepts
    
    def save_concepts(self, concepts, file_path):
        save_concepts(concepts, file_path)
    
    def load_concepts(self, file_path):
        return load_concepts(file_path)

# 防止在LLaVA子进程模式下使用real tokenizer时出错
class DummyTokenizer:
    def __init__(self):
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.eos_token_id = 0
        self.pad_token_id = 0
    
    def encode(self, text, return_tensors=None):
        """返回一个假的token序列"""
        if return_tensors == 'pt':
            return torch.ones((1, 10), dtype=torch.long)
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def decode(self, tokens, skip_special_tokens=True):
        """返回一个假的解码文本"""
        return "Dummy decoded text"

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    language_model_path = args.language_model_path
    if language_model_path is None:
        language_model_path = MODEL_PATHS.get(args.model_name, args.model_name)
    print(f"将使用语言模型: {args.model_name} ({language_model_path})")
    
    clip_model, clip_preprocess = get_clip_model(device)
    print("已加载全局CLIP模型")
    
    if make_env is None:
        print("错误: 无法导入环境模块")
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
    rl_model_path = args.rl_model_path if hasattr(args, 'rl_model_path') else getattr(args, 'model_path', None)
    algorithm_name = args.algorithm_name
    
    if rl_model_path and algorithm_name:
        model = load_model(algorithm_name, env, device, rl_model_path)
        print(f"已加载 {algorithm_name} 模型: {rl_model_path}")
    
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
        
        # 如果设置了采样数量且状态数量超过此值，进行采样
        if args.states_per_episode > 0 and len(episode_states) > args.states_per_episode:
            indices = np.linspace(0, len(episode_states) - 1, args.states_per_episode, dtype=int)
            episode_states = [episode_states[i] for i in indices]
            episode_actions = [episode_actions[i] for i in indices]
            print(f"  已采样 {args.states_per_episode} 个状态")
        
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
    
    print(f"总共收集了 {len(all_states)} 个状态-动作对")
    
    print("\n===== 1. 概念自适应与演化 =====")
    concept_generator = ConceptGenerator(language_model_path=language_model_path, device=device, args=args)
    
    csv_file = os.path.join(save_dir, f"{args.domain_name}_{args.task_name}_concept.csv")
    print(f"概念演化记录将保存到: {csv_file}")
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    concept_history = defaultdict(list)
    fitness_history = defaultdict(dict)
    
    # 检查是否使用关键词引导生成
    use_keyword_guided = hasattr(args, 'use_keyword_guided_generation') and args.use_keyword_guided_generation
    
    print("生成初始概念...")
    if use_keyword_guided:
        print("使用关键词引导生成初始概念...")
        initial_concepts = concept_generator.generate_keyword_guided_concepts(
            n=args.n_concepts,
            n_variations=args.n_variants,
            states=all_states  
        )
    else:
        print("使用标准方法生成初始概念...")
        initial_concepts = concept_generator.generate_initial_concepts(
            n=args.n_concepts,
            n_variations=args.n_variants,
            states=all_states  
        )
    
    print("\n初始概念:")
    for concept_id, variants in initial_concepts.items():
        print(f"{concept_id}:")
        if variants:  # 确保变体列表不为空
            concept_history[concept_id].append(variants[0])
        for j, variant in enumerate(variants):
            print(f"  变体 {j+1}: {variant}")
        else:
            print("  无有效变体")
            concept_history[concept_id].append("无有效描述")  
    
    print("\n评估概念适应度...")
    fitness_scores = concept_generator.evaluate_concept_fitness(
        initial_concepts, all_states, all_actions, 
        use_avg_similarity=args.use_avg_similarity, 
        use_state_consistency=args.use_state_consistency, 
        use_action_consistency=args.use_action_consistency
    )
    
    print("适应度分数:")
    for concept_id, score in fitness_scores.items():
        print(f"{concept_id}: {score:.4f}")
        fitness_history[concept_id][0] = score
    
    save_concepts_to_csv(initial_concepts, fitness_scores, 0, csv_file)
    
    concepts = initial_concepts
    for i in range(args.evolution_iterations):
        print(f"\n执行第 {i+1}/{args.evolution_iterations} 次概念演化...")
        
        if use_keyword_guided:
            print("使用关键词引导演化概念...")
            concepts = concept_generator.evolve_concepts_with_keywords(
                concepts=concepts,
                fitness_scores=fitness_scores,
                num_variants=args.n_variants,
                states=all_states  
            )
        else:
            print("使用标准方法演化概念...")
            concepts = concept_generator.evolve_concepts(
                concepts=concepts,
                fitness_scores=fitness_scores,
                num_variants=args.n_variants,
                states=all_states  
            )
        
        fitness_scores = concept_generator.evaluate_concept_fitness(
            concepts, all_states, all_actions, 
            use_avg_similarity=args.use_avg_similarity, 
            use_state_consistency=args.use_state_consistency, 
            use_action_consistency=args.use_action_consistency
        )
        
        for concept_id, variants in concepts.items():
            if variants:  
                concept_history[concept_id].append(variants[0])
            else:
                concept_history[concept_id].append("无有效描述")
            fitness_history[concept_id][i+1] = fitness_scores.get(concept_id, 0)
        
        save_concepts_to_csv(concepts, fitness_scores, i+1, csv_file)
        
        print(f"第 {i+1} 次演化后的适应度分数:")
        for concept_id, score in fitness_scores.items():
            print(f"{concept_id}: {score:.4f}")
    
    history_chart_path = os.path.join(save_dir, f"{args.domain_name}_{args.task_name}_evolution_history.png")
    visualize_evolution(concept_history, fitness_history, history_chart_path)
    
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
    
    print("\n演示完成!")

if __name__ == "__main__":
    main()

"""

# GPT-2 模型
CUDA_VISIBLE_DEVICES=6 python concept_generator.py \
    --domain_name walker \
    --task_name walk \
    --seed 42 \
    --rl_model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --algorithm_name svea \
    --num_episodes 2 \
    --steps_per_episode 1000 \
    --n_concepts 3 \
    --n_variants 3 \
    --evolution_iterations 3 \
    --use_avg_similarity True \
    --use_state_consistency False \
    --use_action_consistency False \
    --clip_threshold 0.5 \
    --temperature 0.8 \
    --num_candidates 5 \
    --semantic_score_type raw \
    --keywords_filter True \
    --save_concepts \
    --gpt2_max_length 200 \
    --gpt2_top_k 20 \
    --gpt2_top_p 0.9 \
    --gpt2_repetition_penalty 1.2 \
    --model_name gpt2-medium

GPT-2 模型
CUDA_VISIBLE_DEVICES=6 python concept_generator.py \
    --domain_name walker \
    --task_name walk \
    --seed 42 \
    --rl_model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --algorithm_name svea \
    --model_name gpt2-medium \
    --gpt2_max_length 200

Llama-2 模型
CUDA_VISIBLE_DEVICES=3 python concept_generator.py \
    --domain_name walker \
    --task_name walk \
    --seed 42 \
    --rl_model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --algorithm_name svea \
    --model_name llama-2-7b-hf \
    --language_model_path "/mnt/lustre/GPU4/home/wuhanpeng/MI-with-Finetuned-LM/phase1_finetuning/pretrained_model/Llama-2-7b-hf/" \
    --gpt2_max_length 200 \
    --n_concepts 4 \
    --n_variants 2 \
    --evolution_iterations 2 \
    --use_keyword_guided_generation True \
    --keyword_groups_focus "机器人类型" "关节结构" "运动特征" \
    --min_keywords 2 \
    --max_keywords 4

# 使用LLaVA模型
CUDA_VISIBLE_DEVICES=3 python concept_generator.py \
    --domain_name walker \
    --task_name walk \
    --seed 42 \
    --rl_model_path "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/videos/walker_walk/svea/42/20250226_103416/model/400000.pt" \
    --algorithm_name svea \
    --model_name llava-1.5-7b \
    --gpt2_max_length 200 \
    --n_concepts 4 \
    --n_variants 2 \
    --evolution_iterations 2
    --states_per_episode 10

GPT-4V (Vision)
Claude 3 Opus/Sonnet
Gemini
LLaVA

"""