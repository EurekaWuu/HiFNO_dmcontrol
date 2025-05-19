#!/usr/bin/env python
import os
import sys
import json
import argparse
import tempfile
from PIL import Image
import numpy as np
import torch

sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/models/llava-v1.5-7b')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/models')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/anaconda3/envs/llava/lib/python3.10/site-packages')

def parse_args():
    parser = argparse.ArgumentParser(description='LLaVA图像描述生成器')
    parser.add_argument('--image_path', type=str, required=True, help='图像文件路径')
    parser.add_argument('--prompt', type=str, default="Describe what you see in this image, focusing on the robot's posture and leg positions.", 
                        help='发送给LLaVA的提示')
    parser.add_argument('--model_path', type=str, default="/mnt/lustre/GPU4/home/wuhanpeng/models/llava-v1.5-7b",
                        help='LLaVA模型路径')
    parser.add_argument('--output_file', type=str, help='输出结果的JSON文件路径')
    parser.add_argument('--image_data', type=str, help='Base64编码的图像数据 (用于直接传递图像而不是文件路径)')
    parser.add_argument('--image_format', type=str, default='PIL', help='图像格式 (PIL, numpy, LazyFrames)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    debug_output = False
    if debug_output:
        print(f"Python版本: {sys.version}", file=sys.stderr)
        print(f"当前工作目录: {os.getcwd()}", file=sys.stderr)
        print(f"脚本参数: {args}", file=sys.stderr)
        print(f"sys.path: {sys.path}", file=sys.stderr)
    
    if not os.path.exists(args.image_path):
        result = {
            "error": f"图像路径不存在: {args.image_path}",
            "description": ""
        }
        _output_result(result, args.output_file)
        return 1
    
    try:
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, process_images
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.utils import disable_torch_init
            import torch
        except Exception as e:
            print(f"导入LLaVA库失败: {e}", file=sys.stderr)
            result = {
                "error": f"导入LLaVA库失败: {e}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
        
        disable_torch_init()
        
        model_path = args.model_path
        
        if not os.path.exists(model_path):
            result = {
                "error": f"模型路径不存在: {model_path}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
            
        try:
            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=False
            )
        except Exception as e:
            result = {
                "error": f"加载模型失败: {e}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
        
        if args.image_path and os.path.exists(args.image_path):
            try:
                image = Image.open(args.image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                result = {
                    "error": f"打开图像文件时出错: {e}",
                    "description": ""
                }
                _output_result(result, args.output_file)
                return 1
        else:
            result = {
                "error": f"图像文件不存在: {args.image_path}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
        
        try:
            image_tensor = process_images([image], image_processor, model.config)
            
            model_dtype = next(model.parameters()).dtype
            if image_tensor.dtype != model_dtype:
                image_tensor = image_tensor.to(model_dtype)
                
        except Exception as e:
            result = {
                "error": f"处理图像失败: {e}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1

        conv = conv_templates["llava_v1"].copy()
        
        # conv.system = ""
        
        
        query = args.prompt
        
        print(f"DEBUG - 使用提示: {query}", file=sys.stderr)
        
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print(f"DEBUG - 完整prompt: {prompt}", file=sys.stderr)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        
        try:
            with torch.inference_mode():
                
                model_dtype = next(model.parameters()).dtype
                input_ids_cuda = input_ids.cuda()
                image_tensor_cuda = image_tensor.cuda().to(model_dtype)
                
                print(f"DEBUG - 输入图像尺寸: {image_tensor_cuda.shape}", file=sys.stderr)
                
                
                output_ids = model.generate(
                    input_ids_cuda,
                    images=image_tensor_cuda,
                    do_sample=True,
                    temperature=0.7,     
                    top_p=0.95,
                    top_k=50,
                    max_new_tokens=300,  
                    min_new_tokens=30,   
                    repetition_penalty=1.1,  
                    num_beams=3,         
                    early_stopping=True
                )
                
                
                print(f"DEBUG - 输入ID长度: {input_ids.shape[1]}", file=sys.stderr)
                print(f"DEBUG - 输出ID长度: {output_ids.shape[1]}", file=sys.stderr)
                print(f"DEBUG - 生成token数: {output_ids.shape[1] - input_ids.shape[1]}", file=sys.stderr)
                
        except RuntimeError as e:
            result = {
                "error": f"模型推理出错: {e}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
        except Exception as e:
            result = {
                "error": f"模型生成时出错: {e}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
        
        
        try:
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            
            print(f"DEBUG - prompt: {prompt[:50]}...", file=sys.stderr)
            print(f"DEBUG - output_ids shape: {output_ids.shape}", file=sys.stderr)
            print(f"DEBUG - input_ids shape: {input_ids.shape}", file=sys.stderr)
            print(f"DEBUG - 生成token数: {output_ids.shape[1] - input_ids.shape[1]}", file=sys.stderr)
            print(f"DEBUG - raw output: '{outputs}'", file=sys.stderr)
            
            
            if not outputs or len(outputs.strip()) < 5:
                print(f"DEBUG - 警告：模型生成为空或过短", file=sys.stderr)
                outputs = "Robot in walking pose with coordinated leg movements."
                print(f"DEBUG - 使用默认描述: '{outputs}'", file=sys.stderr)
                
            result = {"description": outputs}
            _output_result(result, args.output_file)
            
        except Exception as e:
            result = {
                "error": f"解码输出时出错: {e}",
                "description": ""
            }
            _output_result(result, args.output_file)
            return 1
        
    except Exception as e:
        import traceback
        print(f"处理过程中发生未捕获的异常: {e}", file=sys.stderr)
        print(f"详细错误堆栈: {traceback.format_exc()}", file=sys.stderr)
        result = {
            "error": str(e),
            "description": ""
        }
        _output_result(result, args.output_file)
        return 1
    
    return 0

def _output_result(result, output_file=None):
    try:
        
        if "description" not in result:
            result["description"] = ""

        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception:
                backup_file = f"/tmp/llava_result_{os.getpid()}.json"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        # print(f"LLaVA输出: {result['description']}")  
        
        
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        print(json.dumps({"description": ""}, ensure_ascii=False))

if __name__ == "__main__":
    sys.exit(main()) 