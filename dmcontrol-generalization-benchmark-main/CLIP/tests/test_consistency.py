import os
import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src')
import pytest
import torch
import clip
from PIL import Image
import numpy as np
import gc

def test_consistency():
    print("\n开始ViT-L/14模型一致性测试...")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 检查模型文件是否存在
    model_path = "/mnt/lustre/GPU4/home/wuhanpeng/.cache/clip/ViT-L-14.pt"
    if os.path.exists(model_path):
        print(f"找到模型文件: {model_path}")
        print(f"文件大小: {os.path.getsize(model_path)/1024**2:.2f}MB")
    else:
        print(f"警告: 模型文件不存在于路径 {model_path}")
        print("将尝试从OpenAI服务器下载模型")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        if device == "cuda":
            print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB")
            print(f"可用GPU内存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**2:.2f}MB")
        
        # 加载ViT-L/14模型
        print("正在加载ViT-L/14模型...")
        model, preprocess = clip.load("ViT-L/14", device=device)
        print("ViT-L/14模型加载成功!")
        
        if device == "cuda":
            print(f"模型加载后GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        # 测试文本编码一致性
        print("测试文本编码一致性...")
        text = clip.tokenize(["a test"]).to(device)
        with torch.no_grad():  # no_grad()减少内存使用
            text_features1 = model.encode_text(text)
            text_features2 = model.encode_text(text)
        text_diff = (text_features1 - text_features2).abs().max().item()
        print(f"文本编码一致性差异: {text_diff}")
        assert text_diff == 0
        
        # 测试图像编码一致性
        print("测试图像编码一致性...")
        image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            image_features1 = model.encode_image(image)
            image_features2 = model.encode_image(image)
        image_diff = (image_features1 - image_features2).abs().max().item()
        print(f"图像编码一致性差异: {image_diff}")
        assert image_diff == 0
        
        # 打印模型信息
        print("\n模型信息:")
        print(f"模型类型: {type(model).__name__}")
        print(f"视觉模型: {model.visual.__class__.__name__}")
        
        # 验证是否为ViT-L/14
        if hasattr(model.visual, 'input_resolution'):
            print(f"输入分辨率: {model.visual.input_resolution}px")
        if hasattr(model.visual, 'patch_size'):
            print(f"Patch大小: {model.visual.patch_size}px")
        if hasattr(model.visual, 'width'):
            print(f"模型宽度: {model.visual.width}")
        if hasattr(model.visual, 'layers'):
            print(f"Transformer层数: {len(model.visual.layers)}")
        
        print("\n所有测试通过! ViT-L/14模型工作正常!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU内存不足！建议：")
            print("1. 关闭其他使用GPU的程序")
            print("2. 使用较小的批次大小")
            print("3. 考虑使用CPU进行测试")
            if device == "cuda":
                print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        raise e
    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    test_consistency()



'''

CUDA_VISIBLE_DEVICES=6 python /mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP/tests/test_consistency.py

'''