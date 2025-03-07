import os
import pytest
import torch
import clip
from PIL import Image
import numpy as np
import gc

def test_consistency():
    print("\n开始一致性测试...")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        

        if device == "cuda":
            print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("模型加载成功")
        
        if device == "cuda":
            print(f"模型加载后GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        

        text = clip.tokenize(["a test"]).to(device)
        with torch.no_grad():  # no_grad()减少内存使用
            text_features1 = model.encode_text(text)
            text_features2 = model.encode_text(text)
        text_diff = (text_features1 - text_features2).abs().max().item()
        print(f"文本编码一致性差异: {text_diff}")
        assert text_diff == 0
        

        image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            image_features1 = model.encode_image(image)
            image_features2 = model.encode_image(image)
        image_diff = (image_features1 - image_features2).abs().max().item()
        print(f"图像编码一致性差异: {image_diff}")
        assert image_diff == 0
        
        print("所有测试通过!")
        
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    test_consistency()



'''

CUDA_VISIBLE_DEVICES=6 python /mnt/lustre/GPU4/home/wuhanpeng/RL-ViGen/CLIP/tests/test_consistency.py

'''