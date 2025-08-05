# HiFNO-DMControl基于分层傅里叶神经算子的连续控制视觉泛化

## 项目概述

该项目基于DeepMind的DMControl套件，集成了多种最先进的强化学习算法，提出了**分层傅里叶神经算子（HiFNO）**模型来解决视觉泛化。

### 特点

- **HiFNO模型**: 基于傅里叶神经算子的分层特征学习架构
- **多算法**: 支持SVEA、SODA、PAD、DrQ、RAD、CURL、SAC等算法
- **视觉泛化**: 专门设计用于处理颜色变化和视频背景干扰
- **CLIP集成**: 利用CLIP进行视觉概念生成和理解
- **多GPU支持**: 支持分布式训练加速
- **双重仿真损失**: 实现更稳定的表征学习

## 项目架构

```
HiFNO_dmcontrol/
├── dmcontrol-generalization-benchmark-main/
│   ├── src/                          # 核心源代码
│   │   ├── algorithms/               # 强化学习算法实现
│   │   │   ├── models/              # 模型架构
│   │   │   │   ├── HiFNO_1.py      # HiFNO核心模型
│   │   │   │   ├── HiFNO_multigpu.py # 多GPU版本
│   │   │   │   └── ...
│   │   │   ├── hifno.py             # HiFNO算法主体
│   │   │   ├── hifno_bisim.py       # 带双重仿真损失的HiFNO
│   │   │   ├── drqv2_official.py    # DrQv2算法
│   │   │   └── ...                  # 其他算法
│   │   ├── env/                     # 环境相关
│   │   │   ├── wrappers.py          # 环境包装器
│   │   │   ├── distracting_control/ # 干扰控制环境
│   │   │   └── data/               # 测试数据
│   │   ├── train.py                 # 训练脚本
│   │   ├── eval.py                  # 评估脚本
│   │   └── arguments.py             # 参数配置
│   ├── CLIP/                        # CLIP模型集成
│   │   └── tests/                   # 概念生成工具
│   ├── scripts/                     # 训练脚本
│   └── setup/                       # 环境配置
└── README.md                        # 本文件
```

## 核心技术

### HiFNO (分层傅里叶神经算子)

- **多尺度卷积**: 捕获不同空间尺度的特征
- **傅里叶变换**: 在频域处理全局依赖关系
- **位置编码**: 保持空间结构信息
- **时间聚合器**: 处理时序信息
- **交叉注意力机制**: 实现多模态信息融合

### 支持的算法

| 算法 | 描述 | 论文链接 |
|------|------|----------|
| **HiFNO** | 分层傅里叶神经算子（本项目创新） | - |
| SVEA | 数据增强下的稳定深度Q学习 | [arXiv:2107.00644](https://arxiv.org/abs/2107.00644) |
| SODA | 软数据增强的强化学习泛化 | [arXiv:2011.13389](https://arxiv.org/abs/2011.13389) |
| DrQv2 | 改进的数据正则化Q学习 | [arXiv:2107.09645](https://arxiv.org/abs/2107.09645) |
| CURL | 对比式无监督表征学习 | [arXiv:2004.04136](https://arxiv.org/abs/2004.04136) |
| PAD | 部署时策略适应 | [arXiv:2007.04309](https://arxiv.org/abs/2007.04309) |

### 测试环境

项目提供多种视觉泛化测试环境：

#### 颜色泛化
- **color_easy**: 轻微的颜色变化
- **color_hard**: 大幅度的颜色随机化

#### 视频背景泛化  
- **video_easy**: 简单的视频背景干扰
- **video_hard**: 复杂的视频背景变化

#### 干扰控制套件
- **distracting_cs**: 8个不同强度级别的环境干扰

## 开始

### 环境配置

1. **创建Conda环境**
```bash
conda env create -f setup/conda.yaml
conda activate dmcgb
```

2. **安装依赖**
```bash
sh setup/install_envs.sh
```

3. **配置数据集路径**
```bash
# 编辑 setup/config.cfg 添加数据集路径
```

### 基础训练

```bash
# 使用HiFNO算法训练
python src/train.py \
  --algorithm hifno \
  --domain_name walker \
  --task_name walk \
  --seed 0 \
  --eval_mode color_hard

# 使用HiFNO双重仿真版本
python src/train.py \
  --algorithm hifno_bisim_1 \
  --domain_name walker \
  --task_name walk \
  --seed 0 \
  --eval_mode video_easy
```

### 多GPU训练

```bash
# HiFNO多GPU训练
python src/train.py \
  --algorithm hifno_multigpu \
  --domain_name walker \
  --task_name walk \
  --seed 0 \
  --num_gpus 4
```

### 批量实验

```bash
# 运行预配置的实验
bash src/run_experiments.sh
```

## 配置参数

### HiFNO专用参数

```python
# 模型架构参数
--embed_dim 64          # 嵌入维度
--depth 2               # 网络深度  
--patch_size 4          # 图像块大小
--num_scales 3          # 多尺度层数

# 双重仿真损失参数
--lambda_BB 0.8         # 基础双重仿真损失权重
--lambda_ICC 0.4        # 上下文一致性损失权重  
--lambda_CC 0.4         # 交叉一致性损失权重
```

### 训练参数

```python
--algorithm hifno       # 选择算法
--domain_name walker    # 环境域名
--task_name walk        # 任务名称
--eval_mode color_hard  # 评估模式
--train_steps 500k      # 训练步数
--batch_size 128        # 批大小
--seed 0               # 随机种子
```

## 高级功能

### CLIP概念生成

```python
# 使用CLIP进行视觉概念分析
from CLIP.tests.concept_generator import ConceptGenerator

generator = ConceptGenerator(device='cuda')
concepts = generator.generate_concepts_from_states(states)
```

### 自定义环境

```python
# 创建自定义测试环境
env = make_env(
    domain_name='cartpole',
    task_name='swingup',
    mode='color_hard',
    intensity=0.8
)
```

### 模型可视化

```python
# 特征可视化
python src/visualize.py \
  --model_path logs/walker_walk/hifno/0/model.pt \
  --env_mode color_hard
```

## 性能优化

### 多GPU配置

```bash
# 设置GPU数量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 分布式训练
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  src/train.py \
  --algorithm hifno_multigpu
```

### 内存优化

```python
# 启用梯度检查点
--gradient_checkpointing True

# 混合精度训练  
--use_amp True

# 优化批大小
--batch_size 256
```

## 开发

### 添加新算法

1. 在`src/algorithms/`创建算法文件
2. 继承基础类并实现必要方法
3. 在`factory.py`中注册算法
4. 更新参数配置

### 自定义模型

```python
class CustomModel(nn.Module):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__()
        # 模型定义
        
    def forward(self, x):
        # 前向传播
        return x
```

### 添加新环境

```python
def make_custom_env(**kwargs):
    # 环境创建逻辑
    return env
```