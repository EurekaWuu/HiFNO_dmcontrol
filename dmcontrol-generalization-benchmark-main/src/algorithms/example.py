import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules as m
import augmentations
from algorithms.sac import SAC
from bisimulation_loss import BisimulationLoss

from algorithms.modules import Actor, Critic
from algorithms.models.HiFNO import HierarchicalFNO, PositionalEncoding, MultiScaleConv, PatchExtractor, Mlp, ConvResFourierLayer, SelfAttention, CrossAttentionBlock, TimeAggregator


class HiFNOAgent(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HiFNO Encoder模型
        self.encoder = HiFNOEncoder(obs_shape, args.embed_dim, args).to(self.device)
        self.actor = Actor(self.encoder, action_shape, args.hidden_dim).to(self.device)
        self.critic = Critic(self.encoder, action_shape, args.hidden_dim).to(self.device)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        if step % self.args.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        # 计算critic_loss
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.args.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        return critic_loss

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            action = mu.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            return action


def predict_image(image_path, candidate_descriptions, device="cuda"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(candidate_descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    results = [(desc, conf.item()) for desc, conf in zip(candidate_descriptions, similarity[0])]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def bisimulation_loss(image_features, text_features, action_features, action_target, margin=0.1):
    """
    双模拟损失函数：确保相似状态的动作相似。
    :param image_features: 当前状态的图像特征
    :param text_features: 当前状态对应的文本特征
    :param action_features: 当前状态的动作特征
    :param action_target: 目标动作特征
    :param margin: 损失的容忍度（控制相似状态的行为差异）
    :return: 双模拟损失
    """
    # 计算图像与文本之间的相似度
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # 计算图像与目标动作之间的相似度
    action_similarity = torch.cosine_similarity(action_features, action_target)

    # 计算双模拟损失：状态相似度与动作相似度之间的差异
    bisim_loss = torch.abs(similarity - action_similarity).mean()

    return bisim_loss


class HiFNOEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, args):
        super().__init__()
        self.out_dim = args.hidden_dim
        self.device = torch.device('cuda')

        self.hifno = HierarchicalFNO(
            img_size=(obs_shape[1], obs_shape[2]),
            patch_size=4,
            in_channels=obs_shape[0],
            out_channels=feature_dim,  # feature_dim作为输出维度
            embed_dim=args.embed_dim,
            depth=args.depth if hasattr(args, 'depth') else 2,
            num_scales=args.num_scales,
            truncation_sizes=[16, 12, 8],
            num_heads=4,
            mlp_ratio=2.0,
            activation='gelu'
        )

    def forward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        features = self.hifno(obs)  # 提取图像特征
        return features


if __name__ == "__main__":
    # 设定描述文本
    descriptions = [
        "The mannequin’s torso is upright with a slight forward lean, while one leg is extended behind for maximum push-off, and the other leg is lifting towards the front for the next step.",
        "In this stride, the mannequin's torso remains straight and stable, with one leg bent at the knee in mid-air while the other leg drives forward, ensuring efficient movement and balance.",
        "With its torso aligned, the mannequin is lifting one leg in front of the body while the other leg extends backward, demonstrating an efficient running posture with proper leg extension and cadence.",
        "The mannequin maintains an upright posture, with its knees bent and one leg pushing off the ground while the other leg moves toward the front, ensuring smooth, rhythmic strides.",
        "The torso is held firm and steady, while one leg is fully extended behind, and the other leg is bent, bringing the knee high to prepare for the next stride in a fluid running motion.",
        "As the mannequin runs, its torso stays balanced and straight, with alternating leg movements—one leg propels the body forward while the other leg swings back to maximize speed and efficiency."
    ]

    # 图像路径
    image_path = "path_to_image.jpg"  # 替换为DMControl Walk任务中的图像路径

    # 获取描述预测结果
    results = predict_image(image_path, descriptions)
    print("\n预测结果及置信度:")
    for description, confidence in results:
        print(f"{description:<20}: {confidence:.2f}%")

    # 初始化 HiFNO 代理
    agent = HiFNOAgent(
        obs_shape=(3, 64, 64),  # 假设输入的图像大小
        action_shape=(2,),  # 假设动作空间大小为2
        args={"embed_dim": 64, "hidden_dim": 128, "frame_stack": 4, "actor_update_freq": 2}
    )

    # 获取当前状态的图像特征和动作特征
    image_features = agent.encoder(image_path)
    action_features = torch.randn(1, 64)  # 假设的动作特征
    action_target = torch.randn(1, 64)  # 假设的目标动作特征

    # 计算双模拟损失
    bisim_loss_value = bisimulation_loss(image_features, image_features, action_features, action_target)
    print(f"Bisimulation Loss: {bisim_loss_value:.4f}")

    # 选择动作
    agent.select_action(np.random.rand(3, 64, 64))
