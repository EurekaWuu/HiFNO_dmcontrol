import torch
import torch.nn as nn
import torch.nn.functional as F

class BisimulationLoss(nn.Module):
    def __init__(self, lambda_SC=0.5, lambda_clip=0.5, p=2):
        super().__init__()
        self.lambda_SC = lambda_SC
        self.lambda_clip = lambda_clip
        self.p = p

    def compute_semantic_class_loss(self, state_features, actions, class_labels):

        batch_size = state_features.size(0)
        unique_classes = torch.unique(class_labels)
        
        feature_consistency_loss = 0.0
        action_consistency_loss = 0.0
        
        for cls in unique_classes:
            # 获取该类别的所有样本索引
            indices = (class_labels == cls).nonzero(as_tuple=True)[0]
            
            if len(indices) <= 1:
                continue  # 跳过只有一个样本的类别
                
            cls_features = state_features[indices]
            cls_actions = actions[indices]
            
            mean_feature = cls_features.mean(dim=0, keepdim=True)
            
            mean_action = cls_actions.mean(dim=0, keepdim=True)
            
            # 计算每个样本与类平均值的距离
            feature_dists = F.mse_loss(cls_features, mean_feature.expand_as(cls_features))
            action_dists = F.mse_loss(cls_actions, mean_action.expand_as(cls_actions))
            
            feature_consistency_loss += feature_dists
            action_consistency_loss += action_dists
        
        # 如果没有有效类别，返回零损失
        if len(unique_classes) == 0:
            return torch.tensor(0.0, device=state_features.device)
            
        # 归一化损失
        feature_consistency_loss /= len(unique_classes)
        action_consistency_loss /= len(unique_classes)
        
        sc_loss = feature_consistency_loss + action_consistency_loss
        
        return sc_loss
    
    def compute_clip_guided_bisim_loss(self, state_features, actions, clip_similarities, clip_class_indices=None):

        batch_size = state_features.size(0)
        
        # 状态特征之间的成对距离矩阵
        state_dists = torch.cdist(state_features, state_features, p=self.p)  # [batch, batch]
        
        # 动作之间的成对距离矩阵
        action_dists = torch.cdist(actions, actions, p=self.p)  # [batch, batch]
        
        # CLIP相似度之间的成对相似度矩阵
        # 将相似度归一化
        normalized_sims = F.normalize(clip_similarities, p=2, dim=1)
        clip_sims = torch.mm(normalized_sims, normalized_sims.t())  # [batch, batch]
        
        # 相似度转换为距离 (1 - 相似度)
        clip_dists = 1.0 - clip_sims
        
        # 状态特征距离与CLIP距离之间的一致性损失
        state_clip_loss = F.mse_loss(state_dists, clip_dists)
        
        # 动作距离与CLIP距离之间的一致性损失
        # 相似状态应有相似动作，动作距离应与CLIP距离成正比
        action_clip_loss = F.mse_loss(action_dists, clip_dists)
        
        clip_bisim_loss = state_clip_loss + action_clip_loss
        
        return clip_bisim_loss
    
    def compute_total_loss(self, state_features, actions, clip_similarities, class_labels):
        # 语义类内一致性损失
        sc_loss = self.compute_semantic_class_loss(state_features, actions, class_labels)
        
        # CLIP引导的双模拟损失
        clip_bisim_loss = self.compute_clip_guided_bisim_loss(state_features, actions, clip_similarities)

        total_loss = self.lambda_SC * sc_loss + self.lambda_clip * clip_bisim_loss

        return total_loss, (sc_loss, clip_bisim_loss)
