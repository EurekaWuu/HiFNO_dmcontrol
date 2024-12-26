import torch
import torch.nn as nn
import torch.nn.functional as F

class BisimulationLoss(nn.Module):
    def __init__(self, lambda_BB=1.0, lambda_ICC=0.1, lambda_CC=0.1, p=2, check_inputs=False):
        """
        双模拟损失模块，用于对表示函数进行约束，确保满足三项原则：
        1. 基础双模拟 (Base Bisimulation, BB)
        2. 上下文内一致性 (Inter-context Consistency, ICC)
        3. 跨上下文一致性 (Cross Consistency, CC)

        参数:
        - lambda_BB: BB项的权重
        - lambda_ICC: ICC项的权重
        - lambda_CC: CC项的权重
        - p: 距离范数 (默认p=2为L2距离，p=1为L1距离)
        - check_inputs: 是否对输入张量进行形状和维度检查（调试用）

        使用场景:
        通常在您的训练循环中对不同上下文下的状态表示进行约束时使用。通过保证相同状态在不同上下文下的表示相近（ICC）、
        不同状态的相对关系在不同上下文下保持一致（CC），以及表示距离与真实bisimulation度量尽可能吻合（BB），
        实现更鲁棒、上下文无关的表示空间。

        假设前提：
        - 输入的特征表示 (phi_s_i_theta1 等) 已经通过 HiFNOEncoder 得到，
          HiFNOEncoder会对输入的高维观测进行多步处理，最终输出形如 (B, hidden_dim) 的特征向量。
        - d_sij 为真实状态对 (s_i, s_j) 的bisimulation度量目标值，需由使用方提供。

        HiFNOEncoder参考:
        - HierarchicalFNO：对输入 (B, C, D1, D2, ...) 保持空间维度不变，仅改变通道数并提取特征
        - HiFNOEncoder：在HierarchicalFNO输出的基础上进一步通过聚合和映射，将多维特征压缩至 (B, hidden_dim)
          的固定维度表示，用于下游策略网络或表示学习任务。
        """
        super().__init__()
        self.lambda_BB = lambda_BB
        self.lambda_ICC = lambda_ICC
        self.lambda_CC = lambda_CC
        self.p = p
        self.check_inputs = check_inputs

    def forward(self, phi_si_theta1, phi_sj_theta1, phi_si_theta2, phi_sj_theta2, d_sij):
        """
        计算双模拟损失并返回 total_loss 及各项子损失。

        参数:
        - phi_si_theta1: 张量 [batch, rep_dim]，状态s_i在上下文theta1下的表示
        -  phi_sj_theta1: 张量 [batch, rep_dim]，状态s_j在上下文theta1下的表示
        - phi_si_theta2: 张量 [batch, rep_dim]，状态s_i在上下文theta2下的表示
        - phi_sj_theta2: 张量 [batch, rep_dim]，状态s_j在上下文theta2下的表示
        - d_sij: 张量 [batch]，表示真实MDP中(s_i, s_j)的目标bisimulation距离（由用户事先计算或提供）

        返回:
        - total_loss: 标量，总损失 = lambda_BB*L_BB + lambda_ICC*L_ICC + lambda_CC*L_CC
        - (L_BB, L_ICC, L_CC): 各项子损失，便于日志记录和调试。

        注释:
        1. 基础双模拟 (BB): 保证在相同上下文下的表示距离与真实bisimulation距离匹配。
           L_BB = MSE(d_Y_theta1, d_sij)
        2. 上下文内一致性 (ICC): 对同一状态在不同上下文下的表示应相近，从而抵抗上下文变化的影响。
           L_ICC = mean(d_Y(phi_s_i_theta1, phi_si_theta2))
        3. 跨上下文一致性 (CC): 保证两个状态在不同上下文下的相对关系保持一致。
           L_CC = L1(d_Y_theta1, d_Y_theta2)
        """

        # 可选检查输入是否符合预期形状和批大小
        if self.check_inputs:
            batch_size = phi_si_theta1.size(0)
            assert phi_si_theta1.dim() == 2 and phi_sj_theta1.dim() == 2 \
                   and phi_si_theta2.dim() == 2 and phi_sj_theta2.dim() == 2, \
                   "phi_*_theta* 输入必须为二维张量[B, rep_dim]"
            assert d_sij.dim() == 1 and d_sij.size(0) == batch_size, \
                   "d_sij必须为[B]的一维张量，并与表示张量的batch维度匹配"
            assert phi_sj_theta1.size(0) == batch_size \
                   and phi_si_theta2.size(0) == batch_size \
                   and phi_sj_theta2.size(0) == batch_size, \
                   "所有表示张量的batch维度必须一致"

        # 计算表示空间中的距离
        dist_fn = lambda x, y: torch.norm(x - y, p=self.p, dim=1)
        d_y_theta1 = dist_fn(phi_si_theta1, phi_sj_theta1)
        d_y_theta2 = dist_fn(phi_si_theta2, phi_sj_theta2)

        # 计算三个损失项
        L_BB = F.mse_loss(d_y_theta1, d_sij)  # Base Bisimulation Loss
        L_ICC = F.mse_loss(phi_si_theta1, phi_si_theta2)  # Inter-context Consistency Loss
        L_CC = F.mse_loss(d_y_theta1, d_y_theta2)  # Cross Consistency Loss

        # 合并损失
        total_loss = self.lambda_BB * L_BB + self.lambda_ICC * L_ICC + self.lambda_CC * L_CC

        return total_loss, (L_BB, L_ICC, L_CC)
