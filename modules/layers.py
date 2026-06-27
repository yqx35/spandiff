# -*- coding: utf-8 -*-
from torch import nn as nn
import torch
from torch.nn import functional as F

class EntityBoundaryPredictor(nn.Module):
    def __init__(self, config, prop_drop=0.1):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 用于将 Token 和 Entity 映射到同一空间
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # 最终的边界打分层
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)

    def forward(self, token_embedding, entity_embedding, token_mask):
        """
        Args:
            token_embedding: (Batch, Token_Seq_Len, Hidden)
            entity_embedding: (Batch, Entity_Count, Hidden)
            token_mask: (Batch, Token_Seq_Len)
        Returns:
            entity_token_p: (Batch, Entity_Count, Token_Seq_Len) - 概率矩阵
        """
        # 1. 特征融合 (Broadcasting)
        # token: (B, 1, T, H)
        # entity: (B, N, 1, H)
        # sum: (B, N, T, H)
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + \
                              self.entity_embedding_linear(entity_embedding).unsqueeze(2)

        # 2. 预测分数
        # (B, N, T, 1) -> (B, N, T)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)

        # 3. Mask 处理
        # 将 mask 扩展到与分数矩阵相同的形状
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        # 将无效位置的分数设为极小值
        entity_token_cls[~token_mask] = -1e25

        # 4. 计算概率
        # 使用 Sigmoid (也可以根据需求改为 Softmax)
        entity_token_p = F.sigmoid(entity_token_cls)
        
        return entity_token_p


class EntityTypePredictor(nn.Module):
    def __init__(self, config, entity_type_count, h_size=None):
        super().__init__()
        # 灵活处理 hidden_size 配置
        if h_size:
            self.hidden_size = h_size
        else:
            self.hidden_size = config.hidden_size

        self.linnear = nn.Linear(self.hidden_size, self.hidden_size)

        # Multi-Head Attention 用于捕捉实体与 Token 之间的依赖
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            dropout=config.hidden_dropout_prob
        )

        # 分类器：输入维度是 3倍 hidden_size (实体特征 + 左边界特征 + 右边界特征)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 3, entity_type_count)
        )

    def forward(self, h_entity, h_token, p_left, p_right, token_mask):
        """
        Args:
            h_entity: (Batch, Entity_Count, Hidden) - 实体嵌入
            h_token: (Batch, Token_Seq_Len, Hidden) - Token 嵌入
            p_left: (Batch, Entity_Count, Token_Seq_Len) - 左边界概率分布
            p_right: (Batch, Entity_Count, Token_Seq_Len) - 右边界概率分布
            token_mask: (Batch, Token_Seq_Len)
        """
        # 1. 实体特征投影
        h_entity = self.linnear(torch.relu(h_entity))

        # 2. Multi-Head Attention 增强
        # PyTorch 的 MultiheadAttention 默认输入形状为 (Seq_Len, Batch, Dim)
        query = h_entity.transpose(0, 1).clone() # (N, B, H)
        key = h_token.transpose(0, 1)            # (T, B, H)
        value = h_token.transpose(0, 1)          # (T, B, H)

        # key_padding_mask 需要是 (Batch, Seq_Len)
        attn_output, _ = self.multihead_attn(
            query, 
            key, 
            value, 
            key_padding_mask=~token_mask
        )
        
        # 残差连接 + 转回 (Batch, N, H)
        attn_output = attn_output.transpose(0, 1)
        h_entity = h_entity + attn_output # In-place add 可能会影响梯度，建议显式赋值

        # 3. 边界特征聚合 (Boundary-Aware Aggregation)
        # 利用 p_left/p_right 作为权重，对 Token Embedding 进行加权求和
        # (B, N, T) x (B, T, H) -> (B, N, H)
        # 
        left_token = torch.matmul(p_left, h_token)
        right_token = torch.matmul(p_right, h_token)

        # 4. 特征拼接与分类
        # 拼接：[增强后的实体特征, 左边界上下文, 右边界上下文]
        h_entity_combined = torch.cat([h_entity, left_token, right_token], dim=-1)

        # 最终分类 logits
        entity_logits = self.classifier(h_entity_combined)

        return entity_logits