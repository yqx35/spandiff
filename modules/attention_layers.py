# -*- coding: utf-8 -*-
import math
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from modules.posi tion_encoding_layer import DynamicPositionBias,XPOS,AlibiPositionalBias
from modules.utils import span_iou

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 定义线性变换
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # 定义归一化和 Dropout 层
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, key_padding_mask=None):
        B, Lq, Cq = q.size()
        _, Lk, Ck = k.size()
        _, Lv, Cv = v.size()

        # 线性变换
        q = self.q_linear(q).view(B, Lq, self.num_heads, -1).transpose(1, 2)  # B x H x Lq x D
        k = self.k_linear(k).view(B, Lk, self.num_heads, -1).transpose(1, 2)  # B x H x Lk x D
        v = self.v_linear(v).view(B, Lv, self.num_heads, -1).transpose(1, 2)  # B x H x Lv x D

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.embed_dim, dtype=torch.float32))  # B x H x Lq x Lk
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # B x 1 x 1 x Lk
            scores.masked_fill_(key_padding_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_output = torch.matmul(weights, v)  # B x H x Lq x D'

        # 处理多头注意力输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, -1)  # B x Lq x D
        output = self.out_linear(attn_output)
        output = self.dropout(output) + q.view(B, self.num_heads, Lq, -1).transpose(1, 2).contiguous().view(B, Lq, -1)
        output = self.layer_norm(output)

        return output

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout: float = 0.10,positional_embedding: str='dyn',bias: bool = True,temperature: float = 1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        ##
        self.temperature = temperature
        self.bias = bias
        self.positional_embedding = positional_embedding
        #
        if self.positional_embedding == "dyn":
            self.dynpos = DynamicPositionBias(dim=embed_dim // 4,
                                              heads=num_heads,
                                              depth=2)
        elif self.positional_embedding == "alibi":
            alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
            self.alibi = AlibiPositionalBias(alibi_heads,
                                             self.num_heads)
        elif self.positional_embedding == "xpos":
            self.xpos = XPOS(self.head_dim)

        # Dropout Layer
        self.dropout_layer = nn.Dropout(dropout)

        #
        self.weights = nn.Parameter(
            torch.empty(self.embed_dim, 3 * self.embed_dim)  # Q, K, V of equal sizes in given order
        )
        self.out_w = nn.Parameter(
            torch.empty(self.embed_dim, self.embed_dim)  # Q, K, V of equal sizes in given order
        )
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)
        #
        if self.bias:
            self.out_bias = nn.Parameter(
                torch.empty(1, 1, self.embed_dim)  # Q, K, V of equal sizes in given order
            )
            self.in_bias = nn.Parameter(
                torch.empty(1, 1, 3 * self.embed_dim)  # Q, K, V of equal sizes in given order
            )

            torch.nn.init.constant_(self.out_bias, 0.)
            torch.nn.init.constant_(self.in_bias, 0.)

    def construct_graph(self, ref_query,query,ref_span,span):
        # print('=========')
        # print(ref_span)
        # print('-----------------')
        # print(span)


        with torch.no_grad():
            bz, Lq = query.shape[0], query.shape[1]

            # print('2',query.shape)
            adj_matrix = torch.diag(torch.ones(Lq, device=query.device, dtype=torch.bool)).unsqueeze(0).repeat(bz, 1, 1)  # bz, Lq, Lq            # print(adj_matrix.shape)

            # iou = span_iou(ref_span, span)  # bz,Lq,Lq

            # adj_matrix[iou >= 0.2] = 0
            cos_sim = torch.cosine_similarity(ref_query.unsqueeze(2), query.unsqueeze(1), dim=-1)
            adj_matrix[cos_sim >= 0.7] = 1

            adj_matrix = ~adj_matrix

        return adj_matrix

    def forward(self, Q , K, V ,span,refer_query=None,refer_span=None,key_padding_mask=None):

        h_out=Q
        adj_matrix = self.construct_graph(refer_query,Q, refer_span,span)
        # print(adj_matrix.shape)
        adj_matrix = adj_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)#.flatten(0, 1)
        # print(adj_matrix.shape)

        b, l, h = Q.shape
        # print(Q.shape)
        Q = Q.view(b, l, self.num_heads, -1).permute(0, 2, 1, 3)
        K = K.view(b, l, self.num_heads, -1).permute(0, 2, 1, 3)
        V = V.view(b, l, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.positional_embedding == "xpos":
            Q, K = self.xpos(Q), self.xpos(K, downscale=True)

        norm = self.head_dim ** 0.5
        attention = (Q @ K.transpose(2, 3) / self.temperature / norm)

        if self.positional_embedding == "dyn":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.dynpos(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "alibi":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.alibi(i, j).unsqueeze(0)
            attention = attention + attn_bias

        # print('att',attention.shape)

        # if key_padding_mask is not None:
        #     attention = attention.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # print(key_padding_mask)
        # print(adj_matrix)
        # attention = attention.masked_fill(adj_matrix, float('-inf'))
        # print(attention)

        attention_probs = attention.softmax(dim=-1)  # b, a, l, l
        attention_output = torch.matmul(self.dropout_layer(attention_probs), V)
        output = attention_output.permute(0, 2, 1, 3).flatten(2, 3)
        output = h_out+output + self.out_bias
        # print(output)
        return output

if __name__ == "__main__":
    attn = MultiHeadSelfAttention(embed_dim=1024,num_heads=8,dropout=0.1,positional_embedding='xpos')
    x = torch.randn(8,60,1024)
    out = attn(x)
    print(out.shape)