# -*- coding: utf-8 -*-
import copy
import math
import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.no_grad()
def topk_mask_from_features(x: torch.Tensor, topk: int, ensure_self=True):
    """
    Returns attn_mask: [B, 1, L, L] bool (True=mask out)
    """
    B, L, _ = x.shape
    k = min(topk, L)

    x = F.normalize(x, p=2, dim=-1)
    sim = torch.matmul(x, x.transpose(1, 2))  # [B,L,L]

    if ensure_self:
        eye = torch.eye(L, device=x.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, 10.0)

    idx = torch.topk(sim, k=k, dim=-1, largest=True).indices  # [B,L,k]
    keep = torch.zeros((B, L, L), device=x.device, dtype=torch.bool)
    keep.scatter_(dim=-1, index=idx, value=True)

    if ensure_self:
        diag = torch.arange(L, device=x.device)
        keep[:, diag, diag] = True

    return (~keep).unsqueeze(1)  # [B,1,L,L]


def span_iou_and_dist_bias(spans: torch.Tensor, dist_sigma: float = 10.0):
    """
    Calculates positively bounded geometric bias (IoU + distance affinity)
    spans: [B, L, 2] (start, end)
    """
    s = spans[..., 0].float()
    e = spans[..., 1].float()
    s, e = torch.minimum(s, e), torch.maximum(s, e)

    len_ = (e - s + 1.0).clamp(min=1.0)
    c = (s + e) * 0.5

    s1, e1 = s.unsqueeze(2), e.unsqueeze(2)
    s2, e2 = s.unsqueeze(1), e.unsqueeze(1)

    #  1. 1D IoU [0, 1]
    inter = (torch.minimum(e1, e2) - torch.maximum(s1, s2) + 1.0).clamp(min=0.0)
    union = (len_.unsqueeze(2) + len_.unsqueeze(1) - inter).clamp(min=1e-6)
    iou = inter / union

    #2. Distance bias (0, 1]
    dist = (c.unsqueeze(2) - c.unsqueeze(1)).abs()
    dist_aff = torch.exp(-dist / max(dist_sigma, 1e-6))

    return iou + dist_aff  # 范围 [0, 2]


# =============================================================================
# 2. MultiHeadAttention
# =============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        B, L, _ = query.shape
        Lk = key.size(1)

        q = self.q_proj(query).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Lk), -1e4)

        attn = F.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# =============================================================================
# 3. RelationAttention (Semantic Top-k + Geometric Bias)
# =============================================================================
class RelationAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, topk=8, dist_sigma=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk = topk
        self.dist_sigma = dist_sigma

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.geo_gate = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, query, key=None, value=None, refer_Q=None, ref_span=None):
        if key is None: key = query
        if value is None: value = query
        B, L, _ = query.shape

        # 1) Top-k semantic graph construction
        graph_source = refer_Q if refer_Q is not None else query
        attn_mask = topk_mask_from_features(graph_source.detach(), self.topk)

        # 2)  Linear projections
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3) Geometric bias (Positively bounded reward)
        if ref_span is not None:
            bias = span_iou_and_dist_bias(ref_span.detach(), dist_sigma=self.dist_sigma)
            scores = scores + torch.sigmoid(self.geo_gate) * bias.unsqueeze(1)

        # 4) Masking and numerical protection
        scores = scores.masked_fill(attn_mask, -1e4)
        if scores.dtype == torch.float16:
            scores = scores.clamp(min=-1e4, max=1e4)

        attn = F.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out)


# =============================================================================
# 4. TransformerDecoderLayer 
# =============================================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, 
                 self_attn=True, cross_attn=True, topk=8):
        super().__init__()
        self.self_attn_bool = self_attn
        self.cross_attn_bool = cross_attn

        # Time-step injection MLP
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))

        # 1. Cross Attention
        if self.cross_attn_bool:
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # 2. Self Attention (RelationAttention)
        if self.self_attn_bool:
            self.self_attn = RelationAttention(d_model, n_heads, dropout=dropout, topk=topk)
            self.norm2 = nn.LayerNorm(d_model)

        # 3. FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, ref_tgt, ref_span, time_emb, src, src_mask, pos=None):
        # Time-step injection
        tgt = tgt + self.time_mlp(time_emb).unsqueeze(1)

        # 1. Cross Attention (从文本 Gather 特征)
        if self.cross_attn_bool:
            q = self.with_pos_embed(tgt, pos)
            k_padding_mask = ~src_mask.bool() if src_mask is not None else None
            tgt2 = self.cross_attn(q, src, src, key_padding_mask=k_padding_mask)
            tgt = self.norm1(tgt + tgt2)

        # 2. Self Attention (Relation Modeling)
        if self.self_attn_bool:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(query=q, key=k, value=tgt, refer_Q=ref_tgt, ref_span=ref_span)
            tgt = self.norm2(tgt + tgt2)

        # 3. FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


# =============================================================================
# 5. TransformerDecoder
# =============================================================================
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, ref_tgt, ref_span, time_emb, src, src_mask, pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, ref_tgt, ref_span, time_emb, src, src_mask, pos)
        return output