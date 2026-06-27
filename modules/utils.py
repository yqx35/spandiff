# -*- coding: utf-8 -*-
import torch
import math
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def constant_beta_schedule(timesteps):
    scale = 1000 / timesteps
    constant = scale * 0.01
    return torch.tensor([constant] * timesteps, dtype = torch.float64)


def sigmoid_beta_schedule(timesteps):
    """
    Proposed in 'Improved DiffusionDet':
    Deformable sigmoid variance schedule for better sparse feature capture.
    """
    # 论文推荐参数: range [-10, 0], scale 0.8, shift 0.0008
    betas = torch.linspace(-10, 0, timesteps, dtype=torch.float64)
    betas = torch.sigmoid(betas) * (800e-3) + 0.0008
    return betas.float()

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


def span_lw_to_lr(x):
    l, w = x.unbind(-1)
    b = [l, l + w]
    return torch.stack(b, dim=-1)


def span_lr_to_lw(x):
    l, r = x.unbind(-1)
    b = [l, r-l]
    return torch.stack(b, dim=-1)

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end+1] = 1
    return mask


def span_iou(span1,span2):
    inter = torch.max(torch.zeros(1, device=span1.device),
                      torch.min(span1[:, :, None, 1], span2[:, None, :, 1]) - torch.max(span1[:, :, None, 0],span2[:, None, :, 0]))

    union = (span1[:, :, None, 1] - span1[:, :, None, 0]) + (
                span2[:, None, :, 1] - span2[:, None, :, 0]) - inter
    iou = inter / (union + 1e-6)
    inx = iou<0
    iou[inx] = 0
    return iou


def entity_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    # >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    # >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    # >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left)#.clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / (union+1e-8)
    inx = iou<0
    iou[inx] = 0
    # iou[indx2] = 0
    return iou, union


