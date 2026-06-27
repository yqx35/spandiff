# -*- coding: utf-8 -*-
import csv
import json
import os
import random
import shutil
import math
import numpy as np
import torch
import torch.nn.functional as F

# 如果你有自定义的 entities 文件，保留这行，否则注释掉
# from spandiff.entities import *

CSV_DELIMETER = ';'

# ==========================================================
# 1. 基础工具函数 (File/Dir/Log)
# ==========================================================

def create_directories_file(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d

def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if column_names:
                writer.writerow(column_names)

def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)

def append_csv_multiple(file_path, *rows):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

def read_csv(file_path):
    lines = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)
    return lines[0], lines[1:]

def copy_python_directory(source, dest, ignore_dirs=None):
    source = source if source.endswith('/') else source + '/'
    for (dir_path, dir_names, file_names) in os.walk(source):
        tail = '/'.join(dir_path.split(source)[1:])
        new_dir = os.path.join(dest, tail)
        if ignore_dirs and True in [(ignore_dir in tail) for ignore_dir in ignore_dirs]:
            continue
        create_directories_dir(new_dir)
        for file_name in file_names:
            if file_name.endswith('.py'):
                file_path = os.path.join(dir_path, file_name)
                shutil.copy2(file_path, new_dir)

def save_dict(log_path, dic, name):
    path = os.path.join(log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f, sort_keys=True, indent=4)
    f.close()
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()

def summarize_dict(summary_writer, dic, name):
    table = 'Argument|Value\n-|-'
    for k, v in vars(dic).items():
        row = '\n%s|%s' % (k, v)
        table += row
    summary_writer.add_text(name, table)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    for f in logger.filters[:]:
        logger.removeFilters(f)

def flatten(l):
    return [i for p in l for i in p]

def get_as_list(dic, key):
    if key in dic: return [dic[key]]
    else: return []

# ==========================================================
# 2. Tensor 处理 (Combine, Pad, etc.)
# ==========================================================

def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape
    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)
    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor
    return extended_tensor

def padded_stack(tensors, padding=0):
    if not tensors:
        return torch.tensor([])
    dim_count = len(tensors[0].shape)
    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []
    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)
    stacked = torch.stack(padded_tensors)
    return stacked

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()
    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])

def padded_nonzero(tensor, padding=0):
    indices = padded_stack([tensor[i].nonzero().view(-1) for i in range(tensor.shape[0])], padding)
    return indices

def swap(v1, v2):
    return v2, v1

def to_device(batch, device, skip_keys = ['meta_doc'], nested_keys = ['image_inputs']):
    converted_batch = dict()
    for key in batch.keys():
        if key in nested_keys:
            if batch[key] == None:
                converted_batch[key] = None
            else:
                converted_batch[key] = dict((k, v.to(device)) for k, v in batch[key].items())
            continue
        if batch[key] is None:
            converted_batch[key] = None
            continue
        if key in skip_keys:
            converted_batch[key] = batch[key]
        else:
            converted_batch[key] = batch[key].to(device)
    return converted_batch

def round(arr, n_digits):
    return torch.round(arr * 10**n_digits) / (10**n_digits)

# [CRITICAL FIX] Combine function robust to None mask
def combine(sub, sup_mask, pool_type="max"):
    """
    Sub-word feature pooling.
    sub: (B, SeqLen, Dim)
    sup_mask: (B, SeqLen, WordLen) mapping or None
    """
    # 如果 sup_mask 为 None，说明不需要池化或者已经池化好了，直接返回 sub
    if sup_mask is None:
        return sub

    sup = None
    
    # 针对特定需求的 first/last pooling
    if pool_type == "first":
        sup_mask_shift = torch.roll(sup_mask, 1, -1)
        sup_mask = sup_mask & (~sup_mask_shift)
        m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
        sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
        sup = sup.max(dim=2)[0]
        sup[sup == -1e30] = 0
        return sup

    if pool_type == "last":
        sup_mask_shift = torch.roll(sup_mask, -1, -1)
        sup_mask = sup_mask & (~sup_mask_shift)
        m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
        sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
        sup = sup.max(dim=2)[0]
        sup[sup == -1e30] = 0
        return sup

    # 通用 pooling (mean, sum, max)
    # sub -> B #ST E ==== sup_mask -> B #T #ST
    # 检查维度是否匹配以决定是否需要 unsqueeze
    if len(sub.shape) == len(sup_mask.shape):   
        # 维度一致的情况
        if pool_type == "mean":
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2) / size
        if pool_type == "sum":
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2)
        if pool_type == "max":
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    else: 
        # sub -> B #T #C E ==== sup_mask -> B #T #C
        if pool_type == "mean":
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2) / size
        if pool_type == "sum":
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2)
        if pool_type == "max":
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
            
    return sup

# ==========================================================
# 3. Diffusion & Span Tools (新增，支持 Models.py)
# ==========================================================

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if val is not None: return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def constant_beta_schedule(timesteps):
    return torch.tensor([0.0001] * timesteps, dtype=torch.float64)

def span_lw_to_lr(span):
    """[Center, Width] -> [Left, Right]"""
    center, width = span.unbind(-1)
    left = center - width / 2
    right = center + width / 2
    return torch.stack((left, right), dim=-1)

def span_lr_to_lw(span):
    """[Left, Right] -> [Center, Width]"""
    left, right = span.unbind(-1)
    center = (left + right) / 2
    width = right - left
    return torch.stack((center, width), dim=-1)

def segment_iou(box_a, box_b):
    """1D IoU Calculation"""
    if box_a.dim() == 2: box_a = box_a.unsqueeze(0)
    if box_b.dim() == 2: box_b = box_b.unsqueeze(0)
    box_a = box_a.unsqueeze(2)
    box_b = box_b.unsqueeze(1)
    start = torch.max(box_a[..., 0], box_b[..., 0])
    end = torch.min(box_a[..., 1], box_b[..., 1])
    inter = torch.clamp(end - start, min=0)
    area_a = box_a[..., 1] - box_a[..., 0]
    area_b = box_b[..., 1] - box_b[..., 0]
    union = area_a + area_b - inter
    return inter / (union + 1e-6)

def create_entity_mask(start, end, context_len):
    mask = torch.zeros(context_len, dtype=torch.bool)
    start = max(0, min(int(start), context_len - 1))
    end = max(0, min(int(end), context_len))
    mask[start:end] = True
    return mask


def count_parameters(model, only_trainable=True):
    """Calculate the number of model parameters"""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def format_num_params(num_params):
    """Format parameter count for display"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return f"{num_params}"