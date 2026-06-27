# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityBoundaryPredictor
class SpanBoundaryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.left_boundary_predictor = EntityBoundaryPredictor(config)
        self.right_boundary_predictor = EntityBoundaryPredictor(config)

    def forward(self,memory, query_embed, mask):
        left_event_token_p = self.left_boundary_predictor(memory, query_embed, mask)
        right_event_token_p = self.right_boundary_predictor(memory, query_embed, mask)
        trigger_left = left_event_token_p.argmax(dim=-1)
        trigger_right = right_event_token_p.argmax(dim=-1)
        spans_pred = torch.stack([trigger_left, trigger_right], dim=-1)
        # spans_pred = filter_boundary(spans_pred,mask)
        return spans_pred

def filter_boundary(span,token_masks):
    token_count = token_masks.long().sum(-1, keepdim=True)
    spans = span.to(dtype=torch.long)#torch.round(span)#.to(dtype=torch.long)
    spans[:, :, 0][spans[:, :, 0] < 0] = 0
    spans[:, :, 1][spans[:, :, 1] < 0] = 0
    spans[:, :, 0][spans[:, :, 0] > spans[:, :, 1]] = 0
    spans[:, :, 1][spans[:, :, 1] < spans[:, :, 0]] = 0
    spans[:, :, 0][spans[:, :, 0] >= token_count] = \
    token_count.repeat(1, spans.size(1))[spans[:, :, 0] >= token_count] - 1
    spans[:, :, 1][spans[:, :, 1] >= token_count] = \
    token_count.repeat(1, spans.size(1))[spans[:, :, 1] >= token_count] - 1
    return spans