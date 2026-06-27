import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

try:
    from .lap import auction_lap
except ImportError:
    auction_lap = None

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_span: float = 1, match_boundary_type = 'f1', solver = "hungarian"):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.match_boundary_type = match_boundary_type
        self.solver = solver

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # [FIX] 兼容修复: 确保 sizes 是 int list
        sizes = targets["sizes"]
        if torch.is_tensor(sizes):
            sizes = sizes.cpu().tolist()

        if sum(sizes) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        if self.solver == "order":
            return [(list(range(s)), list(range(s))) for s in sizes]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1)
        entity_left = outputs["pred_left"].flatten(0, 1)
        entity_right = outputs["pred_right"].flatten(0, 1)

        gt_ids = targets["labels"]
        gt_left = targets["gt_left"]
        gt_right = targets["gt_right"]

        cost_class = -out_prob[:, gt_ids]

        cost_span = 0
        if self.match_boundary_type == "f1":
            pred_left_idx = entity_left.argmax(dim=-1).unsqueeze(-1)
            pred_right_idx = entity_right.argmax(dim=-1).unsqueeze(-1)
            cost_span = torch.abs(pred_left_idx - gt_left.unsqueeze(0)) + \
                        torch.abs(pred_right_idx - gt_right.unsqueeze(0))
        elif self.match_boundary_type == "logp":
            max_len = entity_left.size(1)
            # [Safety] Prevent index out of bounds
            gt_l = gt_left.long().clamp(0, max_len-1)
            gt_r = gt_right.long().clamp(0, max_len-1)
            cost_span = -(entity_left[:, gt_l] + entity_right[:, gt_r])

        C = self.cost_span * cost_span + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1)

        # [FIX] 数值清洗，防止 NaN 崩溃
        if torch.isnan(C).any() or torch.isinf(C).any():
            C = torch.nan_to_num(C, nan=100.0, posinf=1e4, neginf=-1e4)

        C = C.cpu()
        
        # sizes 已经是 int list，split 不会报错
        if self.solver == "hungarian":
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        elif self.solver == "auction" and auction_lap is not None:
            indices = [auction_lap(c[i])[:2] for i, c in enumerate(C.split(sizes, -1))]
        else:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]