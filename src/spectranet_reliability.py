from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class SpectraNetReliability(nn.Module):
    """
    SpectraNet backbone with three heads:
      - parameter mean head (normalized thickness, n, k)
      - uncertainty head (positive std proxy in normalized space)
      - risk head (catastrophic probability logit)
    """

    def __init__(self, input_dim: int = 200, hidden_dims: tuple[int, ...] = (512, 256, 128, 64), dropout: float = 0.2):
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            blocks.extend([nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)])
            current = width
        self.backbone = nn.Sequential(*blocks)
        self.param_head = nn.Sequential(nn.Linear(current, 3), nn.Sigmoid())
        self.log_std_head = nn.Linear(current, 3)
        self.risk_head = nn.Linear(current, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        param_mean = self.param_head(h)
        param_std = f.softplus(self.log_std_head(h)) + 1e-6
        risk_logit = self.risk_head(h).squeeze(-1)
        return {
            "param_mean": param_mean,
            "param_std": param_std,
            "risk_logit": risk_logit,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(x)
        out["risk_prob"] = torch.sigmoid(out["risk_logit"])
        return out
