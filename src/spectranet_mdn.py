from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as f


class SpectraNetMDN(nn.Module):
    """
    Mixture Density Network over normalized (thickness, n, k).
    """

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
        dropout: float = 0.2,
        n_components: int = 5,
    ):
        super().__init__()
        self.n_components = int(n_components)
        blocks: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            blocks.extend([nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)])
            current = width
        self.backbone = nn.Sequential(*blocks)
        self.logits_head = nn.Linear(current, self.n_components)
        self.mean_head = nn.Linear(current, self.n_components * 3)
        self.scale_head = nn.Linear(current, self.n_components * 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        logits = self.logits_head(h)
        means = torch.sigmoid(self.mean_head(h)).view(-1, self.n_components, 3)
        scales = (f.softplus(self.scale_head(h)).view(-1, self.n_components, 3) + 1e-4)
        return {"logits": logits, "means": means, "scales": scales}


def mdn_negative_log_likelihood(
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    NLL for independent Gaussian mixture in 3D normalized parameter space.
    """
    target = target.unsqueeze(1)  # (B,1,3)
    var = scales * scales
    log_norm = -0.5 * torch.log(2.0 * math.pi * var)
    quad = -0.5 * ((target - means) ** 2) / var
    comp_log_prob = torch.sum(log_norm + quad, dim=-1)  # (B,K)
    log_mix = torch.log_softmax(logits, dim=-1)
    log_prob = torch.logsumexp(log_mix + comp_log_prob, dim=-1)
    return -log_prob.mean()


@torch.no_grad()
def sample_mdn_posterior(
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    n_samples: int = 64,
) -> torch.Tensor:
    """
    Return posterior samples in normalized space: shape (B, n_samples, 3).
    """
    bsz, n_components, n_dim = means.shape
    probs = torch.softmax(logits, dim=-1)
    comp = torch.multinomial(probs, num_samples=n_samples, replacement=True)  # (B,S)
    comp_expand = comp.unsqueeze(-1).expand(-1, -1, n_dim)
    chosen_mean = torch.gather(means, dim=1, index=comp_expand)
    chosen_scale = torch.gather(scales, dim=1, index=comp_expand)
    eps = torch.randn_like(chosen_mean)
    samples = chosen_mean + chosen_scale * eps
    return torch.clamp(samples, 0.0, 1.0)
