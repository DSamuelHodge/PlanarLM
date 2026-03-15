"""
spider_layer.py — Frobenius (2,1)-spider morphism on the (−1,1)^d manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import CHANNELS, ALPHA_INIT


class SpiderLayer(nn.Module):
    """
    Frobenius (2,1)-spider morphism on the (−1,1)^d manifold.
    Single Conv1d(C→2C) produces filter and gate paths. DyT retracts
    the full 2C tensor before split. Residual is the monoidal identity.

    Architecture:
        x ──┬──────────────────────────────── (+) ── x_{t+1}
            └── Conv1d(C→2C) ── DyT(2C) ── chunk ── f⊙g ──┘

    Parameters
    ----------
    channels   : number of feature channels C
    dilation   : causal dilation factor (receptive field = (3-1)*dilation)
    alpha_init : initial value of the DyT Lipschitz scalar
    """

    def __init__(self, channels: int = CHANNELS, dilation: int = 1, alpha_init: float = ALPHA_INIT, debug: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=3,
            dilation=dilation,
            padding=0,  # causal padding applied manually in forward()
        )
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(1, channels * 2, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels * 2, 1))
        self.debug = debug

    def dyt(self, x: torch.Tensor) -> torch.Tensor:
        """γ · tanh(α · x) + β  — retracts ℝ^{2C} → (−1,1)^{2C} coordinatewise.
        Gradient bound: ‖∂DyT/∂x‖ ≤ sech²(αx) ≤ 1 everywhere."""
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, L)
        Returns:
            (B, C, L) — same shape, manifold-resident
        """
        # Left-only (causal) padding: (kernel_size - 1) * dilation
        pad = (3 - 1) * self.conv.dilation[0]
        h = self.conv(F.pad(x, (pad, 0)))   # (B, 2C, L)
        h = self.dyt(h)                      # retract onto (−1,1)^{2C}
        f, g = torch.chunk(h, 2, dim=1)     # each (B, C, L)
        out = x + f * g                     # monoidal identity residual
        if self.debug:
            print(f"[SpiderLayer] output shape: {out.shape}")
        return out
