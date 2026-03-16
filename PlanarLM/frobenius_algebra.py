"""Algebraic Analysis engine for PlanarLM.

Implements Frobenius duality, morphism gap computation, coproduct correction
distribution, and normal-form reduction for manifold-resident states.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrobeniusDuality(nn.Module):
    """Frobenius form and dual mapping on the tanh manifold."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

    def lift(self, h: torch.Tensor) -> torch.Tensor:
        """Lift manifold points in (-1, 1)^C to tangent space via atanh."""
        h_safe = h.clamp(-1.0 + self.eps, 1.0 - self.eps)
        return torch.atanh(h_safe)

    def fisher_weight(self, h: torch.Tensor) -> torch.Tensor:
        """Fisher-Rao weight induced by tanh coordinates."""
        h_safe = h.clamp(-1.0 + self.eps, 1.0 - self.eps)
        return 1.0 / (1.0 - h_safe**2 + self.eps)

    def sigma(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Frobenius bilinear form sigma(u, v), reduced over channel axis."""
        w = self.fisher_weight(u)
        return (w * u * v).sum(dim=1)

    def sigma_inv(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the Frobenius dual direction sigma^{-1}(h)."""
        return self.lift(h) * self.fisher_weight(h)


class MorphismGap(nn.Module):
    """Compute the morphism gap in tangent space."""

    def __init__(self, channels: int, vocab_size: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.vocab_size = vocab_size
        self.eps = eps
        self.duality = FrobeniusDuality(channels=channels, eps=eps)

    def target_representation(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Map target token ids to tangent-space vectors using embedding basis."""
        del logits
        target_emb = embed_weight[targets]
        return target_emb.transpose(1, 2)

    def forward(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return tangent-space target minus current state."""
        h_tangent = self.duality.lift(h)
        t_tangent = self.target_representation(logits, targets, embed_weight)
        return t_tangent - h_tangent


class PlanarCoproductCorrection(nn.Module):
    """Distribute a correction through a planar coproduct split."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.delta_proj = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=3,
            dilation=dilation,
            padding=0,
        )

    def forward(self, correction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return left/right correction branches for a (2,1)-spider."""
        pad = (3 - 1) * self.dilation
        h = self.delta_proj(F.pad(correction, (pad, 0)))
        h = torch.tanh(h)
        f_corr, g_corr = torch.chunk(h, 2, dim=1)
        return f_corr, g_corr


class FrobeniusNormalForm(nn.Module):
    """Reduce activations toward fixed points of DyT retraction."""

    def __init__(
        self,
        channels: int,
        alpha_init: float = 0.5,
        max_iter: int = 5,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.max_iter = max_iter
        self.tol = tol

    def dyt(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iterate DyT until convergence tolerance or max_iter."""
        for _ in range(self.max_iter):
            x_new = self.dyt(x)
            if (x_new - x).abs().max().item() < self.tol:
                return x_new
            x = x_new
        return x
