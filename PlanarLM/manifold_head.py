"""
manifold_head.py — Geometrically correct output projection for PlanarLM.

The Spider stack outputs h ∈ (−1,1)^C.  A standard nn.Linear projects this
into unbounded ℝ^vocab_size, discarding the manifold geometry.  The correct
approach is:

  1. Lift h to the tangent space via the log map (inverse of the tanh
     retraction used by DyT):
         h_tangent = atanh(h / γ) / α

  2. Apply the linear projection in tangent space, where Euclidean geometry
     is valid and dot products measure genuine angular distance.

  3. Optionally tie the projection weights to the embedding matrix so that
     vocabulary tokens are represented with the same geometry in both the
     encoder and the decoder (weight tying).

The clamp(-1+ε, 1−ε) in log_map is not a numerical hack — it enforces that
atanh stays finite at the manifold boundary.  DyT with finite α never
saturates exactly to ±1 in normal training, so the clamp is dormant except
at pathological initialisation or extreme inputs.
"""
from __future__ import annotations


import torch
import torch.nn as nn
from .constants import CHANNELS, VOCAB_SIZE, ALPHA_INIT


class ManifoldProjectionHead(nn.Module):
    """
    Geometrically correct output projection for PlanarLM.

    Parameters
    ----------
    channels    : feature width C (must match the Spider stack output)
    vocab_size  : number of output classes / token types
    embed_weight: optional embedding weight tensor of shape (vocab_size, C)
                  for weight tying; if supplied the projection matrix is
                  shared with the encoder embedding table
    alpha_init  : initial DyT Lipschitz scalar — should mirror the encoder
    """

    def __init__(
        self,
        channels: int = CHANNELS,
        vocab_size: int = VOCAB_SIZE,
        embed_weight: torch.Tensor | None = None,
        alpha_init: float = ALPHA_INIT,
        debug: bool = False,
    ):
        super().__init__()
        self.channels   = channels
        self.vocab_size = vocab_size
        self.debug = debug

        self.alpha_out = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma_out = nn.Parameter(torch.ones(1, channels, 1))
        self.proj = nn.Linear(channels, vocab_size, bias=False)
        if embed_weight is not None:
            assert embed_weight.shape == (vocab_size, channels), (
                f"embed_weight shape {embed_weight.shape} != ({vocab_size}, {channels})"
            )
            self.proj.weight = embed_weight

    def log_map(self, h: torch.Tensor) -> torch.Tensor:
        """
        Inverse DyT: lifts (−1,1)^C → ℝ^C via atanh.

        Inverts the coordinatewise tanh retraction applied by the Spider
        stack.  Divides by γ before atanh to undo the affine scale, then
        divides by α to undo the Lipschitz contraction.

        Args:
            h : (B, C, L)
        Returns:
            (B, C, L) tangent-space representation
        """
        eps = 1e-6
        h_safe   = h.clamp(-1.0 + eps, 1.0 - eps)
        h_scaled = h_safe / self.gamma_out.clamp(min=eps)   # undo γ scale
        return torch.atanh(h_scaled) / self.alpha_out        # undo α contraction

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (B, C, L) — final Spider stack output, lives on (−1,1)^C
        Returns:
            logits : (B, L, vocab_size) — tangent-space logits for cross-entropy
        """
        h_tangent = self.log_map(h)           # (B, C, L)  — in ℝ^C
        h_tangent = h_tangent.transpose(1, 2) # (B, L, C)  — for Linear
        out = self.proj(h_tangent)            # (B, L, vocab_size)
        if self.debug:
            print(f"[ManifoldHead] logits shape: {out.shape}")
        return out
