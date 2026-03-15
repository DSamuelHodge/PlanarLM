"""
model.py — PlanarLM: a fully manifold-resident language model.

Architecture
------------
  Embedding  →  PlanarMesh (SpiderLayer stack)  →  ManifoldProjectionHead

Every intermediate representation from the first Spider layer onward lives
on the (−1,1)^C hypercube manifold.  The head lifts the final state back to
the tangent space via atanh before projecting to vocabulary logits, keeping
the geometry consistent end-to-end.

Weight tying between the embedding matrix and the output projection is
applied by default, following standard LM practice and reinforcing the
symmetry between encoding and decoding token geometry.
"""
from __future__ import annotations

import torch
import torch.nn as nn


from .planar_mesh import PlanarMesh
from .manifold_head import ManifoldProjectionHead
from .constants import VOCAB_SIZE, CHANNELS, NUM_WIRES, DEPTH, ALPHA_INIT, TIE_WEIGHTS



class PlanarLM(nn.Module):
    """
    Manifold-resident autoregressive language model built from Frobenius
    spider morphisms.

    Parameters
    ----------
    vocab_size  : vocabulary / token-type count
    channels    : feature width C (preserved throughout the mesh)
    num_wires   : spider layers per dilation block — sets dilation cycle
                  [1, 2, 4, …, 2^(num_wires-1)]
    depth       : number of full dilation cycles
    alpha_init  : DyT Lipschitz scalar initialisation
    tie_weights : if True, share embedding ↔ projection matrices
    """

    @classmethod
    def from_config(cls, config: dict = None, **overrides):
        """
        Instantiate PlanarLM from a config dict (or argparse.Namespace).
        Any key in config or overrides is passed as a keyword argument.
        Example:
            config = {"vocab_size": 128, "channels": 32}
            model = PlanarLM.from_config(config)
        """
        if config is None:
            config = {}
        # Allow argparse.Namespace as well as dict
        if hasattr(config, '__dict__'):
            config = vars(config)
        params = {**config, **overrides}
        return cls(**params)

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        channels: int = CHANNELS,
        num_wires: int = NUM_WIRES,
        depth: int = DEPTH,
        alpha_init: float = ALPHA_INIT,
        tie_weights: bool = TIE_WEIGHTS,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels   = channels
        self.debug = debug

        # Token embedding — lives in ℝ^C; first Spider layer retracts it
        # onto the manifold via DyT.
        self.embed = nn.Embedding(vocab_size, channels)

        # Planar mesh: num_wires * depth causal SpiderLayers
        self.mesh = PlanarMesh(
            channels=channels,
            num_wires=num_wires,
            depth=depth,
            alpha_init=alpha_init,
            debug=debug,
        )

        # Geometrically correct output head with optional weight tying
        self.head = ManifoldProjectionHead(
            channels=channels,
            vocab_size=vocab_size,
            embed_weight=self.embed.weight if tie_weights else None,
            alpha_init=alpha_init,
            debug=debug,
        )

    @property
    def receptive_field(self) -> int:
        """Causal receptive field in tokens (delegated to PlanarMesh)."""
        return self.mesh.receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, L) integer token ids
        Returns:
            logits : (B, L, vocab_size) — tangent-space logits
        """
        h = self.embed(x).transpose(1, 2)   # (B, C, L)
        if self.debug:
            print(f"[PlanarLM] embed(x) shape: {h.shape}")
        h = self.mesh(h)                     # (B, C, L) — on (−1,1)^C
        if self.debug:
            print(f"[PlanarLM] mesh(h) shape: {h.shape}")
        out = self.head(h)                  # (B, L, vocab_size)
        if self.debug:
            print(f"[PlanarLM] head(h) shape: {out.shape}")
        return out

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts by sub-module for inspection."""
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        embed_params = _n(self.embed)
        mesh_params  = _n(self.mesh)
        # head shares embed.weight when tied, count non-shared params only
        head_own = sum(
            p.numel() for name, p in self.head.named_parameters()
            if "proj.weight" not in name or not any(
                p is ep for ep in self.embed.parameters()
            )
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "embed"      : embed_params,
            "mesh"       : mesh_params,
            "head (own)" : head_own,
            "total"      : total,
        }
