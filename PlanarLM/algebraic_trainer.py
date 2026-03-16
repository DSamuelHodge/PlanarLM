"""Training loop based on Algebraic Analysis for PlanarLM."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .frobenius_algebra import FrobeniusDuality, MorphismGap
from .model import PlanarLM


class AlgebraicTrainer:
    """Train PlanarLM using algebraic corrections instead of gradients."""

    def __init__(
        self,
        model: PlanarLM,
        step_size: float = 0.05,
        n_rounds: int = 3,
    ):
        self.model = model
        self.step_size = step_size
        self.n_rounds = n_rounds
        self.duality = FrobeniusDuality(model.channels)
        self.gap_fn = MorphismGap(model.channels, model.vocab_size)

    def morphism_loss(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Fisher-weighted tangent-space gap norm."""
        embed_weight = self.model.embed.weight
        target_emb = embed_weight[targets].transpose(1, 2)
        h_tangent = self.duality.lift(h)
        gap = target_emb - h_tangent
        w = self.duality.fisher_weight(h)
        return (w * gap**2).mean()

    def _forward_with_internals(self, x: torch.Tensor):
        """Forward pass returning logits and final mesh state."""
        h = self.model.embed(x).transpose(1, 2)
        h = self.model.mesh(h)
        logits = self.model.head(h)
        return logits, h

    def train_step(self, x: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Single algebraic step with no backward() and no optimizer."""
        with torch.no_grad():
            logits, h = self._forward_with_internals(x)

        alg_loss = self.morphism_loss(h, targets)
        gap = self.gap_fn(h, logits, targets, self.model.embed.weight)

        with torch.no_grad():
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.model.vocab_size),
                targets.reshape(-1),
            )

        layers = list(self.model.mesh.layers)
        with torch.no_grad():
            for _ in range(max(1, self.n_rounds)):
                for layer in reversed(layers):
                    if hasattr(layer, "algebraic_update"):
                        h = layer.algebraic_update(h, gap, self.step_size)
                        gap = self.duality.sigma_inv(h)

        return {
            "algebraic_loss": alg_loss.item(),
            "ce_loss": ce_loss.item(),
            "perplexity": ce_loss.exp().item(),
            "h_norm": h.abs().mean().item(),
            "gap_norm": gap.norm(dim=1).mean().item(),
        }

    def fit(self, dataloader, epochs: int = 1, log_every: int = 10) -> List[Dict[str, float]]:
        """Epoch loop for algebraic training."""
        history: List[Dict[str, float]] = []
        step = 0
        for epoch in range(epochs):
            for x, targets in dataloader:
                metrics = self.train_step(x, targets)
                history.append(metrics)
                if step % log_every == 0:
                    print(
                        f"Epoch {epoch} Step {step} | "
                        f"AlgLoss={metrics['algebraic_loss']:.4f} | "
                        f"CE={metrics['ce_loss']:.4f} | "
                        f"PPL={metrics['perplexity']:.2f} | "
                        f"GapNorm={metrics['gap_norm']:.4f}"
                    )
                step += 1
        return history
