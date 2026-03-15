"""
planar_mesh.py — Planar mesh of SpiderLayers with exponential dilation schedule.

A PlanarMesh has `depth` blocks; each block contains `num_wires` SpiderLayers
whose dilations follow the geometric sequence [1, 2, 4, …, 2^(num_wires-1)].
This tiles the planar diagram with exponentially growing receptive fields:
  receptive_field = (2^num_wires - 1) * 2 * depth + 1

Every layer keeps activations manifold-resident on (−1,1)^C, so the mesh
output can be handed directly to ManifoldProjectionHead without any additional
retraction.
"""

import torch
import torch.nn as nn
from .constants import CHANNELS, NUM_WIRES, DEPTH, ALPHA_INIT
from .spider_layer import SpiderLayer


class PlanarMesh(nn.Module):
    """
    Planar composition of SpiderLayers wired in a dilated mesh.

    Parameters
    ----------
    channels  : feature width C (preserved throughout)
    num_wires : number of SpiderLayers per block; sets the dilation cycle
                [1, 2, 4, …, 2^(num_wires-1)]
    depth     : number of full dilation cycles (blocks)
    alpha_init: DyT Lipschitz scalar initialisation (propagated to all layers)
    """

    def __init__(
        self,
        channels: int = CHANNELS,
        num_wires: int = NUM_WIRES,
        depth: int = DEPTH,
        alpha_init: float = ALPHA_INIT,
        debug: bool = False,
    ):
        super().__init__()
        self.channels  = channels
        self.num_wires = num_wires
        self.depth     = depth
        self.debug     = debug

        layers = []
        for _ in range(depth):
            for wire in range(num_wires):
                dilation = 2 ** wire          # 1, 2, 4, 8, 16, …
                layers.append(
                    SpiderLayer(channels, dilation=dilation, alpha_init=alpha_init, debug=debug)
                )

        self.layers = nn.Sequential(*layers)  # total: num_wires * depth layers

    @property
    def receptive_field(self) -> int:
        """
        Causal receptive field in tokens.
        Each SpiderLayer with dilation d contributes (3-1)*d = 2d tokens.
        Summed over one block: 2*(1+2+4+…+2^(w-1)) = 2*(2^w - 1).
        Summed over depth blocks:  2 * depth * (2^num_wires - 1) + 1.
        """
        return 2 * self.depth * (2 ** self.num_wires - 1) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, L) — may or may not be manifold-resident at entry
        Returns:
            (B, C, L) — manifold-resident on (−1,1)^C
        """
        out = self.layers(x)
        if self.debug:
            print(f"[PlanarMesh] output shape: {out.shape}")
        return out
