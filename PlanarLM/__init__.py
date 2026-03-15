"""PlanarLM — manifold-resident autoregressive language model."""
from .constants import *
from .spider_layer import SpiderLayer
from .planar_mesh import PlanarMesh
from .manifold_head import ManifoldProjectionHead
from .model import PlanarLM

__all__ = [
	"SpiderLayer",
	"PlanarMesh",
	"ManifoldProjectionHead",
	"PlanarLM",
	"VOCAB_SIZE", "CHANNELS", "NUM_WIRES", "DEPTH", "ALPHA_INIT", "TIE_WEIGHTS"
]
