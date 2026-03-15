# PlanarLM

A fully manifold-resident language model built from Frobenius (2,1)-spider morphisms. All activations live on the (−1,1)^d hypercube manifold, with a geometrically correct output head. Hyperparameters are managed via `constants.py` and can be overridden at instantiation.

## Features
- Causal SpiderLayer stack with DyT retraction
- Planar mesh with exponential dilation schedule
- Manifold-aware output projection (atanh log map)
- Weight tying between embedding and output head
- All hyperparameters configurable via `constants.py` or keyword arguments


## Usage

```python
from PlanarLM import PlanarLM

# Default instantiation
model = PlanarLM()

# Custom instantiation
model = PlanarLM(vocab_size=128, channels=32, num_wires=3, depth=2)

# Advanced: from config dict or argparse.Namespace (for experiment scripts)
# config = {"vocab_size": 128, "channels": 32}
# model = PlanarLM.from_config(config)
```

## Testing

Run the smoke test:

```
python PlanarLM/main.py
```

## Files
- `spider_layer.py` — SpiderLayer block
- `planar_mesh.py` — Planar mesh of SpiderLayers
- `manifold_head.py` — Manifold-aware output head
- `model.py` — PlanarLM model wiring
- `constants.py` — Hyperparameter defaults
- `main.py` — Build and test script
