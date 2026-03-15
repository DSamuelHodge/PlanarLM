"""
main.py — PlanarLM build, inspection, and smoke-test.

Exercises:
  1. Model construction and parameter breakdown
  2. Forward pass shape validation
  3. Causality check (perturbation after t propagates, before t is silent)
  4. Loss computation and a single backward pass (gradient health check)
"""

import sys
import torch
import torch.nn as nn
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PlanarLM import PlanarLM
from PlanarLM.constants import VOCAB_SIZE, CHANNELS, NUM_WIRES, DEPTH, ALPHA_INIT, TIE_WEIGHTS, BATCH, SEQ_LEN

# You can override any constant here by passing as a keyword argument to PlanarLM
# Example: model = PlanarLM(vocab_size=128, channels=32, ...)

# ── Build ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PlanarLM — manifold-resident language model")
print("=" * 60)

# Standard instantiation (uses constants)

# Enable debug mode for demonstration
model = PlanarLM(
    vocab_size=VOCAB_SIZE,
    channels=CHANNELS,
    num_wires=NUM_WIRES,
    depth=DEPTH,
    alpha_init=ALPHA_INIT,
    tie_weights=TIE_WEIGHTS,
    debug=True,
)

# Custom instantiation example (overrides defaults)
custom_model = PlanarLM(
    vocab_size=128,
    channels=32,
    num_wires=3,
    depth=2,
    alpha_init=0.8,
    tie_weights=False,
    debug=True,
)

print("\nCustom PlanarLM instantiated with:")
print("  vocab_size=128, channels=32, num_wires=3, depth=2, alpha_init=0.8, tie_weights=False")
custom_x = torch.randint(0, 128, (2, 16))
custom_logits = custom_model(custom_x)
print(f"  Custom input shape: {tuple(custom_x.shape)}")
print(f"  Custom logits shape: {tuple(custom_logits.shape)}")

# ── Parameter breakdown ───────────────────────────────────────────────────────
counts = model.count_parameters()
print("\nParameter breakdown")
print("-" * 40)
for name, n in counts.items():
    print(f"  {name:<18}: {n:>10,}")
print(f"\nReceptive field : {model.receptive_field:,} tokens")
print(f"Spider layers   : {NUM_WIRES * DEPTH}")

# ── Forward pass ──────────────────────────────────────────────────────────────
print("\nForward pass")
print("-" * 40)
x      = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
logits = model(x)
print(f"  Input  : {tuple(x.shape)}")
print(f"  Logits : {tuple(logits.shape)}")
assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), "Logit shape mismatch"
print("  Shape  : OK")

# ── Causality check ───────────────────────────────────────────────────────────
print("\nCausality check")
print("-" * 40)
pivot  = SEQ_LEN // 2
x2     = x.clone()
x2[:, pivot] = (x2[:, pivot] + 1) % VOCAB_SIZE   # perturb one token at pivot
with torch.no_grad():
    l1 = model(x)
    l2 = model(x2)

before = (l1[:, :pivot] - l2[:, :pivot]).abs().max().item()
after  = (l1[:, pivot:] - l2[:, pivot:]).abs().max().item()
print(f"  Max diff before pivot (must be 0.00e+00): {before:.2e}")
print(f"  Max diff after  pivot (must be  > 0)    : {after:.2e}")
assert before == 0.0, f"Causality violated before pivot: {before}"
assert after  >  0.0, f"Perturbation had no effect after pivot: {after}"
print("  Causality: OK")

# ── Loss + backward ───────────────────────────────────────────────────────────
print("\nGradient health check")
print("-" * 40)
targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
loss_fn = nn.CrossEntropyLoss()
# CrossEntropyLoss expects (N, C, *) or reshape to (B*L, vocab)
loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
loss.backward()

grad_norms = {
    name: p.grad.norm().item()
    for name, p in model.named_parameters()
    if p.grad is not None
}
max_norm  = max(grad_norms.values())
min_norm  = min(grad_norms.values())
nan_count = sum(1 for v in grad_norms.values() if v != v)   # NaN check

print(f"  Loss            : {loss.item():.4f}")
print(f"  Grad norm range : [{min_norm:.3e}, {max_norm:.3e}]")
print(f"  NaN gradients   : {nan_count}")
assert nan_count == 0, "NaN gradients detected!"
print("  Gradients: OK")

print("\n" + "=" * 60)
print("All checks passed.")
print("=" * 60)
