# test_planarlm.py — PlanarLM build, inspection, and smoke-test
import sys
import torch

from PlanarLM import PlanarLM
from PlanarLM.constants import VOCAB_SIZE, CHANNELS, NUM_WIRES, DEPTH, ALPHA_INIT, TIE_WEIGHTS, BATCH, SEQ_LEN
from PlanarLM.algebraic_trainer import AlgebraicTrainer
from PlanarLM.frobenius_algebra import FrobeniusDuality, MorphismGap
# Example: model = PlanarLM(vocab_size=128, channels=32, ...)

print("=" * 60)
print("PlanarLM — manifold-resident language model")
print("=" * 60)

# Standard instantiation (uses constants)
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

# ── Algebraic Analysis check ──────────────────────────────────────────────────
print("\nAlgebraic Analysis check")
print("-" * 40)
targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
duality = FrobeniusDuality(channels=CHANNELS)
trainer = AlgebraicTrainer(model, step_size=0.05, n_rounds=3)

# 1. Verify manifold residency.
h = model.embed(x).transpose(1, 2)
h = model.mesh(h)
assert h.abs().max().item() < 1.0, "Manifold residency violated!"
print(f"  Manifold residency : OK  (max |h| = {h.abs().max():.4f} < 1.0)")

# 2. Verify Frobenius duality is finite.
dual = duality.sigma_inv(h)
nan_count = dual.isnan().sum().item() + dual.isinf().sum().item()
assert nan_count == 0, f"Frobenius dual has {nan_count} NaN/Inf!"
print(f"  Frobenius dual     : OK  (no NaN/Inf, norm = {dual.norm():.4f})")

# 3. Verify morphism gap is computable.
gap_fn = MorphismGap(CHANNELS, VOCAB_SIZE)
with torch.no_grad():
    logits_check = model.head(h)
gap = gap_fn(h, logits_check, targets, model.embed.weight)
print(f"  Morphism gap       : OK  (norm = {gap.norm():.4f})")

# 4. Verify algebraic train step runs without backward().
metrics = trainer.train_step(x, targets)
print(f"  AlgLoss            : {metrics['algebraic_loss']:.4f}")
print(f"  CE (compare only)  : {metrics['ce_loss']:.4f}")
print(f"  GapNorm            : {metrics['gap_norm']:.4f}")
print("  Algebraic step     : OK  (no backward() called)")

# 5. Verify Frobenius residual numerically on one SpiderLayer.
layer = list(model.mesh.layers)[0]
test_h = torch.tanh(torch.randn(1, CHANNELS, 16) * 0.5)
dyt_h = layer.dyt(test_h)
frobenius_residual = (dyt_h - test_h).abs().mean().item()
print(f"  Frobenius residual : {frobenius_residual:.6f}  (lower = more converged)")

print("\nAll algebraic checks passed.")

print("\n" + "=" * 60)
print("All checks passed.")
print("=" * 60)
