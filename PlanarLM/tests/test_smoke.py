import torch

from PlanarLM import PlanarLM
from PlanarLM.algebraic_trainer import AlgebraicTrainer
from PlanarLM.frobenius_algebra import FrobeniusDuality, MorphismGap


def _small_model() -> PlanarLM:
    return PlanarLM(
        vocab_size=64,
        channels=8,
        num_wires=2,
        depth=1,
        alpha_init=0.5,
        tie_weights=True,
        debug=False,
    )


def test_algebraic_trainer_step_runs_and_returns_metrics() -> None:
    model = _small_model()
    trainer = AlgebraicTrainer(model=model, step_size=0.05, n_rounds=2)
    x = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))

    metrics = trainer.train_step(x, targets)

    required_keys = {"algebraic_loss", "ce_loss", "perplexity", "h_norm", "gap_norm"}
    assert required_keys.issubset(metrics.keys())
    for key in required_keys:
        assert isinstance(metrics[key], float)
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_frobenius_dual_and_gap_are_finite() -> None:
    model = _small_model()
    duality = FrobeniusDuality(channels=model.channels)
    gap_fn = MorphismGap(channels=model.channels, vocab_size=model.vocab_size)

    x = torch.randint(0, model.vocab_size, (2, 10))
    targets = torch.randint(0, model.vocab_size, (2, 10))

    h = model.embed(x).transpose(1, 2)
    h = model.mesh(h)
    logits = model.head(h)

    dual = duality.sigma_inv(h)
    gap = gap_fn(h, logits, targets, model.embed.weight)

    assert torch.isfinite(dual).all()
    assert torch.isfinite(gap).all()
    assert dual.shape == h.shape
    assert gap.shape == h.shape
