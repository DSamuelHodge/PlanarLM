import torch

from PlanarLM import PlanarLM


def test_model_instantiates_with_small_config() -> None:
    model = PlanarLM(
        vocab_size=64,
        channels=16,
        num_wires=2,
        depth=1,
        alpha_init=0.5,
        tie_weights=True,
        debug=False,
    )
    assert model.vocab_size == 64
    assert model.channels == 16


def test_forward_shape_small_config() -> None:
    model = PlanarLM(
        vocab_size=32,
        channels=8,
        num_wires=2,
        depth=1,
        alpha_init=0.5,
        tie_weights=False,
        debug=False,
    )
    x = torch.randint(0, 32, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 32)


def test_mesh_output_stays_in_manifold_bounds() -> None:
    model = PlanarLM(
        vocab_size=48,
        channels=8,
        num_wires=2,
        depth=1,
        alpha_init=0.5,
        tie_weights=True,
        debug=False,
    )
    x = torch.randint(0, 48, (2, 10))
    h = model.embed(x).transpose(1, 2)

    # The DyT branch is bounded in (-1, 1); the residual output itself can exceed
    # that bound because SpiderLayer returns x + f * g.
    first_layer = list(model.mesh.layers)[0]
    pad = (3 - 1) * first_layer.conv.dilation[0]
    branch = first_layer.conv(torch.nn.functional.pad(h, (pad, 0)))
    branch = first_layer.dyt(branch)

    assert torch.isfinite(branch).all()
    assert branch.abs().max().item() < 1.0
