"""Tests for circuits.ablation.AblationHook — forward-hook neuron zeroing.

Uses a tiny synthetic nn.Module shaped to look like a HF causal LM
(model.layers[i].mlp), so the tests don't require a real LLM checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from circuits.ablation import AblationHook  # noqa: E402


class _ToyMLP(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc(x)


class _ToyLayer(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.mlp = _ToyMLP(dim)


class _ToyTransformer(torch.nn.Module):
    """Minimal nn.Module with `model.layers[i].mlp` structure that
    AblationHook expects."""
    def __init__(self, n_layers: int = 3, dim: int = 8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            _ToyLayer(dim) for _ in range(n_layers)
        )

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.mlp(x)
        return x


def test_hook_zeroes_specified_neurons() -> None:
    model = _ToyTransformer(n_layers=2, dim=8)
    # Make the MLP an identity so we can directly observe the hook's effect.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.fc.weight.copy_(torch.eye(8))

    x = torch.ones(1, 4, 8)  # (batch, seq, dim)
    baseline = model(x)
    assert torch.allclose(baseline, torch.ones_like(baseline))

    with AblationHook(model, layer_id=0, neuron_indices=[2, 5]):
        ablated = model(x)
    # After ablation at layer 0, channels 2 and 5 are zeroed. Layer 1's
    # identity MLP passes them through unchanged.
    assert ablated[0, 0, 2].item() == 0.0
    assert ablated[0, 0, 5].item() == 0.0
    # Other channels unchanged.
    assert ablated[0, 0, 0].item() == 1.0
    assert ablated[0, 0, 7].item() == 1.0


def test_hook_removed_after_context_exit() -> None:
    model = _ToyTransformer(n_layers=1, dim=4)
    x = torch.ones(1, 2, 4)
    with torch.no_grad():
        model.model.layers[0].mlp.fc.weight.copy_(torch.eye(4))

    with AblationHook(model, layer_id=0, neuron_indices=[1]):
        ablated = model(x)
    after = model(x)

    assert ablated[0, 0, 1].item() == 0.0
    # After context exit, neuron 1 should be back to 1.0.
    assert after[0, 0, 1].item() == 1.0


def test_empty_indices_is_noop() -> None:
    model = _ToyTransformer(n_layers=1, dim=4)
    with torch.no_grad():
        model.model.layers[0].mlp.fc.weight.copy_(torch.eye(4))
    x = torch.ones(1, 2, 4)
    baseline = model(x)
    with AblationHook(model, layer_id=0, neuron_indices=[]):
        ablated = model(x)
    assert torch.allclose(baseline, ablated)


def test_missing_layer_raises() -> None:
    model = _ToyTransformer(n_layers=2, dim=4)
    with pytest.raises(ValueError, match="MLP submodule not found"):
        AblationHook(model, layer_id=99, neuron_indices=[0])


def test_negative_layer_id_rejected() -> None:
    model = _ToyTransformer(n_layers=1, dim=4)
    with pytest.raises(AssertionError, match="non-negative"):
        AblationHook(model, layer_id=-1, neuron_indices=[0])
