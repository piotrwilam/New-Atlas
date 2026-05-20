"""Forward-hook ablation of MLP-output neurons at a single layer.

Paper §7.1: per-layer zero-ablation of concept-only neurons is compared
against a size-matched random-null ablation on two prompt sets per
concept (concept prompts vs checker prompts). A clean double
dissociation requires a large negative Δ-log-prob on concept prompts
*and* a small Δ on checker prompts — for the *same* layer.

This module owns the forward-hook machinery; the statistical comparison
of the Δs lives in `atlas.analysis.dissociation`. The separation keeps
the heavy PyTorch dependency out of `atlas/`, so analysis-only
reproductions of the paper figures don't need torch installed.
"""

from __future__ import annotations


class AblationHook:
    """Zero out a fixed set of neuron indices at one MLP layer's output.

    The hook attaches to the layer's `mlp` submodule and rewrites its
    forward output so the specified channels of dim 2 (the MLP hidden
    dim) are set to 0.0. Other channels and other layers are unaffected.

    Usage:
        with AblationHook(model, layer_id=19, neuron_indices=[12, 78, 113]):
            logits = model(input_ids).logits

    On exit, the hook is removed and the model returns to its
    unmodified behaviour. Can also be used as a non-context manager via
    `register()` / `remove()`.

    Parameters
    ----------
    model
        A HuggingFace causal LM whose layers are accessible as
        `model.layers.{i}.mlp`. The hook pattern matches Qwen/Llama-style
        architectures.
    layer_id
        0-based layer index.
    neuron_indices
        Iterable of int indices to zero in the MLP output. Empty list is
        allowed (no-op hook, useful for sanity baselines).
    """

    _HOOK_PATTERN = "model.layers.{layer_id}.mlp"

    def __init__(self, model, layer_id: int, neuron_indices) -> None:
        assert layer_id >= 0, f"layer_id must be non-negative, got {layer_id}"
        self.layer_id = layer_id
        self.neuron_indices = list(neuron_indices)
        self.handle = None

        module_dict = dict(model.named_modules())
        mlp_name = self._HOOK_PATTERN.format(layer_id=layer_id)
        if mlp_name not in module_dict:
            raise ValueError(
                f"MLP submodule not found: {mlp_name!r}. "
                f"Check model.named_modules() for the correct pattern."
            )
        self.module = module_dict[mlp_name]

    def _hook_fn(self, module, input, output):
        # MLP outputs are either a tensor or (tensor, ...). Handle both.
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if not self.neuron_indices:
            return output
        modified = out.clone()
        modified[..., self.neuron_indices] = 0.0
        return (modified,) + output[1:] if is_tuple else modified

    def register(self) -> "AblationHook":
        assert self.handle is None, "hook already registered"
        self.handle = self.module.register_forward_hook(self._hook_fn)
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "AblationHook":
        return self.register()

    def __exit__(self, *_exc) -> None:
        self.remove()
