import logging
import torch

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """
    Wraps a causal LM with forward hooks on MLP hidden states.

    Works with any HuggingFace model — set the hook pattern to match
    the model's MLP module names (e.g. 'model.layers.{layer_id}.mlp'
    for Qwen/Llama-style architectures).

    Inspect model.named_modules() to find the correct pattern, then
    call set_hook_pattern() before register_hooks().
    """

    def __init__(self, model, tokenizer, device, n_layers=28,
                 use_hook_recorder=False, hook_recorder_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = n_layers
        self.use_hook_recorder = use_hook_recorder and hook_recorder_fn is not None
        self.hook_recorder_fn = hook_recorder_fn
        self.activations = {}
        self.hooks = []
        self.hook_pattern = None  # e.g. "blocks.{layer_id}.mlp" — set after inspecting named_modules

    def set_hook_pattern(self, pattern: str):
        """Set the module name pattern, e.g. 'blocks.{layer_id}.mlp'."""
        self.hook_pattern = pattern

    def register_hooks(self):
        """Attach forward hooks to all N MLP layers. Return hook handles."""
        if self.use_hook_recorder:
            logger.info("Using built-in hook_recorder — no manual hooks needed")
            return
        if self.hook_pattern is None:
            raise ValueError(
                "hook_pattern not set. Inspect model.named_modules() and call "
                "set_hook_pattern() with the correct pattern before registering hooks."
            )
        module_dict = dict(self.model.named_modules())
        for i in range(self.n_layers):
            name = self.hook_pattern.format(layer_id=i)
            if name not in module_dict:
                logger.error(f"Could not find layer: {name}")
                continue
            handle = module_dict[name].register_forward_hook(self._make_hook(i))
            self.hooks.append(handle)
        logger.info(f"Registered {len(self.hooks)} manual hooks")

    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract_batch(self, prompts: list, token_pos: int = -1) -> list:
        """
        Run a single batched forward pass, return per-prompt activations.

        Args:
            prompts: list of code strings
            token_pos: which token to extract from (-1 = last non-pad token)

        Returns:
            list[dict[int, torch.Tensor]]: one dict per prompt
            Each tensor is shape [n_neurons] (1D)
        """
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, add_special_tokens=False,
        ).to(self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        batch_size = input_ids.shape[0]

        # Find last non-pad position per sequence
        if token_pos == -1:
            seq_lengths = attention_mask.sum(dim=1)  # [batch]
            last_positions = seq_lengths - 1          # 0-indexed
        else:
            last_positions = torch.full((batch_size,), token_pos, device=self.device)

        self.activations = {}
        with torch.no_grad():
            self.model(input_ids)

        results = []
        for b in range(batch_size):
            pos = last_positions[b].item()
            result = {}
            for lid, act in self.activations.items():
                result[lid] = act[b, pos, :].cpu()
            results.append(result)

        self.activations = {}
        return results

    def extract(self, prompt_text: str, token_pos: int = -1) -> dict:
        """
        Run a single forward pass, return activations at all layers.

        Args:
            prompt_text: the code string
            token_pos: which token to extract from (-1 = last)

        Returns:
            dict[int, torch.Tensor]: {layer_id: activation_vector}
            Each tensor is shape [n_neurons] (1D, the MLP expansion dim)
        """
        # add_special_tokens=False for code-completion style extraction
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        if self.use_hook_recorder:
            with self.hook_recorder_fn() as rec:
                with torch.no_grad():
                    self.model(inputs)
            result = {}
            for key, tensor in rec.items():
                if "mlp" in key and "act" in key:
                    layer_id = int(key.split(".")[0])
                    result[layer_id] = tensor[0, token_pos, :].cpu()
            return result
        else:
            self.activations = {}
            with torch.no_grad():
                self.model(inputs)
            result = {}
            for lid, act in self.activations.items():
                result[lid] = act[0, token_pos, :].cpu()
            self.activations = {}
            return result

    def _make_hook(self, layer_id: int):
        """Factory that returns a hook function capturing activations."""
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # out shape: [batch, seq_len, mlp_dim]
            self.activations[layer_id] = out.detach()
        return fn
