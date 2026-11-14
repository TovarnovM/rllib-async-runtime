"""Controlled factorized NoisyNet heads using RLlib's installed primitive.

RLlib 2.56.1's NoisyLayer samples on every call, including evaluation. This
small extension retains its parameter layout and factorization while making
noise lifetime explicit: once per training/exploration forward, independent
for online and target heads, and absent from deterministic inference.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer


class ControlledNoisyLayer(NoisyLayer):
    """Factorized Gaussian noise with learned scales and mean-only evaluation."""

    def __init__(self, in_size: int, out_size: int, sigma0: float) -> None:
        super().__init__(in_size, out_size, sigma0, activation=None)
        # Use the factorized NoisyNet initialization for both learned scales.
        # The installed primitive initializes sigma_w independently of sigma0.
        bound = in_size**-0.5
        torch.nn.init.uniform_(self.w, -bound, bound)
        torch.nn.init.uniform_(self.b, -bound, bound)
        torch.nn.init.constant_(self.sigma_w, sigma0 * bound)
        torch.nn.init.constant_(self.sigma_b, sigma0 * out_size**-0.5)
        self.register_buffer("epsilon_in", torch.zeros(in_size), persistent=False)
        self.register_buffer("epsilon_out", torch.zeros(out_size), persistent=False)
        self.noise_enabled = False

    def reset_noise(self) -> None:
        # Replace tensors instead of modifying them in place: a previous
        # forward's autograd graph may still need its original noise values.
        self.epsilon_in = self._f_epsilon(
            torch.randn(self.in_size, device=self.w.device, dtype=self.w.dtype)
        )
        self.epsilon_out = self._f_epsilon(
            torch.randn(self.out_size, device=self.w.device, dtype=self.w.dtype)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.noise_enabled:
            return inputs @ self.w + self.b
        noise = self.epsilon_in.unsqueeze(-1) * self.epsilon_out.unsqueeze(0)
        return (
            inputs @ (self.w + self.sigma_w * noise)
            + self.b
            + self.sigma_b * self.epsilon_out
        )


def install_noisy_heads(module: Any) -> None:
    """Replace only the DQN heads; observation encoders retain their catalog."""
    if not module.model_config.get("noisy", False):
        module._noisy_online_layers = ()
        module._noisy_target_layers = ()
        return
    sigma0 = module.model_config["std_init"]

    def replace(parent: torch.nn.Module) -> None:
        for name, child in tuple(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                if child.bias is None:
                    raise ValueError("Noisy DQN heads require biased linear layers")
                setattr(
                    parent,
                    name,
                    ControlledNoisyLayer(child.in_features, child.out_features, sigma0),
                )
            else:
                replace(child)

    replace(module.af)
    if module.uses_dueling:
        replace(module.vf)
    module._noisy_online_layers = tuple(
        layer
        for head in ((module.af, module.vf) if module.uses_dueling else (module.af,))
        for layer in head.modules()
        if isinstance(layer, ControlledNoisyLayer)
    )
    if not module._noisy_online_layers:
        raise ValueError("noisy=True requires supported linear DQN heads")
    module._noisy_target_layers = ()


def register_noisy_targets(module: Any) -> None:
    if not module._noisy_online_layers:
        return
    module._noisy_target_layers = tuple(
        layer
        for head in (
            (module._target_af, module._target_vf)
            if module.uses_dueling
            else (module._target_af,)
        )
        for layer in head.modules()
        if isinstance(layer, ControlledNoisyLayer)
    )


def noisy_forward(mode: str):
    """Apply one noise realization across all heads used by a module forward."""

    def decorate(method):
        @wraps(method)
        def forward(self, *args, **kwargs):
            layers = self._noisy_online_layers
            if mode == "train":
                layers += self._noisy_target_layers
            enabled = mode != "inference"
            previous = tuple(layer.noise_enabled for layer in layers)
            try:
                for layer in layers:
                    layer.noise_enabled = enabled
                    if enabled:
                        layer.reset_noise()
                return method(self, *args, **kwargs)
            finally:
                for layer, was_enabled in zip(layers, previous, strict=True):
                    layer.noise_enabled = was_enabled

        return forward

    return decorate
