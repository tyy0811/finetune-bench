"""Manual DP-SGD via torch.func.vmap with per-group gradient clipping.

Bypasses Opacus's GradSampleModule to support per-group clipping:
different parameter groups (e.g. LoRA vs head) get independent clip
norms and noise scales. This solves the gradient heterogeneity problem
where global clipping kills small-gradient parameters.
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap


def compute_per_sample_grads(
    loss_fn: callable,
    trainable_params: dict[str, torch.Tensor],
    frozen_params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    *batch_inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-sample gradients via vmap.

    loss_fn signature: (trainable_params, frozen_params, buffers, *per_sample_inputs) -> scalar
    Returns dict mapping param name -> (batch_size, *param_shape) gradient tensor.
    """
    ft_grad = grad(loss_fn, argnums=0)
    # vmap over batch dimension of all inputs (trainable/frozen/buffers are not batched)
    in_dims = (None, None, None) + (0,) * len(batch_inputs)
    ft_vmap = vmap(ft_grad, in_dims=in_dims)
    return ft_vmap(trainable_params, frozen_params, buffers, *batch_inputs)


def clip_per_group(
    per_sample_grads: dict[str, torch.Tensor],
    groups: dict[str, dict],
) -> dict[str, torch.Tensor]:
    """Clip per-sample gradients independently per group.

    Each group has a list of param names and a clip_norm. Per-sample
    gradients within each group are clipped to the group's L2 norm bound.

    groups format: {"group_name": {"params": ["name1", ...], "clip_norm": float}}
    """
    clipped = {}
    batch_size = next(iter(per_sample_grads.values())).shape[0]

    for group_cfg in groups.values():
        param_names = group_cfg["params"]
        clip_norm = group_cfg["clip_norm"]

        # Compute per-sample L2 norm across all params in this group
        per_sample_sq = torch.zeros(batch_size, device=next(iter(per_sample_grads.values())).device)
        for name in param_names:
            per_sample_sq += per_sample_grads[name].flatten(1).norm(2, dim=1) ** 2
        per_sample_norm = per_sample_sq.sqrt()

        # Clip factor: min(1, clip_norm / norm) per sample
        clip_factor = (clip_norm / per_sample_norm).clamp(max=1.0)

        for name in param_names:
            g = per_sample_grads[name]
            expanded = clip_factor.view(batch_size, *([1] * (g.dim() - 1)))
            clipped[name] = g * expanded

    return clipped


def add_group_noise(
    averaged_grads: dict[str, torch.Tensor],
    groups: dict[str, dict],
    noise_multiplier: float,
    batch_size: int,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Add calibrated Gaussian noise to averaged gradients, per group.

    Noise std for each group = group.clip_norm * noise_multiplier / batch_size.
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    noised = {}
    for group_cfg in groups.values():
        sigma = group_cfg["clip_norm"] * noise_multiplier / batch_size
        for name in group_cfg["params"]:
            noise = torch.randn(averaged_grads[name].shape, generator=gen) * sigma
            noised[name] = averaged_grads[name] + noise.to(averaged_grads[name].device)
    return noised
