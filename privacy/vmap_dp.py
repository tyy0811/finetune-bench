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


def _calibrate_noise(
    epsilon: float,
    delta: float,
    sample_rate: float,
    epochs: int,
) -> float:
    """Use Opacus to calibrate noise_multiplier for a target (epsilon, delta)."""
    from opacus.accountants.utils import get_noise_multiplier

    return get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
    )


def train_dp_vmap(
    model: torch.nn.Module,
    loss_fn: callable,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    groups: dict[str, dict],
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    epsilon: float,
    delta: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
    class_weights: torch.Tensor | None = None,
    predict_fn: callable | None = None,
) -> dict:
    """Train with manual DP-SGD via vmap and per-group clipping.

    Uses Opacus only for noise calibration and privacy accounting,
    not for model wrapping.

    predict_fn: optional (model, *batch_inputs) -> logits. Defaults to model(*inputs).
    Needed for models with non-standard forward signatures (e.g. dict inputs).
    """
    from opacus.accountants import RDPAccountant

    from evaluation.metrics import compute_metrics

    torch.manual_seed(seed)

    if predict_fn is None:
        predict_fn = lambda m, *args: m(*args)

    model = model.to(device)
    model.eval()  # disable dropout — vmap can't trace data-dependent dropout

    # Split parameters
    trainable_params = {}
    frozen_params = {}
    for name, p in model.named_parameters():
        in_group = any(name in g["params"] for g in groups.values())
        if in_group and p.requires_grad:
            trainable_params[name] = p
        else:
            frozen_params[name] = p
    buffers = dict(model.named_buffers())

    # Calibrate noise
    sample_rate = batch_size / len(train_dataset)
    noise_multiplier = _calibrate_noise(epsilon, delta, sample_rate, epochs)

    # Privacy accountant
    accountant = RDPAccountant()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    epoch_losses: list[float] = []
    epoch_epsilons: list[float] = []

    for _epoch in range(epochs):
        total_loss = 0.0
        step_count = 0

        for batch in train_loader:
            *inputs, labels = [t.to(device) for t in batch]

            # 1. Per-sample gradients via vmap
            per_sample_grads = compute_per_sample_grads(
                loss_fn, trainable_params, frozen_params, buffers,
                *inputs, labels,
            )

            # 2. Per-group clipping
            clipped = clip_per_group(per_sample_grads, groups)

            # 3. Average across batch
            averaged = {name: g.mean(dim=0) for name, g in clipped.items()}

            # 4. Add calibrated noise per group
            noised = add_group_noise(averaged, groups, noise_multiplier, batch_size)

            # 5. SGD update
            with torch.no_grad():
                for name, param in trainable_params.items():
                    param -= lr * noised[name]

            # Track privacy
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

            # Compute loss for logging (forward only, no grad)
            with torch.no_grad():
                out = predict_fn(model, *inputs)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                loss = torch.nn.functional.cross_entropy(
                    out, labels, weight=class_weights,
                ).item()
            total_loss += loss
            step_count += 1

        epoch_losses.append(total_loss / max(step_count, 1))
        epoch_epsilons.append(accountant.get_epsilon(delta))

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            *inputs, labels = [t.to(device) for t in batch]
            out = predict_fn(model, *inputs)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    class_names = [str(i) for i in range(num_classes)]
    metrics = compute_metrics(all_labels, all_preds, class_names)

    return {
        "epsilon_target": epsilon,
        "epsilon_actual": accountant.get_epsilon(delta),
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "val_macro_f1": metrics.macro_f1,
        "val_accuracy": metrics.accuracy,
        "per_class_f1": metrics.per_class_f1.tolist(),
        "epoch_losses": epoch_losses,
        "epoch_epsilons": epoch_epsilons,
        "model_state_dict": model.state_dict(),
    }
