"""Tests for privacy.vmap_dp — manual DP-SGD via vmap with per-group clipping."""

import pytest
import torch
import torch.nn as nn


class TestPerGroupClipping:
    """Verify per-group clipping clips each group independently."""

    def test_clip_reduces_norm_to_max(self):
        from privacy.vmap_dp import clip_per_group

        # Two groups: "a" has large grads, "b" has small grads
        per_sample_grads = {
            "a.weight": torch.randn(4, 10) * 10,  # large
            "b.weight": torch.randn(4, 5) * 0.01,  # small
        }
        groups = {
            "a": {"params": ["a.weight"], "clip_norm": 1.0},
            "b": {"params": ["b.weight"], "clip_norm": 0.1},
        }
        clipped = clip_per_group(per_sample_grads, groups)

        # Each sample's group norm should be <= clip_norm
        a_norms = clipped["a.weight"].flatten(1).norm(2, dim=1)
        b_norms = clipped["b.weight"].flatten(1).norm(2, dim=1)
        assert (a_norms <= 1.0 + 1e-6).all()
        assert (b_norms <= 0.1 + 1e-6).all()

    def test_small_grads_not_clipped(self):
        from privacy.vmap_dp import clip_per_group

        per_sample_grads = {
            "w": torch.randn(4, 5) * 0.001,
        }
        groups = {"g": {"params": ["w"], "clip_norm": 1.0}}
        clipped = clip_per_group(per_sample_grads, groups)

        # Small grads should pass through unchanged
        assert torch.allclose(clipped["w"], per_sample_grads["w"], atol=1e-7)

    def test_independent_clipping(self):
        """Clipping group A should not affect group B magnitudes."""
        from privacy.vmap_dp import clip_per_group

        per_sample_grads = {
            "a.w": torch.ones(2, 10) * 100,  # will be clipped hard
            "b.w": torch.ones(2, 5) * 0.01,  # should be untouched
        }
        groups = {
            "a": {"params": ["a.w"], "clip_norm": 1.0},
            "b": {"params": ["b.w"], "clip_norm": 1.0},
        }
        clipped = clip_per_group(per_sample_grads, groups)

        # b should be unchanged (norm 0.022 < 1.0)
        assert torch.allclose(clipped["b.w"], per_sample_grads["b.w"], atol=1e-7)


class TestNoiseAddition:
    def test_noise_magnitude(self):
        """Noise std should equal clip_norm * noise_multiplier / batch_size."""
        from privacy.vmap_dp import add_group_noise

        torch.manual_seed(0)
        grads = {"w": torch.zeros(100)}  # zero grads so we measure pure noise
        groups = {"g": {"params": ["w"], "clip_norm": 2.0}}
        noised = add_group_noise(grads, groups, noise_multiplier=1.0, batch_size=10, seed=0)

        # Expected std: 2.0 * 1.0 / 10 = 0.2
        actual_std = noised["w"].std().item()
        assert 0.1 < actual_std < 0.3, f"Expected ~0.2, got {actual_std}"

    def test_different_groups_different_noise(self):
        """Groups with different clip_norms get different noise scales."""
        from privacy.vmap_dp import add_group_noise

        grads = {"a": torch.zeros(1000), "b": torch.zeros(1000)}
        groups = {
            "big": {"params": ["a"], "clip_norm": 10.0},
            "small": {"params": ["b"], "clip_norm": 0.1},
        }
        noised = add_group_noise(grads, groups, noise_multiplier=1.0, batch_size=1, seed=42)
        assert noised["a"].std().item() > noised["b"].std().item() * 10


class TestVmapPerSampleGrad:
    """Test that vmap computes per-sample gradients correctly."""

    def test_matches_sequential_grad(self):
        """Per-sample grads from vmap should match manual loop."""
        from privacy.vmap_dp import compute_per_sample_grads

        model = nn.Linear(5, 3, bias=False)
        model.eval()

        inputs = torch.randn(4, 5)
        labels = torch.randint(0, 3, (4,))

        trainable = dict(model.named_parameters())
        frozen = {}
        buffers = dict(model.named_buffers())

        def loss_fn(params, frozen, bufs, x, y):
            out = torch.func.functional_call(model, (params, bufs), (x.unsqueeze(0),))
            return nn.functional.cross_entropy(out, y.unsqueeze(0))

        vmap_grads = compute_per_sample_grads(
            loss_fn, trainable, frozen, buffers, inputs, labels,
        )

        # Manual sequential computation
        for i in range(4):
            model.zero_grad()
            out = model(inputs[i:i+1])
            loss = nn.functional.cross_entropy(out, labels[i:i+1])
            loss.backward()
            manual_grad = model.weight.grad.clone()
            vmap_grad = vmap_grads["weight"][i]
            assert torch.allclose(manual_grad, vmap_grad, atol=1e-5), (
                f"Sample {i}: max diff = {(manual_grad - vmap_grad).abs().max()}"
            )


class TestTrainDpVmap:
    """Integration test: vmap DP training on a simple model."""

    def test_train_dp_vmap_returns_metrics(self):
        from privacy.vmap_dp import train_dp_vmap

        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 10), torch.randint(0, 3, (32,))
        )
        val = torch.utils.data.TensorDataset(
            torch.randn(8, 10), torch.randint(0, 3, (8,))
        )

        def loss_fn(params, frozen, bufs, x, y):
            out = torch.func.functional_call(model, (params, bufs), (x.unsqueeze(0),))
            return nn.functional.cross_entropy(out, y.unsqueeze(0))

        groups = {
            "all": {
                "params": [n for n, _ in model.named_parameters()],
                "clip_norm": 1.0,
            },
        }

        result = train_dp_vmap(
            model=model,
            loss_fn=loss_fn,
            train_dataset=dataset,
            val_dataset=val,
            groups=groups,
            num_classes=3,
            epochs=2,
            batch_size=8,
            lr=0.01,
            epsilon=50.0,
            delta=1e-5,
            device="cpu",
        )
        assert "epsilon_actual" in result
        assert result["epsilon_actual"] > 0
        assert "val_macro_f1" in result
        assert "epoch_losses" in result
        assert len(result["epoch_losses"]) == 2

    def test_stricter_epsilon_changes_result(self):
        from privacy.vmap_dp import train_dp_vmap

        model = nn.Sequential(nn.Linear(10, 3))
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 10), torch.randint(0, 3, (32,))
        )
        val = torch.utils.data.TensorDataset(
            torch.randn(8, 10), torch.randint(0, 3, (8,))
        )

        def loss_fn(params, frozen, bufs, x, y):
            out = torch.func.functional_call(model, ({**params, **bufs}, {}), (x.unsqueeze(0),))
            return nn.functional.cross_entropy(out, y.unsqueeze(0))

        groups = {"all": {"params": [n for n, _ in model.named_parameters()], "clip_norm": 1.0}}

        r1 = train_dp_vmap(model=model, loss_fn=loss_fn, train_dataset=dataset,
                           val_dataset=val, groups=groups, num_classes=3,
                           epochs=2, batch_size=8, lr=0.01, epsilon=50.0,
                           delta=1e-5, device="cpu")
        r2 = train_dp_vmap(model=model, loss_fn=loss_fn, train_dataset=dataset,
                           val_dataset=val, groups=groups, num_classes=3,
                           epochs=2, batch_size=8, lr=0.01, epsilon=2.0,
                           delta=1e-5, device="cpu")
        # Different noise → different loss trajectories
        assert r1["epoch_losses"] != r2["epoch_losses"]
