"""Tests for privacy.dp_training module."""

import numpy as np
import pytest
import torch

from privacy.dp_training import create_dp_training_components, make_dp_config, train_dp


class TestDpConfig:
    def test_amp_disabled(self):
        dp_config = make_dp_config(epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
        assert dp_config["use_amp"] is False

    def test_grad_accumulation_is_one(self):
        dp_config = make_dp_config(epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
        assert dp_config["grad_accumulation_steps"] == 1

    def test_epsilon_stored(self):
        dp_config = make_dp_config(epsilon=1.0, delta=1e-5, max_grad_norm=0.5)
        assert dp_config["epsilon"] == 1.0
        assert dp_config["delta"] == 1e-5
        assert dp_config["max_grad_norm"] == 0.5


class TestDpTrainingComponents:
    """Test that Opacus wraps model/optimizer/dataloader correctly.

    These tests require opacus to be installed. Skip if not available.
    """

    @pytest.fixture
    def _small_components(self):
        """Minimal model, optimizer, dataloader for Opacus testing."""
        pytest.importorskip("opacus")

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = torch.utils.data.TensorDataset(
            torch.randn(32, 10), torch.randint(0, 3, (32,))
        )
        loader = torch.utils.data.DataLoader(data, batch_size=8)
        return model, optimizer, loader

    def test_privacy_engine_attaches(self, _small_components):
        opacus = pytest.importorskip("opacus")
        model, optimizer, loader = _small_components

        dp_model, dp_optimizer, dp_loader, _engine = create_dp_training_components(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=3,
            epsilon=8.0,
            delta=1e-5,
            max_grad_norm=1.0,
        )
        # Opacus wraps model in GradSampleModule
        assert isinstance(dp_model, opacus.GradSampleModule)

    def test_dp_model_trains_one_step(self, _small_components):
        pytest.importorskip("opacus")
        model, optimizer, loader = _small_components

        dp_model, dp_optimizer, dp_loader, _engine = create_dp_training_components(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=3,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
        )
        # One forward-backward-step cycle should not crash
        batch_x, batch_y = next(iter(dp_loader))
        logits = dp_model(batch_x)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        dp_optimizer.step()
        dp_optimizer.zero_grad()

    def test_privacy_budget_consumed(self, _small_components):
        pytest.importorskip("opacus")
        model, optimizer, loader = _small_components

        dp_model, dp_optimizer, dp_loader, engine = create_dp_training_components(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=3,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
        )
        # Train one full epoch
        for batch_x, batch_y in dp_loader:
            if batch_x.shape[0] == 0:
                continue
            logits = dp_model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            dp_optimizer.step()
            dp_optimizer.zero_grad()

        eps = engine.get_epsilon(1e-5)
        assert eps > 0  # some budget consumed


class TestTrainDp:
    def test_train_dp_returns_expected_keys(self):
        """Integration test: train_dp on tiny synthetic data returns correct output shape."""
        pytest.importorskip("opacus")

        result = train_dp(
            model_class=torch.nn.Sequential,
            model_args=(torch.nn.Linear(10, 3),),
            train_dataset=torch.utils.data.TensorDataset(
                torch.randn(32, 10), torch.randint(0, 3, (32,))
            ),
            val_dataset=torch.utils.data.TensorDataset(
                torch.randn(8, 10), torch.randint(0, 3, (8,))
            ),
            num_classes=3,
            epochs=1,
            batch_size=8,
            lr=0.01,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
            device="cpu",
        )
        assert "epsilon_actual" in result
        assert "val_macro_f1" in result
        assert "train_loss" in result
        assert result["epsilon_actual"] > 0

    def test_train_dp_noise_affects_output(self):
        """Verify DP training produces different results from non-DP (noise is added)."""
        pytest.importorskip("opacus")

        torch.manual_seed(42)
        data = torch.utils.data.TensorDataset(
            torch.randn(32, 10), torch.randint(0, 3, (32,))
        )
        val = torch.utils.data.TensorDataset(
            torch.randn(8, 10), torch.randint(0, 3, (8,))
        )

        # Train with very loose DP (eps=50) and very strict DP (eps=0.1)
        r_loose = train_dp(
            model_class=torch.nn.Sequential,
            model_args=(torch.nn.Linear(10, 3),),
            train_dataset=data, val_dataset=val, num_classes=3,
            epochs=2, batch_size=8, lr=0.01,
            epsilon=50.0, delta=1e-5, max_grad_norm=1.0, device="cpu",
        )
        r_strict = train_dp(
            model_class=torch.nn.Sequential,
            model_args=(torch.nn.Linear(10, 3),),
            train_dataset=data, val_dataset=val, num_classes=3,
            epochs=2, batch_size=8, lr=0.01,
            epsilon=0.1, delta=1e-5, max_grad_norm=1.0, device="cpu",
        )
        # Not asserting which is better — just that they differ
        assert r_loose["train_loss"] != r_strict["train_loss"]


class TestTrainDpMultimodal:
    """Integration test with the real MultimodalClassifier + ComplaintDataset.

    Uses a tiny subset (32 train, 8 val) to validate the full Opacus wrapping
    pipeline against the actual model architecture (DistilBERT + tabular MLP).
    """

    def test_multimodal_dp_training_completes(self):
        pytest.importorskip("opacus")

        import numpy as np
        from transformers import DistilBertTokenizer

        from models.fusion_model import MultimodalClassifier
        from training.train import ComplaintDataset

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        num_classes = 3
        tabular_dim = 5

        # Tiny synthetic data matching ComplaintDataset format
        train_narratives = [f"complaint about issue {i}" for i in range(32)]
        val_narratives = [f"different complaint {i}" for i in range(8)]
        train_tabular = np.random.randn(32, tabular_dim).astype(np.float32)
        val_tabular = np.random.randn(8, tabular_dim).astype(np.float32)
        train_labels = np.random.randint(0, num_classes, 32)
        val_labels = np.random.randint(0, num_classes, 8)

        train_ds = ComplaintDataset(
            train_narratives, train_tabular, train_labels, tokenizer, max_length=32,
        )
        val_ds = ComplaintDataset(
            val_narratives, val_tabular, val_labels, tokenizer, max_length=32,
        )

        # Flatten for Opacus (same approach as modal_privacy.py)
        flat_train = torch.utils.data.TensorDataset(
            train_ds.encodings["input_ids"],
            train_ds.encodings["attention_mask"],
            train_ds.tabular_features,
            train_ds.labels,
        )
        flat_val = torch.utils.data.TensorDataset(
            val_ds.encodings["input_ids"],
            val_ds.encodings["attention_mask"],
            val_ds.tabular_features,
            val_ds.labels,
        )

        # Opacus-compatible wrapper (same as modal_privacy.py):
        # - Flat tensor forward signature for Opacus hooks
        # - Frozen encoder (Opacus 1.4 can't handle transformer per-sample grads)
        class OpacusWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = MultimodalClassifier(
                    num_classes=num_classes,
                    tabular_input_dim=tabular_dim,
                    text_model_name="distilbert-base-uncased",
                )
                for p in self.inner.text_encoder.parameters():
                    p.requires_grad = False

            def forward(self, input_ids, attention_mask, tabular):
                text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                return self.inner(text_inputs, tabular)

        result = train_dp(
            model_class=OpacusWrapper,
            model_args=(),
            train_dataset=flat_train,
            val_dataset=flat_val,
            num_classes=num_classes,
            epochs=1,
            batch_size=8,
            lr=2e-5,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
            device="cpu",
        )

        assert "epsilon_actual" in result
        assert result["epsilon_actual"] > 0
        assert "val_macro_f1" in result
        assert "model_state_dict" in result
        assert result["train_loss"] > 0


class TestDpTransformersIntegration:
    """Test full-model DP training with dp-transformers layer conversion."""

    def test_convert_encoder_and_train(self):
        """Full-model DP-SGD via dp-transformers produces valid results with budget consumed."""
        pytest.importorskip("opacus")
        pytest.importorskip("dp_transformers")

        from dp_transformers.module_modification import convert_model_to_dp
        from transformers import DistilBertModel, DistilBertTokenizer

        from models.fusion_model import MultimodalClassifier
        from training.train import ComplaintDataset

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        num_classes = 3
        tabular_dim = 5

        train_narratives = [f"complaint about issue {i}" for i in range(32)]
        val_narratives = [f"different complaint {i}" for i in range(8)]
        train_tabular = np.random.randn(32, tabular_dim).astype(np.float32)
        val_tabular = np.random.randn(8, tabular_dim).astype(np.float32)
        train_labels = np.random.randint(0, num_classes, 32)
        val_labels = np.random.randint(0, num_classes, 8)

        train_ds = ComplaintDataset(
            train_narratives, train_tabular, train_labels, tokenizer, max_length=32,
        )
        val_ds = ComplaintDataset(
            val_narratives, val_tabular, val_labels, tokenizer, max_length=32,
        )

        flat_train = torch.utils.data.TensorDataset(
            train_ds.encodings["input_ids"],
            train_ds.encodings["attention_mask"],
            train_ds.tabular_features,
            train_ds.labels,
        )
        flat_val = torch.utils.data.TensorDataset(
            val_ds.encodings["input_ids"],
            val_ds.encodings["attention_mask"],
            val_ds.tabular_features,
            val_ds.labels,
        )

        # Approach B: convert encoder first, then inject into model
        encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        encoder = convert_model_to_dp(encoder)

        class OpacusWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = MultimodalClassifier(
                    num_classes=num_classes,
                    tabular_input_dim=tabular_dim,
                    text_model_name="distilbert-base-uncased",
                    text_encoder=encoder,
                )
                # No freeze — full model is DP-trainable

            def forward(self, input_ids, attention_mask, tabular):
                text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                return self.inner(text_inputs, tabular)

        result = train_dp(
            model_class=OpacusWrapper,
            model_args=(),
            train_dataset=flat_train,
            val_dataset=flat_val,
            num_classes=num_classes,
            epochs=1,
            batch_size=8,
            lr=2e-5,
            epsilon=50.0,
            delta=1e-5,
            max_grad_norm=1.0,
            device="cpu",
        )

        assert result["epsilon_actual"] > 0
        assert "val_macro_f1" in result
        assert result["train_loss"] > 0
