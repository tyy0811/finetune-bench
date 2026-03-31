# GPU Profiling & Mixed-Precision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add mixed-precision (fp16) training with `torch.cuda.amp`, GPU memory profiling, and ONNX fp16 export to finetune-bench.

**Architecture:** GPUProfiler hooks into the existing training loop via three callbacks (`on_epoch_start`, `on_step_end`, `on_epoch_end`). AMP integrates via `GradScaler` + `autocast` wrapping the existing forward/loss/backward paths, including the tail-end incomplete accumulation handler. All features degrade gracefully on CPU (CI-safe).

**Tech Stack:** PyTorch 2.2.2 (`torch.cuda.amp.GradScaler`, `torch.amp.autocast`), onnxruntime (existing dep), MLflow (existing), Pydantic (existing `TrainConfig`)

**Design doc:** `docs/plans/2026-03-30-gpu-profiling-mixed-precision-design.md`

---

## Task 1: GPUProfiler Class

**Files:**
- Create: `training/gpu_profiler.py`
- Test: `tests/test_gpu_profiler.py`

**Step 1: Write the failing tests**

Create `tests/test_gpu_profiler.py`:

```python
"""Tests for GPU memory profiler — all CPU-safe."""

from training.gpu_profiler import GPUProfiler


def test_profiler_disabled_returns_empty_summary():
    """GPUProfiler(enabled=False) produces {"gpu_available": False}."""
    profiler = GPUProfiler(enabled=False)
    profiler.on_epoch_start(0)
    profiler.on_step_end(0, 0)
    profiler.on_step_end(0, 50)
    profiler.on_epoch_end(0)
    assert profiler.summary() == {"gpu_available": False}
    assert profiler.snapshots == []


def test_profiler_sampling_interval():
    """Disabled profiler records no snapshots regardless of step."""
    profiler = GPUProfiler(enabled=False)
    for step in range(100):
        profiler.on_step_end(0, step, sample_every=25)
    assert len(profiler.snapshots) == 0


def test_profiler_epoch_timing():
    """Disabled profiler does not record epoch times."""
    profiler = GPUProfiler(enabled=False)
    profiler.on_epoch_start(0)
    profiler.on_epoch_end(0)
    assert profiler.epoch_times == []
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_profiler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.gpu_profiler'`

**Step 3: Write the implementation**

Create `training/gpu_profiler.py`:

```python
"""GPU memory and utilization tracking for training runs."""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class GPUSnapshot:
    """Single point-in-time GPU measurement."""

    epoch: int
    step: int
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    timestamp: float = field(default_factory=time.time)


class GPUProfiler:
    """Collects GPU metrics during training.

    Usage:
        profiler = GPUProfiler(device=0)
        for epoch in range(n_epochs):
            profiler.on_epoch_start(epoch)
            for step, batch in enumerate(loader):
                # ... training step ...
                profiler.on_step_end(epoch, step)
            profiler.on_epoch_end(epoch)
        report = profiler.summary()
    """

    def __init__(self, device: int = 0, enabled: bool = True):
        self.device = device
        self.enabled = enabled and torch.cuda.is_available()
        self.snapshots: list[GPUSnapshot] = []
        self._epoch_start_time: float = 0.0
        self.epoch_times: list[float] = []

    def on_epoch_start(self, epoch: int):
        if not self.enabled:
            return
        torch.cuda.reset_peak_memory_stats(self.device)
        self._epoch_start_time = time.time()

    def on_step_end(self, epoch: int, step: int, sample_every: int = 50):
        """Sample GPU state every N steps to avoid overhead."""
        if not self.enabled or step % sample_every != 0:
            return
        self.snapshots.append(
            GPUSnapshot(
                epoch=epoch,
                step=step,
                allocated_mb=torch.cuda.memory_allocated(self.device) / 1e6,
                reserved_mb=torch.cuda.memory_reserved(self.device) / 1e6,
                max_allocated_mb=torch.cuda.max_memory_allocated(self.device) / 1e6,
            )
        )

    def on_epoch_end(self, epoch: int):
        if not self.enabled:
            return
        elapsed = time.time() - self._epoch_start_time
        self.epoch_times.append(elapsed)

    def summary(self) -> dict:
        """Return summary dict suitable for MLflow logging."""
        if not self.enabled or not self.snapshots:
            return {"gpu_available": False}

        peak = max(s.max_allocated_mb for s in self.snapshots)
        mean_alloc = sum(s.allocated_mb for s in self.snapshots) / len(self.snapshots)
        gpu_name = torch.cuda.get_device_name(self.device)
        gpu_total_mb = torch.cuda.get_device_properties(self.device).total_mem / 1e6

        return {
            "gpu_available": True,
            "gpu_name": gpu_name,
            "gpu_total_mb": round(gpu_total_mb, 0),
            "gpu_peak_allocated_mb": round(peak, 1),
            "gpu_mean_allocated_mb": round(mean_alloc, 1),
            "gpu_peak_utilization_pct": round(peak / gpu_total_mb * 100, 1),
            "epoch_mean_time_s": round(
                sum(self.epoch_times) / len(self.epoch_times), 1
            ),
            "total_training_time_s": round(sum(self.epoch_times), 1),
        }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gpu_profiler.py -v`
Expected: 3 passed

**Step 5: Lint**

Run: `ruff check training/gpu_profiler.py tests/test_gpu_profiler.py`
Expected: clean

**Step 6: Commit**

```bash
git add training/gpu_profiler.py tests/test_gpu_profiler.py
git commit -m "feat: add GPUProfiler class for memory tracking during training"
```

---

## Task 2: Integrate GPUProfiler into Training Loop

**Files:**
- Modify: `training/train.py:1-10` (imports), `101-151` (train_one_epoch signature + body), `185-358` (train function)

**Step 1: Add profiler parameter to train_one_epoch**

In `training/train.py`, add the import at line 10 (after existing imports):

```python
from training.gpu_profiler import GPUProfiler
```

Change the `train_one_epoch` function signature at line 101-110 to add `profiler` parameter:

```python
def train_one_epoch(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    class_weights: torch.Tensor,
    config: TrainConfig,
    epoch: int,
    device: torch.device,
    profiler: GPUProfiler | None = None,
) -> float:
```

Add `profiler.on_step_end(epoch, step)` call after the MLflow logging at line 140, inside the batch loop:

```python
        global_step = epoch * len(train_loader) + step
        mlflow.log_metric("train_loss_step", batch_loss, step=global_step)

        if profiler is not None:
            profiler.on_step_end(epoch, step)
```

**Step 2: Integrate profiler into train() function**

In the `train()` function, after `early_stopper = EarlyStopper(...)` at line 283, add:

```python
    profiler = GPUProfiler(enabled=torch.cuda.is_available())
```

Wrap the epoch loop (lines 295-326) with profiler hooks — add `profiler.on_epoch_start(epoch)` before `train_one_epoch` and `profiler.on_epoch_end(epoch)` after it:

```python
        for epoch in range(config.num_epochs):
            profiler.on_epoch_start(epoch)
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                class_weights, config, epoch, device,
                profiler=profiler,
            )
            profiler.on_epoch_end(epoch)
            val_metrics = evaluate(model, val_loader, adapter.class_names, device)
```

After the test metrics logging (after line 341), add GPU summary logging:

```python
        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

        # Log GPU profiling metrics
        gpu_summary = profiler.summary()
        for key, value in gpu_summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"gpu/{key}", value)
```

**Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v --ignore=tests/test_overfit_smoke.py`
Expected: all pass (skip overfit smoke — it trains a real model and is slow)

**Step 4: Commit**

```bash
git add training/train.py
git commit -m "feat: integrate GPUProfiler into training loop with MLflow logging"
```

---

## Task 3: Add use_amp to TrainConfig

**Files:**
- Modify: `training/config.py:37` (after `early_stopping_patience`)
- Test: `tests/test_amp_config.py`

**Step 1: Write the failing tests**

Create `tests/test_amp_config.py`:

```python
"""Tests for AMP configuration — all CPU-safe."""

import torch

from training.config import TrainConfig


def test_use_amp_default_false():
    """use_amp defaults to False to preserve existing fp32 behavior."""
    config = TrainConfig()
    assert config.use_amp is False


def test_use_amp_serializes_to_dict():
    """use_amp appears in model_dump() for MLflow param logging."""
    config = TrainConfig(use_amp=True)
    params = config.model_dump()
    assert params["use_amp"] is True


def test_gradscaler_disabled_is_noop():
    """GradScaler with enabled=False passes tensors through unchanged."""
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    t = torch.tensor(1.0)
    assert scaler.scale(t) == t
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amp_config.py -v`
Expected: FAIL — `test_use_amp_default_false` fails with `AttributeError` (field doesn't exist yet)

**Step 3: Add use_amp field to TrainConfig**

In `training/config.py`, after line 37 (`early_stopping_patience: int = 2`), add:

```python
    use_amp: bool = False  # Enable mixed-precision (fp16) training
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amp_config.py -v`
Expected: 3 passed

**Step 5: Run full test suite to verify no regression**

Run: `python -m pytest tests/ -v --ignore=tests/test_overfit_smoke.py`
Expected: all pass

**Step 6: Commit**

```bash
git add training/config.py tests/test_amp_config.py
git commit -m "feat: add use_amp config field (default False) for mixed-precision training"
```

---

## Task 4: AMP Integration in Training Loop

**Files:**
- Modify: `training/train.py:101-151` (train_one_epoch), `training/train.py:154-182` (evaluate), `training/train.py:185-358` (train)

**Step 1: Add AMP imports and device_type detection in train()**

In `training/train.py`, add to imports (after `import torch` at line 13):

```python
from torch.amp import autocast
from torch.cuda.amp import GradScaler
```

In the `train()` function, after `device = torch.device(...)` at line 188, add device_type detection:

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AMP: determine device type and disable if not CUDA
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"
        config.use_amp = False
```

After the `profiler = GPUProfiler(...)` line (added in Task 2), add scaler:

```python
    profiler = GPUProfiler(enabled=torch.cuda.is_available())
    scaler = GradScaler(enabled=config.use_amp)
```

**Step 2: Modify train_one_epoch to accept scaler and device_type**

Update the `train_one_epoch` signature to add `scaler` and `device_type`:

```python
def train_one_epoch(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    class_weights: torch.Tensor,
    config: TrainConfig,
    epoch: int,
    device: torch.device,
    profiler: GPUProfiler | None = None,
    scaler: GradScaler | None = None,
    device_type: str = "cpu",
) -> float:
```

**Step 3: Wrap forward+loss in autocast, use scaler for backward/step**

Replace the training loop body (lines 116-149) with the AMP version. The full replacement for `train_one_epoch`:

```python
def train_one_epoch(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    class_weights: torch.Tensor,
    config: TrainConfig,
    epoch: int,
    device: torch.device,
    profiler: GPUProfiler | None = None,
    scaler: GradScaler | None = None,
    device_type: str = "cpu",
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    use_scaler = scaler is not None

    for step, batch in enumerate(train_loader):
        text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
        tabular_features = batch["tabular"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type=device_type, enabled=config.use_amp):
            logits = model(text_inputs, tabular_features)
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
            loss = loss / config.grad_accumulation_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config.grad_accumulation_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * config.grad_accumulation_steps
        epoch_loss += batch_loss
        num_batches += 1

        global_step = epoch * len(train_loader) + step
        mlflow.log_metric("train_loss_step", batch_loss, step=global_step)

        if profiler is not None:
            profiler.on_step_end(epoch, step)

    # Flush any remaining accumulated gradients from incomplete final batch
    if (step + 1) % config.grad_accumulation_steps != 0:
        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_grad_norm
        )
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return epoch_loss / max(num_batches, 1)
```

**Step 4: Add autocast to evaluate()**

In the `evaluate()` function (line 154), add `device_type` and `use_amp` parameters:

```python
def evaluate(
    model: MultimodalClassifier,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    return_probs: bool = False,
    device_type: str = "cpu",
    use_amp: bool = False,
) -> MetricsResult | tuple[MetricsResult, np.ndarray]:
```

Wrap the forward pass inside the `torch.no_grad()` block:

```python
    with torch.no_grad():
        for batch in loader:
            text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
            tabular_features = batch["tabular"].to(device)
            with autocast(device_type=device_type, enabled=use_amp):
                logits = model(text_inputs, tabular_features)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
            if return_probs:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
```

**Step 5: Update train() to pass scaler/device_type to train_one_epoch and evaluate**

Update the `train_one_epoch` call inside the epoch loop:

```python
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                class_weights, config, epoch, device,
                profiler=profiler, scaler=scaler, device_type=device_type,
            )
```

Update all `evaluate()` calls to pass `device_type` and `use_amp`:

```python
            val_metrics = evaluate(
                model, val_loader, adapter.class_names, device,
                device_type=device_type, use_amp=config.use_amp,
            )
```

```python
        test_metrics = evaluate(
            model, test_loader, adapter.class_names, device,
            device_type=device_type, use_amp=config.use_amp,
        )
```

**Step 6: Update CLI argparse at bottom of train.py**

In the `if __name__ == "__main__"` block (line 361), add `--use-amp` argument:

```python
    parser.add_argument("--use-amp", action="store_true", help="Enable fp16 mixed-precision")
    args = parser.parse_args()

    cfg = TrainConfig(
        variant=args.variant,
        seed=args.seed,
        sample_size=args.sample_size,
        num_epochs=args.epochs,
        use_amp=args.use_amp,
    )
```

**Step 7: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v --ignore=tests/test_overfit_smoke.py`
Expected: all pass — AMP is disabled on CPU, so all existing behavior is identical

**Step 8: Lint**

Run: `ruff check training/train.py`
Expected: clean

**Step 9: Commit**

```bash
git add training/train.py
git commit -m "feat: integrate AMP (GradScaler + autocast) into training and evaluation loops"
```

---

## Task 5: ONNX fp16 Conversion

**Files:**
- Modify: `evaluation/export.py` (add `convert_onnx_to_fp16` function after line 59)
- Modify: `tests/test_export.py` (add fp16 conversion test)

**Step 1: Write the failing test**

Add to `tests/test_export.py`:

```python
from evaluation.export import convert_onnx_to_fp16


def test_onnx_fp16_conversion():
    """fp16 ONNX model is smaller than fp32 and runs inference."""
    import onnxruntime as ort

    torch.manual_seed(42)
    model = MultimodalClassifier(
        num_classes=5,
        tabular_input_dim=20,
        tabular_hidden_dim=32,
        tabular_embed_dim=16,
        fusion_hidden_dim=32,
    )
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = os.path.join(tmpdir, "model.onnx")
        fp16_path = os.path.join(tmpdir, "model_fp16.onnx")

        export_to_onnx(model, fp32_path, tabular_dim=20, seq_length=16)
        convert_onnx_to_fp16(fp32_path, fp16_path)

        assert os.path.exists(fp16_path)
        assert os.path.getsize(fp16_path) < os.path.getsize(fp32_path)

        # Verify inference runs and output shape is correct
        session = ort.InferenceSession(fp16_path)
        batch_size = 2
        seq_len = 16
        result = session.run(
            None,
            {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).numpy(),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long).numpy(),
                "tabular_features": torch.randn(batch_size, 20).numpy(),
            },
        )[0]
        assert result.shape == (batch_size, 5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py::test_onnx_fp16_conversion -v`
Expected: FAIL — `ImportError: cannot import name 'convert_onnx_to_fp16'`

**Step 3: Write the implementation**

Add to `evaluation/export.py` after the `export_to_onnx` function (after line 59):

```python
def convert_onnx_to_fp16(input_path: str, output_path: str) -> None:
    """Convert fp32 ONNX model to fp16, keeping fp32 IO for compatibility."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model_fp32 = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, output_path)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export.py -v`
Expected: 3 passed (2 existing + 1 new)

**Step 5: Lint**

Run: `ruff check evaluation/export.py tests/test_export.py`
Expected: clean

**Step 6: Commit**

```bash
git add evaluation/export.py tests/test_export.py
git commit -m "feat: add ONNX fp16 conversion with keep_io_types for deployment"
```

---

## Task 6: Update Modal Script (A10G + AMP Comparison)

**Files:**
- Modify: `scripts/modal_run.py`

**Step 1: Upgrade GPU from T4 to A10G**

In `scripts/modal_run.py`, change line 41:

```python
    gpu="T4",
```

to:

```python
    gpu=modal.gpu.A10G(),
```

**Step 2: Add AMP comparison experiment function**

After the existing `run_all()` function (after line 122), add a new function:

```python
@app.function(
    image=image,
    gpu=modal.gpu.A10G(),
    volumes={"/data": vol},
    timeout=14400,
)
def run_amp_comparison():
    """Run fp32 vs fp16 comparison for M2 and M3 across 3 seeds."""
    import os
    import shutil
    import subprocess

    os.chdir("/root/finetune-bench")
    os.environ["PYTHONUNBUFFERED"] = "1"
    subprocess.run(["pip", "install", "-e", ".", "--no-deps", "-q"], check=True)

    # Patch robustness.py for CUDA (same as run_all)
    path = "evaluation/robustness.py"
    with open(path) as f:
        code = f.read()
    code = code.replace(
        'gen = torch.Generator()',
        'gen = torch.Generator(device="cpu")',
    )
    code = code.replace(
        "mask = torch.rand(result.shape, generator=gen) < rate",
        "mask = (torch.rand(result.shape, generator=gen) < rate).to(result.device)",
    )
    code = code.replace(
        "mask = torch.rand(features.shape, generator=gen) >= rate",
        "mask = (torch.rand(features.shape, generator=gen) >= rate).to(features.device)",
    )
    with open(path, "w") as f:
        f.write(code)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data: check volume cache
    data_dir = "/data/complaints.csv"
    local_data = "data/complaints.csv"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(data_dir):
        print("Using cached CFPB data from volume...")
        os.symlink(data_dir, local_data)
    else:
        print("Downloading CFPB data...")
        subprocess.run(["python", "scripts/download_data.py"], check=True)
        shutil.copy(local_data, data_dir)
        vol.commit()

    os.makedirs("results", exist_ok=True)

    from training.config import TrainConfig
    from training.train import train

    SEEDS = [42, 123, 456]
    AMP_CONFIGS = [
        {"variant": "M2", "use_amp": False},
        {"variant": "M2", "use_amp": True},
        {"variant": "M3", "use_amp": False},
        {"variant": "M3", "use_amp": True},
    ]

    all_results = []
    for cfg in AMP_CONFIGS:
        for seed in SEEDS:
            precision = "fp16" if cfg["use_amp"] else "fp32"
            run_name = f"{cfg['variant']}_{precision}_seed{seed}"
            print(f"\n{'=' * 60}")
            print(f"Running {run_name}")
            print("=" * 60)

            config = TrainConfig(
                variant=cfg["variant"],
                seed=seed,
                use_amp=cfg["use_amp"],
                run_name=run_name,
            )
            result = train(config)
            result["precision"] = precision
            result["run_name"] = run_name
            all_results.append(result)

    import json
    with open("results/amp_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Copy results to volume
    results_vol_dir = "/data/results"
    if os.path.exists(results_vol_dir):
        shutil.rmtree(results_vol_dir)
    shutil.copytree("results", results_vol_dir)
    vol.commit()

    return all_results
```

**Step 3: Update local_entrypoint to support --amp flag**

Replace the `main()` local entrypoint:

```python
@app.local_entrypoint()
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amp", action="store_true",
        help="Run AMP comparison (fp32 vs fp16) instead of v2 experiments",
    )
    args = parser.parse_args()

    if args.amp:
        print("Starting AMP comparison on Modal A10G GPU...")
        print("12 runs: 4 configs x 3 seeds (~60 min)")
        results = run_amp_comparison.remote()
    else:
        print("Starting finetune-bench on Modal A10G GPU...")
        print("This will run v2 experiments (~2-3 hours)")
        results = run_all.remote()

    import json
    from pathlib import Path

    local_results = Path("/Users/zenith/Desktop/finetune-bench/results")
    local_results.mkdir(exist_ok=True)

    for filename, data in results.items():
        if filename.startswith("_"):
            continue
        with open(local_results / filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {filename}")

    print("\nDone! JSON results saved locally.")
    print("Retrieve PNGs with: modal volume get finetune-bench-data results/")
```

Wait — `run_amp_comparison` returns a list, not a dict. Fix the local entrypoint to handle both:

```python
@app.local_entrypoint()
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amp", action="store_true",
        help="Run AMP comparison (fp32 vs fp16) instead of v2 experiments",
    )
    args = parser.parse_args()

    import json
    from pathlib import Path

    local_results = Path("/Users/zenith/Desktop/finetune-bench/results")
    local_results.mkdir(exist_ok=True)

    if args.amp:
        print("Starting AMP comparison on Modal A10G GPU...")
        print("12 runs: 4 configs x 3 seeds (~60 min)")
        results = run_amp_comparison.remote()
        with open(local_results / "amp_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved amp_comparison.json")
    else:
        print("Starting finetune-bench on Modal A10G GPU...")
        print("This will run v2 experiments (~2-3 hours)")
        results = run_all.remote()
        for filename, data in results.items():
            if filename.startswith("_"):
                continue
            with open(local_results / filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {filename}")

    print("\nDone! JSON results saved locally.")
    print("Retrieve PNGs with: modal volume get finetune-bench-data results/")
```

**Step 4: Lint**

Run: `ruff check scripts/modal_run.py`
Expected: clean

**Step 5: Commit**

```bash
git add scripts/modal_run.py
git commit -m "feat: upgrade Modal to A10G, add AMP comparison experiment (--amp flag)"
```

---

## Task 7: Add --use-amp to run_all_experiments.py

**Files:**
- Modify: `scripts/run_all_experiments.py:57-76` (run_dl_variants), `183-225` (main/argparse)

**Step 1: Add --use-amp argument to argparse**

In `scripts/run_all_experiments.py`, after line 196 (`--skip-onnx` argument), add:

```python
    parser.add_argument(
        "--use-amp", action="store_true",
        help="Enable fp16 mixed-precision training",
    )
```

**Step 2: Pass use_amp through to TrainConfig**

In `run_dl_variants()` (line 57), add `use_amp` parameter:

```python
def run_dl_variants(sample_size: int = 20_000, num_epochs: int = 3, use_amp: bool = False):
```

And pass it to `TrainConfig` at line 67:

```python
            config = TrainConfig(
                variant=variant,
                seed=seed,
                sample_size=sample_size,
                num_epochs=num_epochs,
                use_amp=use_amp,
            )
```

**Step 3: Pass use_amp from main to run_dl_variants**

At line 206:

```python
    dl_results = run_dl_variants(
        sample_size=args.sample_size, num_epochs=args.epochs,
        use_amp=args.use_amp,
    )
```

**Step 4: Lint**

Run: `ruff check scripts/run_all_experiments.py`
Expected: clean

**Step 5: Commit**

```bash
git add scripts/run_all_experiments.py
git commit -m "feat: add --use-amp flag to run_all_experiments.py"
```

---

## Task 8: ONNX fp16 Export in Experiment Runner

**Files:**
- Modify: `scripts/run_all_experiments.py:130-180` (run_onnx_export)

**Step 1: Add fp16 ONNX conversion and benchmarking**

In `run_all_experiments.py`, add import at top:

```python
from evaluation.export import benchmark_latency, convert_onnx_to_fp16, export_to_onnx
```

(Replace the existing `from evaluation.export import benchmark_latency, export_to_onnx` at line 14.)

In `run_onnx_export()`, after the existing `latency = benchmark_latency(...)` call at line 174, add fp16 export and benchmarking:

```python
    latency = benchmark_latency(model, onnx_path, tabular_dim=tabular_dim)
    print(f"Latency results (fp32): {latency}")

    # fp16 ONNX conversion + benchmark
    onnx_fp16_path = str(RESULTS_DIR / "model_m2_fp16.onnx")
    convert_onnx_to_fp16(onnx_path, onnx_fp16_path)
    print(f"ONNX fp16 model exported to {onnx_fp16_path}")

    latency_fp16 = benchmark_latency(model, onnx_fp16_path, tabular_dim=tabular_dim)
    print(f"Latency results (fp16): {latency_fp16}")

    combined_latency = {
        "fp32": latency,
        "fp16": latency_fp16,
    }

    with open(RESULTS_DIR / "latency_results.json", "w") as f:
        json.dump(combined_latency, f, indent=2)
```

**Step 2: Run lint**

Run: `ruff check scripts/run_all_experiments.py`
Expected: clean

**Step 3: Commit**

```bash
git add scripts/run_all_experiments.py
git commit -m "feat: add fp16 ONNX export and benchmarking to experiment runner"
```

---

## Task 9: DECISIONS.md

**Files:**
- Create: `DECISIONS.md`

**Step 1: Write DECISIONS.md**

Create `DECISIONS.md` in project root:

```markdown
# Design Decisions

## Why manual AMP integration, not HuggingFace Trainer

The training loop is custom (AdamW + linear warmup + cosine decay + gradient
accumulation + modality dropout + differential learning rates). HuggingFace
Trainer would abstract away the details that make this a portfolio piece.

Manual `torch.cuda.amp` integration with `GradScaler` demonstrates
understanding of the underlying mechanism: which operations run in fp16
(matmul, convolutions), which stay in fp32 (softmax, layer norm, loss), and
how loss scaling prevents gradient underflow in fp16.

## Why fp16 with GradScaler, not bf16

fp16 with GradScaler works on all CUDA GPUs (Volta and newer) and
demonstrates the full loss-scaling mechanism. bf16 (available on Ampere+) has
a larger dynamic range that eliminates the need for loss scaling — simpler in
production but less instructive as a portfolio piece.

**Production recommendation:** On Ampere+ GPUs (A10G, A100, H100), use bf16
via `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` with no
GradScaler. Simpler code, no risk of gradient underflow.

## Why honest fp16 limitations reporting

DistilBERT has 66M parameters. The memory savings from fp16 are real but
modest (~20-30% peak reduction), not the dramatic 50%+ savings seen on 7B+
parameter models. Reporting this honestly — including the GPU name, total
VRAM, and utilization percentage — shows engineering maturity. Inflating the
benefit would be caught by any reviewer who understands the relationship
between model size and precision scaling.
```

**Step 2: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: add DECISIONS.md with AMP design rationale"
```

---

## Task 10: Run Full Test Suite

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v --ignore=tests/test_overfit_smoke.py`
Expected: all pass (existing 36 + 6 new = 42 tests)

**Step 2: Run linter on all changed files**

Run: `ruff check training/ evaluation/ scripts/ tests/`
Expected: clean

---

## Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | GPUProfiler class | `training/gpu_profiler.py` | 3 new |
| 2 | Profiler integration in train.py | `training/train.py` | regression check |
| 3 | `use_amp` config field | `training/config.py` | 3 new |
| 4 | AMP in training + eval loops | `training/train.py` | regression check |
| 5 | ONNX fp16 conversion | `evaluation/export.py` | 1 new |
| 6 | Modal A10G + AMP experiment | `scripts/modal_run.py` | — |
| 7 | --use-amp in experiment runner | `scripts/run_all_experiments.py` | — |
| 8 | ONNX fp16 in experiment runner | `scripts/run_all_experiments.py` | — |
| 9 | DECISIONS.md | `DECISIONS.md` | — |
| 10 | Final verification | — | full suite |
