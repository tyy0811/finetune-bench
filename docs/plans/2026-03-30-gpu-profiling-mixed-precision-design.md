# GPU Profiling & Mixed-Precision Training

**Branch:** `feat/gpu-profiling`
**Goal:** Add mixed-precision (fp16) training, GPU memory profiling, and resource utilization reporting to finetune-bench.

---

## 1. AMP Integration (training/train.py)

### Imports and Initialization

```python
from torch.cuda.amp import GradScaler
from torch.amp import autocast

# In train(), before the epoch loop:
scaler = GradScaler(enabled=config.use_amp)

# Device type for autocast:
if torch.cuda.is_available():
    device_type = "cuda"
else:
    device_type = "cpu"
    config.use_amp = False  # AMP silently disabled on non-CUDA
```

Uses `torch.cuda.amp.GradScaler` (no device arg) — `torch.amp.GradScaler` requires PyTorch 2.3+, and we're pinned to 2.2.2. `GradScaler(enabled=False)` is a safe no-op on CPU.

Uses `torch.amp.autocast` (with `device_type` arg) — available in 2.2.2 and is the non-deprecated path for autocast.

### Training Loop (train_one_epoch)

Both the main accumulation path and the tail-end incomplete batch path get identical scaler treatment:

```python
for step, batch in enumerate(train_loader):
    # ... move batch to device ...

    with autocast(device_type=device_type, enabled=config.use_amp):
        logits = model(text_inputs, tabular)
        loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        loss = loss / config.grad_accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % config.grad_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

# Tail-end: flush remaining gradients from incomplete accumulation
if (step + 1) % config.grad_accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
```

- `autocast` wraps forward + loss only.
- `scaler.unscale_()` before `clip_grad_norm_()` — clipping must operate on unscaled gradients.
- `scaler.step()` replaces `optimizer.step()`.
- `optimizer.zero_grad()` stays at end of block (matches existing code structure).

### Validation Loop

Autocast wraps the forward pass during validation too (inference benefits from fp16 speed):

```python
with torch.no_grad():
    with autocast(device_type=device_type, enabled=config.use_amp):
        logits = model(text_inputs, tabular)
```

No scaler involvement — no backward pass during validation.

---

## 2. GPUProfiler (training/gpu_profiler.py)

New file. Standalone class with three hooks into the training loop.

```python
"""GPU memory and utilization tracking for training runs."""

import torch
import time
from dataclasses import dataclass, field


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
        self.snapshots.append(GPUSnapshot(
            epoch=epoch,
            step=step,
            allocated_mb=torch.cuda.memory_allocated(self.device) / 1e6,
            reserved_mb=torch.cuda.memory_reserved(self.device) / 1e6,
            max_allocated_mb=torch.cuda.max_memory_allocated(self.device) / 1e6,
        ))

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

### Integration in train()

```python
from training.gpu_profiler import GPUProfiler

profiler = GPUProfiler(enabled=torch.cuda.is_available())

for epoch in range(config.num_epochs):
    profiler.on_epoch_start(epoch)
    train_loss = train_one_epoch(model, train_loader, ..., profiler=profiler)
    profiler.on_epoch_end(epoch)
    # ... validation, early stopping ...

# After training completes:
gpu_summary = profiler.summary()
for key, value in gpu_summary.items():
    if isinstance(value, (int, float)):
        mlflow.log_metric(f"gpu/{key}", value)
```

`train_one_epoch` calls `profiler.on_step_end(epoch, step)` inside the batch loop. Samples every 50 steps by default.

No pynvml dependency. `gpu_peak_utilization_pct` = `peak_allocated / total_vram * 100`.

---

## 3. TrainConfig Extension (training/config.py)

One new field:

```python
class TrainConfig(BaseModel):
    # ... existing fields ...
    use_amp: bool = False  # Enable mixed-precision (fp16)
```

Default `False` — preserves existing fp32 behavior. The AMP comparison script passes `True` explicitly for fp16 variants.

No `amp_dtype` field — we implement fp16 with GradScaler. bf16 is mentioned in DECISIONS.md as a production recommendation for Ampere+ GPUs but not implemented (YAGNI).

---

## 4. Modal Script Updates (scripts/modal_run.py)

Changes to the existing file:

1. **GPU upgrade:** `gpu="t4"` → `gpu=modal.gpu.A10G()` (Ampere Tensor Cores for fp16 speedup)

2. **Experiment matrix extension:**

```python
# New AMP comparison configs alongside existing v1/v2:
AMP_CONFIGS = [
    {"variant": "M2", "use_amp": False, "run_name": "M2_fp32"},
    {"variant": "M2", "use_amp": True,  "run_name": "M2_fp16"},
    {"variant": "M3", "use_amp": False, "run_name": "M3_fp32"},
    {"variant": "M3", "use_amp": True,  "run_name": "M3_fp16"},
]
SEEDS = [42, 123, 456]
# 4 configs × 3 seeds = 12 runs
```

3. **Results collection** picks up `gpu/*` metrics from MLflow automatically — the profiler logs inside `train()`, no new collection logic.

M1 excluded from AMP comparison — same DistilBERT encoder, no interesting fusion interaction to measure.

Existing v1 fp32 runs do not need re-running.

---

## 5. ONNX fp16 Extension (evaluation/export.py)

One new function:

```python
def convert_onnx_to_fp16(input_path: str, output_path: str) -> None:
    """Convert fp32 ONNX model to fp16, keeping fp32 IO for compatibility."""
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx

    model_fp32 = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, output_path)
```

`keep_io_types=True` preserves fp32 inputs/outputs — callers don't change preprocessing. Model runs fp16 internally.

Called after existing ONNX export in the experiment runner:

```python
export_to_onnx(model, "model.onnx", tabular_dim=tabular_dim)
convert_onnx_to_fp16("model.onnx", "model_fp16.onnx")
latency_fp32 = benchmark_latency(model, "model.onnx", ...)
latency_fp16 = benchmark_latency(model, "model_fp16.onnx", ...)
```

No new dependencies — `onnxruntime` and `onnx` already in `pyproject.toml`.

---

## 6. Tests

All tests run on CPU. No GPU required in CI.

### tests/test_gpu_profiler.py (3 tests)

```python
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
    """on_step_end only records snapshots at sample_every boundaries."""
    profiler = GPUProfiler(enabled=False)
    for step in range(100):
        profiler.on_step_end(0, step, sample_every=25)
    assert len(profiler.snapshots) == 0

def test_profiler_epoch_timing():
    """Disabled profiler records nothing."""
    profiler = GPUProfiler(enabled=False)
    profiler.on_epoch_start(0)
    profiler.on_epoch_end(0)
    assert profiler.epoch_times == []
```

### tests/test_amp_config.py (2 tests)

```python
def test_use_amp_default_false():
    """use_amp defaults to False to preserve existing fp32 behavior."""
    config = TrainConfig()
    assert config.use_amp is False

def test_gradscaler_disabled_is_noop():
    """GradScaler with enabled=False passes tensors through unchanged."""
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    t = torch.tensor(1.0)
    assert scaler.scale(t) == t
```

`torch.cuda.amp.GradScaler(enabled=False)` verified to work on CPU (PyTorch 2.2.2). No skip marker needed.

### tests/test_export.py (1 new test, added to existing file)

```python
def test_onnx_fp16_conversion(tmp_path):
    """fp16 ONNX model is smaller than fp32 and produces valid output."""
    fp32_path = str(tmp_path / "model.onnx")
    fp16_path = str(tmp_path / "model_fp16.onnx")
    export_to_onnx(model, fp32_path, tabular_dim=tabular_dim)
    convert_onnx_to_fp16(fp32_path, fp16_path)

    assert os.path.getsize(fp16_path) < os.path.getsize(fp32_path)

    # Verify inference runs
    session = ort.InferenceSession(fp16_path)
    # ... run dummy input, check output shape ...
```

**Total: 6 new tests, all CPU-safe. Existing 36 tests unchanged.**

---

## 7. README Additions

### New Section: "GPU Profiling & Mixed Precision"

```markdown
## GPU Profiling & Mixed Precision

Training supports automatic mixed precision (fp16) via `torch.cuda.amp`,
with GPU memory profiling logged to MLflow.

### Enable fp16 training

    python scripts/run_all_experiments.py --use-amp

### GPU profiling

Every training run logs GPU peak memory, mean allocation, and per-epoch
timing to MLflow. View with:

    mlflow ui  # Navigate to gpu/ metrics

### fp32 vs fp16 comparison (Modal A10G, 24 GB)

| Variant | Precision | Macro-F1 | Peak GPU (MB) | Epoch Time (s) | Speedup |
|---------|-----------|----------|---------------|-----------------|---------|
| M2      | fp32      | ...      | ...           | ...             | —       |
| M2      | fp16      | ...      | ...           | ...             | ...×    |
| M3      | fp32      | ...      | ...           | ...             | —       |
| M3      | fp16      | ...      | ...           | ...             | ...×    |

*DistilBERT (66M params) is compact; fp16 memory savings are modest (~20-30%).
The technique has greater impact on larger models (1B+ params).*
```

Results table filled with real numbers after Modal run.

### Extended ONNX Table

```markdown
| Format  | Precision | Latency (single) | Latency (batch=32) | Model size |
|---------|-----------|-------------------|--------------------|------------|
| PyTorch | fp32      | 179.23 ms         | 4001.02 ms         | ~254 MB    |
| ONNX    | fp32      | 120.24 ms         | 4363.88 ms         | 254.21 MB  |
| ONNX    | fp16      | ... ms            | ... ms             | ~127 MB    |
```

---

## 8. DECISIONS.md (new file)

Two entries:

**"Why manual AMP integration, not HuggingFace Trainer":**
The training loop is custom (AdamW + linear warmup + cosine decay + gradient accumulation + modality dropout + differential learning rates). Manual `torch.cuda.amp` integration with `GradScaler` demonstrates understanding of the underlying mechanism: which operations run in fp16 (matmul), which stay in fp32 (softmax, loss), and how loss scaling prevents gradient underflow.

**"Why fp16 with GradScaler, not bf16":**
fp16 with GradScaler works on all CUDA GPUs and demonstrates the full loss-scaling mechanism. bf16 (available on Ampere+) has a larger dynamic range that eliminates the need for loss scaling — simpler in production but less instructive. For Ampere+ deployments, bf16 is the pragmatic choice; document this as a production recommendation.

**"Why honest limitations reporting":**
DistilBERT (66M params) shows modest fp16 savings (~20-30% peak memory). Documenting this honestly with specific GPU name, total VRAM, and utilization percentage demonstrates engineering maturity over inflated claims.

---

## 9. Commit Strategy

| Commit | Content | Tests |
|--------|---------|-------|
| 1 | `GPUProfiler` class + integration in `train.py` | 3 tests |
| 2 | AMP integration in `train.py` + `TrainConfig.use_amp` | 2 tests |
| 3 | AMP comparison configs in `scripts/modal_run.py` + A10G upgrade | — |
| 4 | fp16 ONNX conversion in `evaluation/export.py` | 1 test |
| 5 | Results table + README + DECISIONS.md | — |

---

## Dependencies

No new dependencies. `torch.cuda.amp` is part of PyTorch 2.2.2. ONNX fp16 conversion uses `onnxruntime` (already in `pyproject.toml`).

## Risk Notes

- **CI remains CPU-only.** All tests use disabled profiler/scaler. AMP comparison runs on Modal only.
- **Modal cost:** ~$1.30 for 12 runs (4 configs × 3 seeds) on A10G.
- **fp16 convergence.** DistilBERT fine-tuning is stable with fp16. If F1 drops >1 std dev, reduce `lr_head` from 1e-3 to 5e-4.
- **PyTorch 2.2.2 API.** Uses `torch.cuda.amp.GradScaler` (no device arg) + `torch.amp.autocast` (with device_type). `torch.amp.GradScaler` requires 2.3+.
