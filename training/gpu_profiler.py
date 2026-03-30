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
