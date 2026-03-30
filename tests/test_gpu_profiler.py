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
