"""Tests for GPU memory profiler — all CPU-safe."""

from unittest.mock import MagicMock, patch

from training.gpu_profiler import GPUProfiler, GPUSnapshot


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


def _make_enabled_profiler():
    """Create a profiler with mocked CUDA, pre-populated with realistic data."""
    profiler = GPUProfiler(enabled=False)
    # Force enabled to bypass torch.cuda.is_available() check
    profiler.enabled = True
    profiler.snapshots = [
        GPUSnapshot(epoch=0, step=0, allocated_mb=500.0, reserved_mb=800.0, max_allocated_mb=600.0),
        GPUSnapshot(epoch=0, step=50, allocated_mb=550.0, reserved_mb=850.0, max_allocated_mb=650.0),
    ]
    profiler.epoch_times = [12.5, 11.8]
    # Authoritative epoch peaks from torch.cuda.max_memory_allocated()
    # (higher than sampled snapshots to represent inter-sample spikes)
    profiler.epoch_peaks_mb = [700.0, 680.0]
    return profiler


@patch("training.gpu_profiler.torch.cuda")
def test_profiler_summary_enabled_path(mock_cuda):
    """Enabled profiler summary returns correct GPU metrics."""
    mock_cuda.get_device_name.return_value = "NVIDIA A10G"
    mock_props = MagicMock()
    mock_props.total_memory = 24_000_000_000  # 24 GB in bytes
    mock_cuda.get_device_properties.return_value = mock_props

    profiler = _make_enabled_profiler()
    summary = profiler.summary()

    assert summary["gpu_available"] is True
    assert summary["gpu_name"] == "NVIDIA A10G"
    assert summary["gpu_total_mb"] == round(24_000_000_000 / 1e6, 0)
    # Peak uses authoritative epoch_peaks_mb (700.0), not sampled snapshots (650.0)
    assert summary["gpu_peak_allocated_mb"] == 700.0
    assert summary["gpu_mean_allocated_mb"] == 525.0
    assert summary["epoch_mean_time_s"] == 12.2
    assert summary["total_training_time_s"] == 24.3
    assert "gpu_peak_utilization_pct" in summary


@patch("training.gpu_profiler.torch.cuda")
def test_profiler_summary_no_bool_values(mock_cuda):
    """Enabled summary values are numeric (not bool), safe for mlflow.log_metric."""
    mock_cuda.get_device_name.return_value = "NVIDIA T4"
    mock_props = MagicMock()
    mock_props.total_memory = 16_000_000_000
    mock_cuda.get_device_properties.return_value = mock_props

    profiler = _make_enabled_profiler()
    summary = profiler.summary()

    for key, value in summary.items():
        if key == "gpu_available":
            assert value is True
        else:
            # Every non-boolean value must be numeric and NOT a bool
            assert isinstance(value, (int, float, str)), f"{key} has unexpected type {type(value)}"
            if isinstance(value, (int, float)):
                assert not isinstance(value, bool), f"{key} is a bool, would break mlflow.log_metric"


def test_disabled_summary_bool_guard():
    """Disabled summary's gpu_available=False must not pass an isinstance(_, (int, float)) guard."""
    profiler = GPUProfiler(enabled=False)
    summary = profiler.summary()
    # Simulate the train.py logging guard
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise AssertionError(f"{key}={value} would be logged as metric but shouldn't be")
