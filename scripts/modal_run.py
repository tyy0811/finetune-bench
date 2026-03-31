"""Run finetune-bench experiments on Modal with A10G GPU."""

import modal

app = modal.App("finetune-bench")

# Build image with deps + local code baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.44.0",
        "scikit-learn==1.8.0",
        "lightgbm==4.6.0",
        "pandas==2.3.3",
        "numpy==1.26.4",
        "mlflow==3.10.1",
        "pydantic==2.12.5",
        "onnx==1.20.1",
        "onnxruntime==1.23.2",
        "matplotlib==3.10.8",
        "requests==2.32.5",
    )
    .add_local_dir(
        "/Users/zenith/Desktop/finetune-bench",
        remote_path="/root/finetune-bench",
        ignore=[
            ".git/**", "data/**", "mlruns/**", "**/__pycache__/**",
            "**/*.egg-info/**", ".claude/**", "mlflow.db",
            "*.zip", "*.ipynb", "docs/**",
        ],
    )
)

# Persistent volume for data + results
vol = modal.Volume.from_name("finetune-bench-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": vol},
    timeout=14400,
)
def run_all():
    import os
    import shutil
    import subprocess

    os.chdir("/root/finetune-bench")
    os.environ["PYTHONUNBUFFERED"] = "1"
    subprocess.run(["pip", "install", "-e", ".", "--no-deps", "-q"], check=True)

    # Patch robustness.py for CUDA
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

    # Data: check volume cache first
    data_dir = "/data/complaints.csv"
    local_data = "data/complaints.csv"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(data_dir):
        print("Using cached CFPB data from volume...")
        os.symlink(data_dir, local_data)
    else:
        print("Downloading CFPB data (first run, will be cached)...")
        subprocess.run(["python", "scripts/download_data.py"], check=True)
        shutil.copy(local_data, data_dir)
        vol.commit()
        print("Data cached to volume.")

    os.makedirs("results", exist_ok=True)

    # === V2: M2b + Temporal only ===
    print("\n" + "=" * 70)
    print("V2 EXPERIMENTS: M2b + Temporal (no v1 models needed)")
    print("=" * 70)
    subprocess.run([
        "python", "scripts/run_v2_experiments.py",
        "--skip-50k", "--skip-calibration", "--skip-heatmap",
    ], check=True)

    # Collect results
    results = {}
    import json
    from pathlib import Path
    for f in sorted(Path("results").glob("*.json")):
        with open(f) as fh:
            results[f.name] = json.load(fh)
        print(f"\n=== {f.name} ===")
        print(json.dumps(results[f.name], indent=2)[:2000])

    # Copy results to volume
    results_vol_dir = "/data/results"
    if os.path.exists(results_vol_dir):
        shutil.rmtree(results_vol_dir)
    shutil.copytree("results", results_vol_dir)
    vol.commit()
    print("\nResults copied to volume")

    return results


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": vol},
    timeout=14400,
)
def run_amp_comparison():
    """Run fp32 vs fp16 comparison for M2 and M3 across 3 seeds."""
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/root/finetune-bench")
    sys.path.insert(0, "/root/finetune-bench")
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


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": vol},
    timeout=3600,
)
def run_onnx_benchmark():
    """Export best M2 to ONNX fp32+fp16 and benchmark latency."""
    import os
    import subprocess
    import sys

    os.chdir("/root/finetune-bench")
    sys.path.insert(0, "/root/finetune-bench")
    os.environ["PYTHONUNBUFFERED"] = "1"
    subprocess.run(["pip", "install", "-e", ".", "--no-deps", "-q"], check=True)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Copy model checkpoints from volume
    vol_results = "/data/results"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(vol_results):
        import shutil
        for f in os.listdir(vol_results):
            src = os.path.join(vol_results, f)
            dst = os.path.join("results", f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"Copied {len(os.listdir(vol_results))} files from volume")

    # Data: need adapter for tabular_dim
    data_dir = "/data/complaints.csv"
    local_data = "data/complaints.csv"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(data_dir):
        os.symlink(data_dir, local_data)

    from adapters.cfpb import CFPBAdapter
    from evaluation.export import benchmark_latency, convert_onnx_to_fp16, export_to_onnx
    from models.fusion_model import MultimodalClassifier

    adapter = CFPBAdapter(sample_size=20_000, seed=42)
    splits = adapter.preprocess()
    tabular_dim = splits["train"]["tabular_features"].shape[1]

    # Find best M2 checkpoint
    model_path = "results/M2_fp32_seed42_best.pt"
    if not os.path.exists(model_path):
        model_path = "results/M2_fp32_seed42.pt"
    if not os.path.exists(model_path):
        # Try default naming from earlier runs
        model_path = "results/M2_seed42_best.pt"
    if not os.path.exists(model_path):
        model_path = "results/M2_seed42.pt"

    print(f"Using model: {model_path}")

    model = MultimodalClassifier(
        num_classes=len(adapter.class_names),
        tabular_input_dim=tabular_dim,
        modality_dropout=False,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    # fp32 ONNX export + benchmark
    onnx_fp32_path = "results/model_m2.onnx"
    export_to_onnx(model, onnx_fp32_path, tabular_dim=tabular_dim)
    latency_fp32 = benchmark_latency(model, onnx_fp32_path, tabular_dim=tabular_dim)
    print(f"fp32 latency: {latency_fp32}")

    # fp16 ONNX conversion + benchmark
    onnx_fp16_path = "results/model_m2_fp16.onnx"
    convert_onnx_to_fp16(onnx_fp32_path, onnx_fp16_path)
    latency_fp16 = benchmark_latency(model, onnx_fp16_path, tabular_dim=tabular_dim)
    print(f"fp16 latency: {latency_fp16}")

    return {"fp32": latency_fp32, "fp16": latency_fp16}


@app.local_entrypoint()
def main(amp: bool = False, onnx: bool = False):
    import json
    from pathlib import Path

    local_results = Path("/Users/zenith/Desktop/finetune-bench/results")
    local_results.mkdir(exist_ok=True)

    if onnx:
        print("Starting ONNX fp32+fp16 benchmark on Modal A10G GPU...")
        latency = run_onnx_benchmark.remote()
        with open(local_results / "latency_results.json", "w") as f:
            json.dump(latency, f, indent=2)
        print("Saved latency_results.json")
        print(f"fp32: {latency['fp32']}")
        print(f"fp16: {latency['fp16']}")
    elif amp:
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
