"""Run finetune-bench v1 + v2 experiments on Modal with T4 GPU."""

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
    gpu="T4",
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


@app.local_entrypoint()
def main():
    print("Starting finetune-bench on Modal T4 GPU...")
    print("This will run v1 + v2 experiments (~2-3 hours)")
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
