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

DistilBERT has 66M parameters. On an NVIDIA A10 (24 GB), fp16 reduces peak
memory by only 8% (1952 MB -> 1797 MB) — not the dramatic 50%+ savings seen
on 7B+ parameter models. The real win is a 2x epoch speedup from Tensor Core
acceleration (70.9s -> 36.2s for M2). Macro-F1 is preserved within noise
(0.6562 fp32 vs 0.6570 fp16). Reporting this honestly — including the GPU
name, total VRAM, and utilization percentage — shows engineering maturity.
Inflating the benefit would be caught by any reviewer who understands the
relationship between model size and precision scaling.
