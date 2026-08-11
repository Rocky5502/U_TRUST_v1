# Windows + RTX 5070 Ti setup

This is the primary environment for the U-TRUST experiments.

## Hardware rule

A standard GeForce RTX 5070 Ti has 16 GB GDDR7. Never write "48 GB RTX 5070 Ti" in the paper unless the frozen `nvidia-smi` manifest demonstrates multiple devices or another environment exposing that aggregate memory. The experiment code loads one 8B model at a time.

## Prerequisites

- Windows 11 x86-64 recommended.
- Current NVIDIA GeForce/Studio driver.
- Git for Windows.
- Python 3.11 x64.
- Hugging Face account. Qwen3-8B is open; Llama 3.1 8B Instruct is gated and requires accepting Meta's terms.

## One-command setup

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

The script creates `.venv`, installs PyTorch 2.11.0 CUDA 12.8 wheels, installs Transformers/bitsandbytes, verifies CUDA, runs tests, and clones pinned benchmarks.

## Hugging Face authentication

```powershell
.\.venv\Scripts\python.exe -m pip install -U huggingface_hub
hf auth login
```

Before Llama download, accept Meta Llama 3.1 terms on Hugging Face. Do not commit tokens.

## Capture exact machine

```powershell
.\scripts\capture_windows_manifest.ps1
```

This writes `results/machine_manifest_windows.json` with CPU, Windows build, RAM, disks, each GPU, VRAM, driver, aggregate GPU memory, Python, PyTorch, CUDA runtime, and compute capability.

## Models — sequential, not simultaneous

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --model qwen3-8b
.\.venv\Scripts\python.exe scripts\model_smoke_test.py --model qwen3-8b

.\.venv\Scripts\python.exe scripts\download_models.py --model llama31-8b
.\.venv\Scripts\python.exe scripts\model_smoke_test.py --model llama31-8b
```

Use 4-bit NF4 and record exact Hugging Face snapshot SHAs before the frozen test run.

## Completion order

1. Freeze environment manifest and pass tests.
2. Freeze model snapshot SHAs after smoke tests.
3. Run a small benchmark pilot for schema/debugging only.
4. Run development split, fit calibrator, select thresholds.
5. Freeze prompts/models/benchmarks/seeds/quantization/calibrator/thresholds.
6. Run RQ1 frozen test.
7. Run RQ2 frozen test.
8. Run ablations and statistics.
9. Generate tables from frozen raw records only.
10. Update the paper; never retune after viewing test outcomes.
