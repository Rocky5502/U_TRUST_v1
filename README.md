# U-TRUST

**U-TRUST: Uncertainty-Guided Trust Routing for Robust Multi-Agent Alignment under Prompt Injection**

Official experiment repository for the AAAI-27 AI Alignment project. This repository contains the implementation and reproducibility scaffold; it intentionally contains **no fabricated paper results**.

## Primary compute environment

The main campaign is designed for **native Windows** on an Intel-based workstation with 64 GB system RAM, Samsung 990 PRO 2 TB SSD, and NVIDIA GeForce RTX 5070 Ti-class GPU hardware. A standard RTX 5070 Ti has 16 GB VRAM, so the code loads **one model at a time** using 4-bit NF4. Exact CPU, Windows build, GPU count, per-device memory, aggregate GPU memory, driver, and PyTorch CUDA runtime are captured before the frozen run.

Start here: **[`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md)**.

## Models

| Key | Model | Local mode |
|---|---|---|
| `qwen3-8b` | `Qwen/Qwen3-8B` | 4-bit NF4, non-thinking primary condition |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 4-bit NF4, 32k experiment cap |

The same backbone is reused sequentially across Planner, Worker, Verifier, and Executor/Synthesizer roles. We do not keep four copies or both model families resident at the same time.

## Windows quick start

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\capture_windows_manifest.ps1
```

Then authenticate to Hugging Face and smoke-test the models one at a time:

```powershell
hf auth login
.\.venv\Scripts\python.exe scripts\download_models.py --model qwen3-8b
.\.venv\Scripts\python.exe scripts\model_smoke_test.py --model qwen3-8b

.\.venv\Scripts\python.exe scripts\download_models.py --model mistral7b
.\.venv\Scripts\python.exe scripts\model_smoke_test.py --model mistral7b
```

See `docs/WINDOWS_SETUP.md` and `docs/EXPERIMENT_PROTOCOL.md` for the complete workflow.

## Scientific discipline

- Never fit on the frozen test split.
- Keep benchmark-native IDs and matched benign controls.
- Do not change prompts, model snapshots, attacks, or thresholds after the research freeze.
- Populate manuscript tables only from frozen raw results.

## Current status

**Pre-experiment engineering stage.** Qwen3-8B has passed the local scoring smoke test; Mistral-7B-Instruct-v0.3 is the second active backbone and must pass the same local smoke test before the benchmark pilot. Final benchmark execution must occur on the target Windows/RTX 5070 Ti machine. Any manuscript result marked TBD remains unfilled until those frozen runs are complete.
