param(
    [string]$Python = "py -3.11",
    [switch]$SkipTorch,
    [switch]$SkipBenchmarks
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

Write-Host "[1/7] Checking NVIDIA driver" -ForegroundColor Cyan
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install/update the NVIDIA driver first."
}
nvidia-smi

Write-Host "[2/7] Creating Python 3.11 environment" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    Invoke-Expression "$Python -m venv .venv"
}
$Py = (Resolve-Path ".venv\Scripts\python.exe").Path
& $Py -m pip install --upgrade pip setuptools wheel

if (-not $SkipTorch) {
    Write-Host "[3/7] Installing reproducible PyTorch CUDA 12.8 build" -ForegroundColor Cyan
    & $Py -m pip uninstall -y torch torchvision torchaudio 2>$null
    & $Py -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
}

Write-Host "[4/7] Installing U-TRUST + local model dependencies" -ForegroundColor Cyan
& $Py -m pip install -e ".[local,dev]"

Write-Host "[5/7] Verifying CUDA / Blackwell visibility" -ForegroundColor Cyan
& $Py -c "import torch; print('torch=',torch.__version__); print('torch_cuda=',torch.version.cuda); print('cuda_available=',torch.cuda.is_available()); print('gpu_count=',torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory//(1024**2), 'MiB', torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]; assert torch.cuda.is_available(), 'PyTorch cannot see CUDA'"

Write-Host "[6/7] Running repository tests" -ForegroundColor Cyan
& $Py scripts\validate_environment.py
& $Py scripts\smoke_test.py
& $Py -m pytest -q

if (-not $SkipBenchmarks) {
    Write-Host "[7/7] Cloning pinned benchmark revisions" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File scripts\setup_benchmarks.ps1
} else {
    Write-Host "[7/7] Benchmark cloning skipped" -ForegroundColor Yellow
}

Write-Host "Windows setup completed. Next: scripts\capture_windows_manifest.ps1" -ForegroundColor Green
