param(
    [string]$Output = "results\machine_manifest_windows.json"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

$cpu = Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name
$os = Get-CimInstance Win32_OperatingSystem
$ramBytes = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum
$disks = Get-PhysicalDisk | Select-Object FriendlyName, MediaType, Size

$gpus = @()
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $rows = nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv,noheader,nounits
    foreach ($row in $rows) {
        $p = $row -split ',\s*'
        if ($p.Count -lt 5) {
            throw "Unexpected nvidia-smi row: $row"
        }
        $gpus += [pscustomobject][ordered]@{
            index = [int]$p[0]
            name = $p[1]
            memory_total_mib = [int]$p[2]
            driver_version = $p[3]
            pci_bus_id = $p[4]
        }
    }
}

# Sum explicitly for compatibility with Windows PowerShell's handling of
# ordered dictionaries / PSCustomObjects returned from nvidia-smi parsing.
$totalGpuMiB = 0
foreach ($gpu in $gpus) {
    $totalGpuMiB += [int64]$gpu.memory_total_mib
}

$nvcc = $null
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $nvcc = (nvcc --version | Out-String).Trim()
}

$pythonInfo = $null
if (Test-Path ".venv\Scripts\python.exe") {
    $pythonInfoRaw = & .venv\Scripts\python.exe scripts\capture_python_env.py
    $pythonInfo = $pythonInfoRaw | ConvertFrom-Json
}

$manifest = [ordered]@{
    captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    hostname = $env:COMPUTERNAME
    os = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build_number = $os.BuildNumber
        architecture = $os.OSArchitecture
    }
    cpu = $cpu
    system_ram_bytes = [int64]$ramBytes
    system_ram_gib = [math]::Round($ramBytes / 1GB, 2)
    storage = @($disks)
    nvidia_gpus = $gpus
    gpu_count = @($gpus).Count
    aggregate_gpu_memory_mib = [int64]$totalGpuMiB
    aggregate_gpu_memory_gib = [math]::Round($totalGpuMiB / 1024, 2)
    nvcc = $nvcc
    python_environment = $pythonInfo
}

$dir = Split-Path -Parent $Output
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $Output
Write-Host "Wrote $Output" -ForegroundColor Green
Write-Host "GPU count: $(@($gpus).Count); aggregate reported VRAM: $totalGpuMiB MiB" -ForegroundColor Cyan
