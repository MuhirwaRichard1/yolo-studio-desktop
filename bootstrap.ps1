<#
    Creates the .venv for YOLO Studio and installs the CUDA build of PyTorch.

    Safe to re-run: an existing venv is reused and packages are upgraded in
    place. Pass -Recreate to start from a clean environment.
#>
param(
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu124",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"
$py   = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Python not found at $Python" -ForegroundColor Red
    Write-Host "Install it with:  winget install --id Python.Python.3.11 --scope user"
    exit 1
}

if ($Recreate -and (Test-Path $venv)) {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment at $venv" -ForegroundColor Cyan
    & $Python -m venv $venv
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip setuptools wheel --quiet

# torch first, from the CUDA index. Doing this before ultralytics stops pip
# from resolving the default CPU-only wheels and then refusing to replace them.
Write-Host "Installing PyTorch (CUDA) - this is a multi-GB download..." -ForegroundColor Cyan
& $py -m pip install torch torchvision --index-url $CudaIndex

Write-Host "Installing application dependencies..." -ForegroundColor Cyan
& $py -m pip install -r (Join-Path $root "requirements.txt")

Write-Host ""
Write-Host "Verifying GPU access..." -ForegroundColor Cyan
& $py -c @"
import torch, sys
print('torch      ', torch.__version__)
print('cuda build ', torch.version.cuda)
print('available  ', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('device     ', p.name, f'{p.total_memory/1024**3:.1f} GB')
else:
    print('NOTE: training will fall back to CPU.', file=sys.stderr)
"@

Write-Host ""
Write-Host "Done. Launch the app with:  .\run.ps1" -ForegroundColor Green
