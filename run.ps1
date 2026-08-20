<#
    Launches YOLO Studio from the project's virtual environment.
    Run bootstrap.ps1 first if .venv does not exist yet.
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py   = Join-Path $root ".venv\Scripts\pythonw.exe"
$con  = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $con)) {
    Write-Host "No virtual environment found. Run .\bootstrap.ps1 first." -ForegroundColor Red
    exit 1
}

# -Console keeps the terminal attached so tracebacks are visible.
if ($args -contains "-Console" -or -not (Test-Path $py)) {
    & $con -m yolostudio
} else {
    Start-Process -FilePath $py -ArgumentList "-m", "yolostudio" -WorkingDirectory $root
}
