<#
    Build the Windows bundle and installer.

        .\packaging\build_windows.ps1              # bundle + installer
        .\packaging\build_windows.ps1 -SkipInstaller
        .\packaging\build_windows.ps1 -Version 1.1.0

    Requires the project .venv (see bootstrap.ps1) and, for the installer,
    Inno Setup 6.
#>
param(
    [string]$Version = "1.0.0",
    [switch]$SkipInstaller,
    [switch]$SkipBundle
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py   = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "No virtual environment. Run .\bootstrap.ps1 first." -ForegroundColor Red
    exit 1
}

Push-Location $root
try {
    if (-not $SkipBundle) {
        Write-Host "==> generating icons" -ForegroundColor Cyan
        & $py "packaging\make_icon.py"

        Write-Host "==> ensuring PyInstaller" -ForegroundColor Cyan
        & $py -m pip install --quiet --upgrade pyinstaller

        Write-Host "==> running PyInstaller (several minutes)" -ForegroundColor Cyan
        if (Test-Path "dist\YOLOStudio") { Remove-Item -Recurse -Force "dist\YOLOStudio" }
        & $py -m PyInstaller "packaging\yolostudio.spec" --noconfirm `
              --distpath "dist" --workpath "build"
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    }

    $gui    = "dist\YOLOStudio\YOLOStudio.exe"
    $worker = "dist\YOLOStudio\yolostudio-worker.exe"
    foreach ($f in @($gui, $worker)) {
        if (-not (Test-Path $f)) { throw "Expected build output missing: $f" }
    }

    $bytes = (Get-ChildItem "dist\YOLOStudio" -Recurse -File | Measure-Object Length -Sum).Sum
    Write-Host ("==> bundle: {0:N0} files, {1:N1} GB" -f `
        (Get-ChildItem "dist\YOLOStudio" -Recurse -File).Count, ($bytes / 1GB)) -ForegroundColor Green

    if (-not $SkipInstaller) {
        $iscc = @(
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1

        if (-not $iscc) {
            Write-Host "Inno Setup not found; skipping installer." -ForegroundColor Yellow
            Write-Host "  winget install --id JRSoftware.InnoSetup"
        } else {
            Write-Host "==> compiling installer (LZMA2 over several GB — slow)" -ForegroundColor Cyan
            New-Item -ItemType Directory -Force "dist\installer" | Out-Null
            & $iscc "/DAppVersion=$Version" "packaging\installer.iss"
            if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }

            Get-ChildItem "dist\installer\*.exe" | ForEach-Object {
                Write-Host ("==> installer: {0} ({1:N2} GB)" -f $_.Name, ($_.Length / 1GB)) `
                    -ForegroundColor Green
            }
        }
    }
}
finally {
    Pop-Location
}
