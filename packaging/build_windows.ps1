<#
    Build the Windows bundle and installer.

        .\packaging\build_windows.ps1              # bundle + installer
        .\packaging\build_windows.ps1 -SkipInstaller
        .\packaging\build_windows.ps1 -Version 1.1.0

    Code signing is optional; with none of the switches below the build comes
    out unsigned exactly as it did before. Pick one:

        # certificate already in the CurrentUser\My store (preferred: no
        # password ever reaches a command line)
        .\packaging\build_windows.ps1 -CertThumbprint A1B2C3...

        # .pfx file - imported into CurrentUser\My, then used by thumbprint
        .\packaging\build_windows.ps1 -CertPath codesign.pfx `
            -CertPassword (Read-Host "pfx password" -AsSecureString)

        # free, self-signed. Exercises the whole pipeline and produces a
        # genuinely signed build, but it is only trusted on machines where the
        # certificate is installed in Trusted Root. It does NOT clear
        # SmartScreen for anyone downloading the installer.
        .\packaging\build_windows.ps1 -SelfSigned

    Signing covers both bundled executables, the installer, and the uninstaller.

    Requires the project .venv (see bootstrap.ps1) and, for the installer,
    Inno Setup 6. Signing additionally requires signtool.exe from the Windows
    SDK (winget install --id Microsoft.WindowsSDK), or $env:SIGNTOOL pointing
    at one.
#>
param(
    [string]$Version = "1.0.0",
    [switch]$SkipInstaller,
    [switch]$SkipBundle,
    [string]$CertThumbprint,
    [string]$CertPath,
    [System.Security.SecureString]$CertPassword,
    [switch]$SelfSigned,
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py   = Join-Path $root ".venv\Scripts\python.exe"

# Inno Setup preprocessor placeholders, held in variables so PowerShell does not
# try to expand them as its own $-variables.
$InnoQuote = '$q'
$InnoFile  = '$f'

$SelfSignedSubject = "CN=YOLO Studio (self-signed, development only)"

function Resolve-SignTool {
    if ($env:SIGNTOOL) {
        if (Test-Path $env:SIGNTOOL) { return $env:SIGNTOOL }
        throw "SIGNTOOL is set to '$env:SIGNTOOL' but that path does not exist."
    }
    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin",
        "$env:ProgramFiles\Windows Kits\10\bin"
    ) | Where-Object { Test-Path $_ }

    $candidates = foreach ($r in $roots) {
        Get-ChildItem $r -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.Directory.Name -eq "x64" }
    }
    # Newest SDK wins; the version lives one directory above the x64 folder.
    $best = $candidates |
        Sort-Object -Descending -Property @{ Expression = {
            try { [version]$_.Directory.Parent.Name } catch { [version]"0.0.0.0" }
        } } |
        Select-Object -First 1
    if (-not $best) {
        throw "signtool.exe not found. Install the Windows SDK (winget install --id Microsoft.WindowsSDK) or set the SIGNTOOL environment variable."
    }
    return $best.FullName
}

function Resolve-SigningCertificate {
    # Returns a thumbprint, or $null when signing was not requested. Everything
    # downstream signs by thumbprint, so no password lands on a command line.
    if ($CertThumbprint) {
        $clean = $CertThumbprint -replace '[^0-9A-Fa-f]', ''
        $cert = Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My -ErrorAction SilentlyContinue |
            Where-Object { $_.Thumbprint -eq $clean } | Select-Object -First 1
        if (-not $cert) { throw "No certificate with thumbprint $clean in CurrentUser\My or LocalMachine\My." }
        return $cert.Thumbprint
    }

    if ($CertPath) {
        if (-not (Test-Path $CertPath)) { throw "Certificate file not found: $CertPath" }
        Write-Host "==> importing $CertPath into CurrentUser\My" -ForegroundColor Cyan
        $importArgs = @{ FilePath = $CertPath; CertStoreLocation = "Cert:\CurrentUser\My" }
        if ($CertPassword) { $importArgs.Password = $CertPassword }
        $cert = Import-PfxCertificate @importArgs
        return $cert.Thumbprint
    }

    if ($SelfSigned) {
        $existing = Get-ChildItem Cert:\CurrentUser\My |
            Where-Object { $_.Subject -eq $SelfSignedSubject -and $_.NotAfter -gt (Get-Date) } |
            Sort-Object NotAfter -Descending | Select-Object -First 1
        if ($existing) {
            Write-Host "==> reusing self-signed certificate $($existing.Thumbprint)" -ForegroundColor Cyan
            return $existing.Thumbprint
        }
        Write-Host "==> creating self-signed code-signing certificate" -ForegroundColor Cyan
        $cert = New-SelfSignedCertificate -Type CodeSigningCert `
            -Subject $SelfSignedSubject `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -NotAfter (Get-Date).AddYears(3) `
            -KeyExportPolicy Exportable
        Write-Host "    thumbprint: $($cert.Thumbprint)" -ForegroundColor DarkGray
        Write-Host "    Not trusted by other machines; SmartScreen will still warn." -ForegroundColor Yellow
        return $cert.Thumbprint
    }

    return $null
}

function Invoke-SignFile {
    param([string]$SignTool, [string]$Thumbprint, [string]$Path)

    $signArgs = @(
        "sign",
        "/fd", "sha256",
        "/sha1", $Thumbprint,
        "/tr", $TimestampUrl,
        "/td", "sha256",
        $Path
    )
    # Timestamp servers flake; a couple of retries is cheaper than a failed build.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & $SignTool @signArgs
        if ($LASTEXITCODE -eq 0) { return }
        if ($attempt -lt 3) {
            Write-Host "    signtool failed (exit $LASTEXITCODE), retry $attempt/3" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    }
    throw "signtool failed for $Path with exit code $LASTEXITCODE"
}

if (-not (Test-Path $py)) {
    Write-Host "No virtual environment. Run .\bootstrap.ps1 first." -ForegroundColor Red
    exit 1
}

Push-Location $root
try {
    # Resolve the tool before the certificate: -SelfSigned writes to the
    # certificate store, and there is no point doing that only to discover the
    # SDK is missing a moment later.
    $signingRequested = [bool]($CertThumbprint -or $CertPath -or $SelfSigned)
    $signtool   = $null
    $thumbprint = $null
    if ($signingRequested) {
        $signtool   = Resolve-SignTool
        $thumbprint = Resolve-SigningCertificate
        Write-Host "==> signing with $thumbprint" -ForegroundColor Cyan
        Write-Host "    signtool: $signtool" -ForegroundColor DarkGray
    } else {
        Write-Host "==> no signing certificate given; building unsigned" -ForegroundColor Yellow
    }

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

    # Sign the bundled executables before ISCC packages them. These matter more
    # than the installer for antivirus heuristics: the PyInstaller bootloader is
    # what the scanners actually flag.
    if ($thumbprint) {
        foreach ($f in @($gui, $worker)) {
            Write-Host "==> signing $f" -ForegroundColor Cyan
            Invoke-SignFile -SignTool $signtool -Thumbprint $thumbprint -Path $f
            Write-Host ("    {0}: {1}" -f $f, (Get-AuthenticodeSignature $f).Status) -ForegroundColor DarkGray
        }
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
            Write-Host "==> compiling installer (LZMA2 over several GB - slow)" -ForegroundColor Cyan
            New-Item -ItemType Directory -Force "dist\installer" | Out-Null

            $isccArgs = @("/DAppVersion=$Version")
            if ($thumbprint) {
                # /S<name>=<command> defines a sign tool for this run. installer.iss
                # only activates it when Sign is also defined, so an unsigned build
                # still compiles cleanly.
                $signCmd = "$InnoQuote$signtool$InnoQuote sign /fd sha256 /sha1 $thumbprint " +
                           "/tr $TimestampUrl /td sha256 $InnoFile"
                $isccArgs += "/Ssigntool=$signCmd"
                $isccArgs += "/DSign"
            }
            $isccArgs += "packaging\installer.iss"

            & $iscc @isccArgs
            if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }

            Get-ChildItem "dist\installer\*.exe" | ForEach-Object {
                Write-Host ("==> installer: {0} ({1:N2} GB)" -f $_.Name, ($_.Length / 1GB)) `
                    -ForegroundColor Green
                if ($thumbprint) {
                    Write-Host ("    signature: {0}" -f (Get-AuthenticodeSignature $_.FullName).Status) `
                        -ForegroundColor DarkGray
                }
            }
        }
    }
}
finally {
    Pop-Location
}
