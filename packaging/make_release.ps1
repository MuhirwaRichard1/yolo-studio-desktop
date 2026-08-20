<#
    Assemble release artifacts, checksum them, split anything over GitHub's
    2 GB per-file limit, and optionally publish a GitHub Release.

        .\packaging\make_release.ps1 -Version 1.0.0
        .\packaging\make_release.ps1 -Version 1.0.0 -Publish

    Collects whatever exists: the Windows installer, and a Linux AppImage if
    one has been copied out of WSL into dist\.
#>
param(
    [string]$Version = "1.0.0",
    [switch]$Publish,
    [switch]$Draft
)

$ErrorActionPreference = "Stop"
$root    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$release = Join-Path $root "dist\release"
# GitHub rejects any single asset over 2 GB. Stay under it with margin.
$limit   = 1900MB

New-Item -ItemType Directory -Force $release | Out-Null

function Split-Large($file) {
    if ($file.Length -le $limit) { return @($file.FullName) }

    Write-Host ("  splitting {0} ({1:N2} GB) into parts" -f $file.Name, ($file.Length / 1GB)) `
        -ForegroundColor Yellow
    $parts = @()
    $stream = [System.IO.File]::OpenRead($file.FullName)
    try {
        $buffer = New-Object byte[] (8MB)
        $index = 0
        while ($stream.Position -lt $stream.Length) {
            $partPath = "{0}.part-{1:d3}" -f $file.FullName, $index
            $out = [System.IO.File]::Create($partPath)
            try {
                $written = 0
                while ($written -lt $limit -and $stream.Position -lt $stream.Length) {
                    $want = [Math]::Min($buffer.Length, $limit - $written)
                    $read = $stream.Read($buffer, 0, $want)
                    if ($read -le 0) { break }
                    $out.Write($buffer, 0, $read)
                    $written += $read
                }
            } finally { $out.Dispose() }
            $parts += $partPath
            $index++
        }
    } finally { $stream.Dispose() }

    # Tell the user how to put it back together.
    $join = Join-Path $release "REJOIN-$($file.BaseName).md"
    @"
# Rejoining ``$($file.Name)``

This file was split because GitHub Releases rejects assets over 2 GB.
Download every ``.part-NNN`` file into one folder, then:

**Linux / macOS**
``````bash
cat $($file.Name).part-* > $($file.Name)
chmod +x $($file.Name)
sha256sum -c SHA256SUMS
``````

**Windows (PowerShell)**
``````powershell
`$out = [System.IO.File]::Create("$($file.Name)")
Get-ChildItem "$($file.Name).part-*" | Sort-Object Name | ForEach-Object {
    `$in = [System.IO.File]::OpenRead(`$_.FullName)
    `$in.CopyTo(`$out); `$in.Dispose()
}
`$out.Dispose()
Get-FileHash "$($file.Name)" -Algorithm SHA256
``````
"@ | Set-Content $join -Encoding utf8
    return $parts
}

Write-Host "==> collecting artifacts" -ForegroundColor Cyan
$sources = @()
$sources += Get-ChildItem "$root\dist\installer\*.exe" -ErrorAction SilentlyContinue
$sources += Get-ChildItem "$root\dist\*.AppImage" -ErrorAction SilentlyContinue

if (-not $sources) {
    Write-Host "Nothing to release. Build first:" -ForegroundColor Red
    Write-Host "  .\packaging\build_windows.ps1"
    Write-Host "  wsl -d Ubuntu-22.04 bash packaging/build_linux.sh   (then copy the AppImage into dist\)"
    exit 1
}

$assets = @()
foreach ($src in $sources) {
    $dest = Join-Path $release $src.Name
    Write-Host ("  {0} ({1:N2} GB)" -f $src.Name, ($src.Length / 1GB))
    Copy-Item $src.FullName $dest -Force
    $assets += Split-Large (Get-Item $dest)
}

# Checksums cover the whole files, so verification happens after rejoining.
Write-Host "==> writing SHA256SUMS" -ForegroundColor Cyan
$sumFile = Join-Path $release "SHA256SUMS"
Remove-Item $sumFile -ErrorAction SilentlyContinue
foreach ($src in $sources) {
    $hash = (Get-FileHash $src.FullName -Algorithm SHA256).Hash.ToLower()
    Add-Content $sumFile "$hash  $($src.Name)"
    Write-Host "  $hash  $($src.Name)"
}
$assets += $sumFile
$assets += (Get-ChildItem "$release\REJOIN-*.md" -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })

# Split parts replace the oversized original; never upload both.
$assets = $assets | Where-Object {
    $item = Get-Item $_
    -not ($item.Length -gt $limit)
} | Select-Object -Unique

Write-Host ""
Write-Host "==> release assets in $release" -ForegroundColor Green
$assets | ForEach-Object { Write-Host ("   {0}" -f (Split-Path $_ -Leaf)) }

if ($Publish) {
    $gh = @("$env:LOCALAPPDATA\gh\bin\gh.exe", "gh") |
          Where-Object { (Get-Command $_ -ErrorAction SilentlyContinue) -or (Test-Path $_) } |
          Select-Object -First 1
    if (-not $gh) { throw "gh CLI not found." }

    $tag = "v$Version"

    # If the release already exists, add to it instead of failing. Artifacts for
    # the two platforms are built on different machines and rarely finish
    # together, so a second run has to be able to top up an existing release.
    & $gh release view $tag --repo MuhirwaRichard1/yolo-studio-desktop *> $null
    $exists = ($LASTEXITCODE -eq 0)

    if ($exists) {
        Write-Host "==> $tag exists; uploading assets to it" -ForegroundColor Cyan
        # --clobber replaces same-named assets, so re-running is idempotent.
        $ghArgs = @("release", "upload", $tag) + $assets + @("--clobber")
    } else {
        Write-Host "==> publishing $tag" -ForegroundColor Cyan
        $ghArgs = @("release", "create", $tag) + $assets +
                  @("--title", "YOLO Studio $Version", "--notes-file", (Join-Path $release "NOTES.md"))
        if ($Draft) { $ghArgs += "--draft" }
    }

    if (-not (Test-Path (Join-Path $release "NOTES.md"))) {
        @"
YOLO Studio $Version

Self-contained builds - Python, Qt, PyTorch with CUDA and ultralytics are all
included, so nothing needs installing first.

- **Windows 10/11 x64**: run the ``-setup.exe``. Installs per-user, no admin
  needed. SmartScreen will warn that the publisher is unknown; the binaries are
  unsigned.
- **Linux x86_64**: ``chmod +x`` the ``.AppImage`` and run it. Built against
  glibc 2.35 (Ubuntu 22.04+, Debian 12+, Fedora 36+).

Needs ~8 GB free disk, and an NVIDIA driver supporting CUDA 12.4 (R550+) for
GPU training. Without one it still runs, on the CPU.

Verify downloads against ``SHA256SUMS``. Any file split into ``.part-NNN``
pieces has a ``REJOIN-*.md`` explaining how to reassemble it.
"@ | Set-Content (Join-Path $release "NOTES.md") -Encoding utf8
    }

    & $gh @ghArgs
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed ($LASTEXITCODE)" }
    Write-Host "==> published" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Add -Publish to create the GitHub Release." -ForegroundColor Yellow
}
