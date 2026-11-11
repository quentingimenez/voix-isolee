# Builds the release programs and packs them, with the libraries they need,
# into target\voix-isolee-<version>-windows-x64.zip (the archive install.ps1
# downloads from the GitHub release). Run from the repository root.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$version = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$stage = Join-Path $root 'target\package'
$archive = Join-Path $root "target\voix-isolee-$version-windows-x64.zip"

Push-Location $root
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw 'cargo build a échoué.' }
} finally {
    Pop-Location
}

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

$files = @(
    'target\release\essai.exe',
    'target\release\collecte.exe',
    'vendor\sherpa-onnx\lib\sherpa-onnx-c-api.dll',
    'vendor\sherpa-onnx\lib\onnxruntime.dll',
    'vendor\sherpa-onnx\lib\onnxruntime_providers_shared.dll',
    'LICENSE',
    'NOTICE'
)
foreach ($file in $files) {
    Copy-Item (Join-Path $root $file) $stage
}

if (Test-Path $archive) { Remove-Item $archive }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $archive
Write-Host "Archive écrite : $archive ($([math]::Round((Get-Item $archive).Length / 1MB, 1)) Mo)"
