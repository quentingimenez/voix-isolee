# Fetches what a source build needs and git does not track: the sherpa-onnx
# libraries (vendor\sherpa-onnx) and the models (models\). Requires the GitHub
# CLI (gh) and tar. Run from the repository root, then: cargo build --release
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$sherpaVersion = 'v2.0.1'
$sherpaArchive = "sherpa-onnx-$sherpaVersion-win-x64-shared-MD-Release-no-tts"
$vendor = Join-Path $root 'vendor'
$models = Join-Path $root 'models'

New-Item -ItemType Directory -Force $vendor, $models | Out-Null

if (-not (Test-Path (Join-Path $vendor 'sherpa-onnx\lib\sherpa-onnx-c-api.lib'))) {
    gh release download $sherpaVersion -R k2-fsa/sherpa-onnx -p "$sherpaArchive.tar.bz2" -D $vendor --skip-existing
    tar -xjf (Join-Path $vendor "$sherpaArchive.tar.bz2") -C $vendor
    Rename-Item (Join-Path $vendor $sherpaArchive) 'sherpa-onnx'
    Remove-Item (Join-Path $vendor "$sherpaArchive.tar.bz2")
}

gh release download sherpa-onnx-models -R arbazsiddiqui/voxlock -p 'voxlock_enroll.onnx' -p 'voxlock_chunk_int8.onnx' -p 'voxlock_chunk.onnx' -D $models --skip-existing
gh release download speech-enhancement-models -R k2-fsa/sherpa-onnx -p 'dpdfnet2.onnx' -D $models --skip-existing

Write-Host 'Dépendances prêtes. Compilez avec : cargo build --release'
