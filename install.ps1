# Installs Voix isolée on a Windows machine without any build tool:
#
#   irm https://raw.githubusercontent.com/quentingimenez/voix-isolee/main/install.ps1 | iex
#
# Downloads the latest release (programs and libraries) and the models from
# their upstream projects. Running it again updates the programs and keeps the
# models and the recordings already there.
param(
    [string]$Dossier = (Join-Path $env:LOCALAPPDATA 'Programs\VoixIsolee')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repository = 'quentingimenez/voix-isolee'
$models = @(
    @{ File = 'voxlock_enroll.onnx'; Url = 'https://github.com/arbazsiddiqui/voxlock/releases/download/sherpa-onnx-models/voxlock_enroll.onnx' },
    @{ File = 'voxlock_chunk_int8.onnx'; Url = 'https://github.com/arbazsiddiqui/voxlock/releases/download/sherpa-onnx-models/voxlock_chunk_int8.onnx' },
    @{ File = 'voxlock_chunk.onnx'; Url = 'https://github.com/arbazsiddiqui/voxlock/releases/download/sherpa-onnx-models/voxlock_chunk.onnx' },
    @{ File = 'dpdfnet2.onnx'; Url = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx' }
)

function Get-File([string]$Url, [string]$Path) {
    $partial = "$Path.partiel"
    Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing -Headers @{ 'User-Agent' = 'voix-isolee-install' }
    Move-Item -Force $partial $Path
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'Voix isolée nécessite Windows 64 bits.'
}

Write-Host "Installation de Voix isolée dans $Dossier"
New-Item -ItemType Directory -Force $Dossier | Out-Null

Write-Host 'Recherche de la dernière version...'
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repository/releases/latest" -Headers @{ 'User-Agent' = 'voix-isolee-install' }
$asset = $release.assets | Where-Object { $_.name -like 'voix-isolee-*-windows-x64.zip' } | Select-Object -First 1
if (-not $asset) {
    throw "La version $($release.tag_name) ne contient pas d'archive Windows."
}

Write-Host "Téléchargement de $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) Mo)..."
$archive = Join-Path $Dossier $asset.name
Get-File $asset.browser_download_url $archive
Expand-Archive -Path $archive -DestinationPath $Dossier -Force
Remove-Item $archive

$modelFolder = Join-Path $Dossier 'models'
New-Item -ItemType Directory -Force $modelFolder | Out-Null
foreach ($model in $models) {
    $path = Join-Path $modelFolder $model.File
    if ((Test-Path $path) -and (Get-Item $path).Length -gt 0) {
        Write-Host "Modèle déjà présent : $($model.File)"
        continue
    }
    Write-Host "Téléchargement du modèle $($model.File)..."
    Get-File $model.Url $path
}

if (-not (Test-Path (Join-Path $env:SystemRoot 'System32\vcruntime140.dll'))) {
    Write-Warning 'Le composant « Microsoft Visual C++ Redistributable » semble absent. Si les programmes ne démarrent pas, installez-le : https://aka.ms/vs/17/release/vc_redist.x64.exe'
}

Write-Host ''
Write-Host "Voix isolée $($release.tag_name) est installée."
Write-Host ''
Write-Host 'Pour commencer, dans PowerShell :'
Write-Host "  cd `"$Dossier`""
Write-Host '  .\essai.exe micros                          # liste les micros'
Write-Host '  .\essai.exe empreinte --micro "<nom>"       # enregistre votre empreinte vocale (20 s, seul, sans bruit)'
Write-Host '  .\essai.exe direct --micro "<nom>"          # écoute en direct, au casque'
Write-Host ''
Write-Host 'Aide complète : .\essai.exe'
