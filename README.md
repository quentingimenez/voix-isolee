# Voix isolée

Outil Windows léger qui ne laisse passer que **votre** voix, en direct : les bruits
ambiants et les voix des autres personnes proches du micro sont supprimés.

État : prototype en ligne de commande. L'interface graphique et le micro virtuel
ne sont pas encore faits.

## Installation

Dans PowerShell, sur un Windows 10 ou 11 en 64 bits :

```powershell
irm https://raw.githubusercontent.com/quentingimenez/voix-isolee/main/install.ps1 | iex
```

Le script télécharge les programmes et les modèles (environ 190 Mo) dans
`%LOCALAPPDATA%\Programs\VoixIsolee`. Rien d'autre n'est modifié sur la machine.
Le relancer met à jour les programmes en gardant les modèles et vos enregistrements.

Pour désinstaller, supprimez ce dossier.

## Utilisation

```powershell
cd "$env:LOCALAPPDATA\Programs\VoixIsolee"
.\essai.exe micros
```

1. **Empreinte vocale** (une seule fois, 20 secondes, seul et sans bruit) :

   ```powershell
   .\essai.exe empreinte --micro "<partie du nom du micro>"
   ```

2. **Écoute en direct**, au casque (sur des enceintes, le micro reprendrait le son) :

   ```powershell
   .\essai.exe direct --micro "<partie du nom du micro>"
   ```

   Entrée arrête le programme. Un nombre puis Entrée change la sensibilité aux
   chuchotements pendant l'écoute.

Options utiles :

- `--canal 1` : sur une carte son à plusieurs entrées, ne prendre que l'entrée 1
  (sans cette option, les entrées sont mélangées, ce qui affaiblit la voix).
- `--sortie "<nom>"` : choisir la sortie audio (`.\essai.exe sorties` les liste).
- `--mode leger|equilibre|qualite` : compromis charge / retard / qualité.

`.\essai.exe` sans argument affiche l'aide complète. `.\essai.exe test` enregistre
20 secondes et écrit le résultat de chaque mode dans le dossier `essais`, pour
comparer à l'écoute.

## Principe

```
Micro → VoxLock (ne garde que la voix de l'empreinte) → antibruit → sortie
          └ porte adaptative : laisse repasser vos chuchotements et vos rires
```

- Retard d'environ 200 ms ; charge d'environ la moitié d'un cœur de processeur.
- Limite connue : une voix de fond aussi forte que la vôtre au micro n'est pas
  entièrement retirée pendant que vous parlez.

## Compiler depuis les sources

Il faut Rust (toolchain MSVC), les Build Tools de Visual Studio et le CLI GitHub (`gh`).

```powershell
.\scripts\setup-dev.ps1
cargo build --release
cargo test --release
```

`scripts\package.ps1` produit l'archive publiée dans les versions GitHub.

## Composants tiers

- [VoxLock](https://github.com/arbazsiddiqui/voxlock) (Apache-2.0) : extraction de la voix cible.
- [DPDFNet](https://huggingface.co/Ceva-IP/DPDFNet) : antibruit.
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0) et ONNX Runtime (MIT) : exécution des modèles.

Les modèles ne sont pas dans ce dépôt : l'installation les télécharge depuis leurs projets d'origine.
Voir `NOTICE`.
