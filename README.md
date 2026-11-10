# Voix isolée

Outil Windows léger qui ne laisse passer que **votre** voix, en direct : les bruits
ambiants et les voix des autres personnes proches du micro sont supprimés. Les autres
applications (Discord, OBS, jeux…) reçoivent le son nettoyé via un micro virtuel.

État : en construction.

## Principe

```
Micro réel → VoxLock (extraction de la voix enregistrée) → micro virtuel
```

- Empreinte vocale enregistrée une seule fois (~30 s).
- Trois modes : Léger, Équilibre, Qualité.
- Icône près de l'horloge + petite fenêtre de réglage.

## Composants tiers

- [VoxLock](https://github.com/arbazsiddiqui/voxlock) (Apache-2.0) — modèle d'extraction de voix cible.
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0) — exécution du modèle.

Les modèles et bibliothèques binaires ne sont pas versionnés (`models/`, `vendor/`).
