# Préparation des données d'entraînement

Tous les scripts s'exécutent depuis la racine du projet avec le Python de
l'environnement d'entraînement :

```powershell
C:\Users\SuperKnacky\code\toolchains\voix-isolee-entrainement\Scripts\python.exe training\prepare_data.py
C:\Users\SuperKnacky\code\toolchains\voix-isolee-entrainement\Scripts\python.exe training\make_rirs.py
```

## Ordre

1. `prepare_data.py` : concatène les prises de voix de l'utilisateur par session
   (`user\<session>.npy` et `user\index.json`, avec découpage train/val) et les
   phrases LibriSpeech (`others\audio.npy` et `others\index.json`). Les fichiers
   sont écrits sous `--out` (défaut `entrainement\prepared`). Options utiles :
   `--minutes-per-speaker`, `--skip-user`, `--skip-others`.
2. `make_rirs.py` : génère les réponses impulsionnelles proche, moyenne et
   lointaine dans `entrainement\prepared\rirs.npz` (`--rooms`, `--seed`).
3. `selftest.py` : vérifie l'ensemble sur de petites données synthétiques. Il
   n'écrit que dans `target\training-selftest\`, puis supprime ce dossier ; il
   ne touche jamais au vrai dossier `entrainement\`.

`common.py` regroupe les constantes et les fonctions partagées ; il n'est pas
exécuté directement.

## Entraînement personnalisé

`train_personal.py` réentraîne VoxLock pour une seule personne à partir des
données préparées. Il garde les sessions `normale`, `animee`, `basse`,
`chuchotement` et `rires`, puis retire le téléphone, les autres voix et les
bruits de bureau. `personal_mixer.py` construit les mélanges à la volée et
`compat.py` adapte le chargement de VoxLock à cet environnement.

Depuis la racine du projet :

```powershell
C:\Users\SuperKnacky\code\toolchains\voix-isolee-entrainement\Scripts\python.exe training\train_personal.py --run runs\personnel
```

Les dossiers par défaut sont `entrainement\prepared`, `entrainement\voxlock`
et `entrainement\ckpt\voxlock`. Le script utilise le CPU ou le GPU disponible
(`--device`, `--gpu-index`) et accepte notamment `--steps`, `--every`, `--bs`,
`--accum`, `--lr`, `--workers`, `--threads`, `--seed` et `--val-examples`.

À chaque sauvegarde, le script écrit `ckpt_NNNNNN\avg_model.pt` et
`config.yaml`, puis évalue des exemples de validation fixes. Si `--run`
contient déjà des points de contrôle, il repart du plus récent. Les sorties
sont :

- `log.jsonl` : perte, débit et taux d'apprentissage ;
- `val.jsonl` : SI-SDR de la sortie et de l'entrée, niveau des absents ;
- `listen\step_NNNNNN\` : fichiers traités avec `--listen` et l'enrôlement
  `--voiceprint` (WAV 16 kHz 16 bits) ;
- `TRAIN_DONE` : entraînement terminé.

Le test autonome construit ses propres données synthétiques, entraîne deux pas,
vérifie la reprise, puis supprime `target\training-selftest-train\` :

```powershell
C:\Users\SuperKnacky\code\toolchains\voix-isolee-entrainement\Scripts\python.exe training\selftest_train.py
```

Il ne touche jamais aux données ni aux poids réels de `entrainement\`.

