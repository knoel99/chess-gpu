# ♟ Chess-GPU

Apprendre à jouer aux échecs par imitation des Grands Maîtres, avec des réseaux de neurones entraînés sur GPU.

## Structure

```
src/
├── common/              # code partagé (téléchargement, préparation)
├── phase1_mlp/          # réseau feedforward MLP
└── phase2_transformer/  # Transformer avec attention (à venir)

results/
└── phase1_mlp/          # résultats et article (results.md)

docs/                    # documentation théorique (matrices, modèle)
```

## Phase 1 — MLP (feedforward)

Réseau dense 832→1024→512→256→N entraîné sur 208k parties de 10 GMs.
- **Top-1 accuracy** : 24.8% — **Top-5** : 52.5%
- **Elo estimé** : ~838–950 vs Stockfish 1350
- 📄 [Résultats détaillés](results/phase1_mlp/results.md)

## Phase 2 — Transformer (à venir)

Architecture avec mécanisme d'attention pour capturer le contexte des coups précédents.

## Démarrage rapide (Colab)

```bash
cd /content && python chess-gpu/run_colab.py
```

## Licence

MIT