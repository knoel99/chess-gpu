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

## Phase 2 — Transformer

Architecture avec mécanisme d'attention pour capturer le contexte des coups précédents.

- **Encodage enrichi** : 846 features (pièces + tour + roque + en-passant + n° coup)
- **Séquence** : 16 dernières positions (8 coups complets de contexte)
- **Architecture** : 4 couches × 8 têtes d'attention, d_model=256, ffn=1024
- **Masque causal** : chaque position ne voit que les précédentes

```bash
cd /content && python chess-gpu/run_colab.py --phase2   # phase 2 seule
cd /content && python chess-gpu/run_colab.py --all      # phases 1 + 2
```

## Démarrage rapide (Colab)

```bash
cd /content && python chess-gpu/run_colab.py            # phase 1 (MLP)
cd /content && python chess-gpu/run_colab.py --phase2   # phase 2 (Transformer)
cd /content && python chess-gpu/run_colab.py --all      # les deux
```

## Licence

MIT