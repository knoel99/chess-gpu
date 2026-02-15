# ♟ Chess-GPU — Résultats d'entraînement et d'évaluation

> Apprendre à jouer aux échecs par imitation des Grands Maîtres, avec un réseau de neurones entraîné sur GPU.

---

## 1. Vue d'ensemble du projet

### Concept

Au lieu de coder des règles et des heuristiques, on **apprend le jeu par imitation** :
on télécharge les parties de 10 des meilleurs joueurs du monde sur Chess.com,
on encode chaque position en vecteur matriciel, et on entraîne un réseau de neurones
à prédire le prochain coup — exactement comme un GM le jouerait.

### Pipeline

```
Téléchargement PGN  →  Encodage matriciel  →  Entraînement PyTorch  →  Évaluation vs Stockfish
    (Chess.com)         (position → 832D)       (GPU, batch=8192)        (alpha-beta search)
```

---

## 2. Données d'entraînement

### Joueurs sélectionnés (Top 10 GM — comptes Chess.com)

| # | Joueur | Compte Chess.com | Parties |
|---|--------|------------------|---------|
| 1 | Magnus Carlsen | `magnuscarlsen` | 9 092 |
| 2 | Hikaru Nakamura | `hikaru` | 67 411 |
| 3 | Fabiano Caruana | `fabianocaruana` | — |
| 4 | Ian Nepomniachtchi | `lachesisq` | — |
| 5 | Alireza Firouzja | `firouzja2003` | — |
| 6 | Nihal Sarin | `nihalsarin` | — |
| 7 | Praggnanandhaa | `rpragchess` | — |
| 8 | Maxime Vachier-Lagrave | `lyonbeast` | — |
| 9 | Alexander Grischuk | `grischuk` | — |
| 10 | Anish Giri | `anishgiri` | — |

### Volume de données

| Métrique | Valeur |
|----------|--------|
| Fichier PGN total | **598.5 Mo** |
| Parties parsées | **208 588** |
| Exemples d'entraînement | **17 730 038** |
| Coups uniques (tokens) | **1 968** |
| Fichier NPZ (données encodées) | **256.9 Mo** |

### Encodage des positions

Chaque position est encodée en un vecteur **one-hot de dimension 832** :
- 64 cases × 13 catégories (vide + 6 types de pièces × 2 couleurs)
- `board_to_vector()` utilise `piece_map()` pour ne parcourir que les pièces présentes (~20-32) au lieu des 64 cases

```
Position → [0,0,0,1,0,0,0,0,0,0,0,0,0, 1,0,0,...,0] → (832,)
              case a1: tour blanche          case a2: ...
```

### Optimisations du parsing

| Étape | Avant | Après | Gain |
|-------|-------|-------|------|
| Parsing PGN | ~80 min (séquentiel) | **86 s** (8 workers) | **56×** |
| `board_to_vector` | 64× `piece_at()` | `piece_map()` (~25 items) | **~2.5×** |

---

## 3. Architecture du réseau

### ChessNet — Réseau profond à 4 couches

```
Entrée (832)
    │
    ├── Linear(832, 1024)
    ├── BatchNorm1d(1024)
    ├── ReLU
    ├── Dropout(0.3)
    │
    ├── Linear(1024, 512)
    ├── BatchNorm1d(512)
    ├── ReLU
    ├── Dropout(0.3)
    │
    ├── Linear(512, 256)
    ├── BatchNorm1d(256)
    ├── ReLU
    ├── Dropout(0.3)
    │
    └── Linear(256, 1968)
         └── Softmax → probabilité de chaque coup
```

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Optimizer | Adam (lr=1e-3, wd=1e-5) |
| Scheduler | CosineAnnealing → 1e-6 |
| Batch size | 8 192 (auto-tuné selon VRAM) |
| Epochs | 50 (patience=10) |
| Dropout | 0.3 |
| Paramètres totaux | **2 018 480** (7.7 Mo) |
| Train/Val split | 90% / 10% |

---

## 4. Entraînement

### Configuration matérielle

| Composant | Détail |
|-----------|--------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 95 Go (58 Go utilisés — dataset entier sur GPU) |
| Puissance | ~300-365 W / 600 W max |
| Température | 44-49°C |

### Optimisations GPU

| Technique | Impact |
|-----------|--------|
| Dataset complet sur VRAM | Élimine le transfert CPU→GPU (goulot principal) |
| AMP Mixed Precision (float16) | ~2× plus rapide, 2× moins de VRAM/batch |
| `torch.compile` | Optimisation des kernels CUDA |
| `cudnn.benchmark` | Auto-tune des algorithmes cuDNN |
| Batch size = 8192 | Sature les cores GPU |

### Courbes d'entraînement

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Durée |
|-------|-----------|----------|-------|-------|-------|
| 1 | 4.6879 | 3.8459 | 18.3% | 40.5% | 6.7s |
| 5 | 4.0104 | 3.4933 | 21.2% | 45.9% | 2.9s |
| 10 | 3.8581 | 3.3715 | 22.3% | 48.1% | 2.9s |
| 20 | 3.7472 | 3.2693 | 23.4% | 50.0% | 2.9s |
| 30 | 3.6837 | 3.2121 | 24.1% | 51.3% | 2.8s |
| 40 | 3.6329 | 3.1704 | 24.6% | 52.2% | 2.8s |
| **50** | **3.6117** | **3.1548** | **24.8%** | **52.5%** | 2.8s |

**Durée totale : 2 min 48 s** (50 epochs)

### Interprétation

- **Top-1 = 24.8%** : le réseau joue le même coup qu'un GM 1 fois sur 4
- **Top-5 = 52.5%** : le coup du GM est dans le top-5 des prédictions 1 fois sur 2
- La loss continue de baisser à epoch 50 → potentiel d'amélioration avec plus d'epochs
- Pas d'overfitting détecté (train loss > val loss, gap stable)

---

## 5. Évaluation vs Stockfish

Le modèle est évalué contre Stockfish limité à **Elo ~1350** (niveau club amateur).

### Modes d'évaluation

| Mode | Description |
|------|-------------|
| **Instantané** | Le réseau joue directement son coup le plus probable |
| **Recherche α-β** | Alpha-beta pruning (depth=4, budget=1000 nœuds) sur les probabilités du réseau |
| **Benchmark** | Compare les deux modes côte à côte |

### Résultats

| Mode | W | D | L | Score | Win Rate | Elo estimé |
|------|---|---|---|-------|----------|------------|
| Instantané | 0 | 1 | 9 | 0.5/10 | 5% | **~838** |
| Recherche d=4 n=1000 | 0 | 0 | 10 | 0.0/10 | 0% | **~950** |

### Analyse des parties

**Mode instantané — observations :**
- Les parties durent en moyenne **155 demi-coups** (le réseau survit longtemps)
- Le réseau perd du matériel progressivement sans s'en rendre compte
- Exemple : G04 a **Δ+12** en matériel (avantage écrasant) mais ne sait pas convertir → se fait mater
- Le score interne est toujours `sc=+0.0` — le réseau n'a aucune notion de valeur positionnelle

**Mode recherche — observations :**
- Les parties sont plus courtes (**109 demi-coups** en moyenne)
- La recherche alpha-beta utilise très peu de nœuds (30-160 par coup)
- Le score oscille (`+0.5`, `-0.9`, `+8.3`) — non fiable car la fonction d'évaluation est la probabilité de coup, pas une évaluation positionnelle
- G01 (nulle) : matériel égal 3 vs 3 pendant 200 coups — le réseau tient en fin de partie simplifiée

### Pourquoi la recherche n'aide pas (encore)

La recherche alpha-beta a besoin d'une **fonction d'évaluation fiable** pour comparer les positions.
Or notre réseau ne prédit que le **prochain coup probable**, pas la **qualité de la position**.
Avec 24.8% de précision top-1, le réseau se trompe 3 fois sur 4, et la recherche amplifie ces erreurs
en s'engageant avec confiance dans des lignes perdantes.

---

## 6. Fichiers de sortie

Chaque évaluation produit un ensemble de fichiers horodatés (pas d'écrasement) :

```
data/top_players_model_runs/
├── 20260215_175526_instantané.log              # 87 Ko — log coup par coup
├── 20260215_175526_instantané_summary.json     # résumé machine-readable
├── 20260215_175526_instantané_material.csv     # données matériel par coup
├── 20260215_175526_instantané_material.png     # graphique matériel
├── 20260215_175526_instantané_games.html       # visualisation interactive
├── 20260215_175537_recherche_d4_n1000.log
├── 20260215_175537_recherche_d4_n1000_summary.json
├── 20260215_175537_recherche_d4_n1000_material.csv
├── 20260215_175537_recherche_d4_n1000_material.png
└── 20260215_175537_recherche_d4_n1000_games.html
```

---

## 7. Performance de la pipeline

| Étape | Durée | Détail |
|-------|-------|--------|
| Téléchargement PGN | ~15 min | 10 joueurs, 8 workers parallèles |
| Parsing + encodage | **86 s** | 208k parties → 17.7M exemples, 8 workers |
| Entraînement | **2 min 48 s** | 50 epochs, batch=8192, dataset sur GPU |
| Évaluation instantanée | ~10 s | 10 parties |
| Évaluation recherche | ~20 s | 10 parties, depth=4, 1000 nœuds |
| **Total pipeline** | **~20 min** | Première exécution (avec téléchargement) |
| **Relance (données existantes)** | **~3 min** | Entraînement + évaluations seulement |

---

## 8. Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.12 |
| Deep Learning | PyTorch 2.0+ (CUDA, AMP, torch.compile) |
| Échecs | python-chess |
| Adversaire | Stockfish 14 |
| Calcul matriciel | NumPy / CuPy (GPU) |
| Environnement | Google Colab |
| Versioning | Git + GitHub |

---

## 9. Pistes d'amélioration

| Piste | Impact attendu | Effort |
|-------|---------------|--------|
| Plus d'epochs (100-200) | Top-1 ~26-28% | Faible |
| Réseau plus large (2048→1024→512→256) | Top-1 ~27-30% | Faible |
| Encoder tour de jeu, roque, en-passant | +1-2% top-1 | Moyen |
| Tête d'évaluation (prédire qui gagne) | Score fiable pour α-β | Moyen |
| Entraînement adversarial (self-play) | Elo +200-400 | Élevé |
| Architecture Transformer | State-of-the-art | Élevé |

---

## 10. Conclusion

Avec **17.7 millions de positions** de Grands Maîtres et **2 min 48 s** d'entraînement sur GPU,
le réseau atteint **24.8% de précision top-1** — il joue le même coup qu'un GM 1 fois sur 4.
C'est remarquable pour un MLP simple, mais insuffisant pour battre même un Stockfish bridé à 1350 Elo.

Le goulot d'étranglement n'est plus le calcul (86s de parsing, 3 min d'entraînement) mais
la **représentation** : le réseau voit la position mais ne comprend pas la stratégie.
Les prochaines étapes — tête d'évaluation positionnelle et architecture plus expressive —
devraient permettre de franchir la barre des 1000 Elo.

---

*Projet : [chess-gpu](https://github.com/knoel99/chess-gpu) — Entraîné le 15 février 2026*
