# ♟ Échecs 100 % matriciels — mouvements, règles, inversion, transitions

Suite de `king_matrix.md` : le moteur complet est implémenté dans `src/matrix_chess/`.
Toute la mécanique du jeu est exprimée en opérations matricielles (numpy, batch-aware),
prête à être portée telle quelle sur GPU (les shifts sont des convolutions).

## 1. Mouvements des pièces = shifts matriciels

Chaque pièce a son équivalent matriciel (`engine.py`, testé pièce par pièce dans
`test_engine.py`) :

| Pièce | Opération matricielle |
|---|---|
| Roi | somme de 8 shifts unitaires (≡ convolution 3×3, cf. `king_matrix.md`) |
| Cavalier | somme de 8 shifts (±1, ±2) — saute les obstacles par construction |
| Pion | shift avant ∧ vide (+ double shift depuis la rangée initiale), diagonales ∧ ennemi (+ en passant) |
| Tour / Fou / Dame | **propagation avec blocage** : `front ← shift(front ∧ vide, d)` itéré 7 fois par direction, union des fronts |

La propagation glissante inclut la première case occupée rencontrée (cible de prise) ;
les cases amies sont retirées par un simple masque `∧ (vide ∨ ennemi)`.

## 2. Règles en équivalent matriciel

- **Blocage** : intrinsèque à la propagation (le front ne traverse que `vide`).
- **Prise** : la matrice de coup écrase la case cible (voir §3).
- **Échec** : `roi ∧ attaques(adversaire) ≠ 0` — une réduction booléenne, batch-aware.
- **Mouvements interdits (clouages, roi en prise)** : les N coups candidats sont
  appliqués d'un bloc → tenseur `(N, 8, 8)`, et l'échec est testé **en un seul
  calcul vectorisé** sur le batch (`Game.legal_moves`). C'est exactement le motif
  GPU : mille plateaux se vérifient comme un seul.
- **Échec et mat / pat** : `aucun coup légal` ∧ (`échec` / `pas d'échec`).
- **Roque, en passant, promotion** : voir §3 — tous sont des matrices.

## 3. Chaque coup est une matrice 64×64

Plateau aplati `b ∈ Z⁶⁴`. Un coup de `f` vers `t` est l'opérateur **linéaire**

```
T = I − E_ff − E_tt + E_tf          b' = T · b
```

(`E_ij` = matrice élémentaire). Cette même forme couvre tout le jeu :

| Coup | Matrice |
|---|---|
| Déplacement / prise | `I − E_ff − E_tt + E_tf` |
| Promotion | `I − E_ff − E_tt + k·E_tf` (k = 5 pour dame : le pion vaut ±1) |
| En passant | forme standard + `− E_cc` (case du pion capturé) |
| Roque | forme standard appliquée au roi **et** à la tour dans la même matrice |

Une partie entière est donc une **composition** : `b_n = T_n ··· T_1 · b_0 = G · b_0`.

## 4. Résultat empirique : l'inversion est affine, pas linéaire

Sur la démo 5 coups (`demo.py`, italienne avec deux prises) :

- `G · b0 = b5` vérifié exactement ;
- **`rang(G) = 54/64`** : G est singulière. Chaque coup met à zéro la ligne de la
  case de départ et écrase la case d'arrivée — les prises **projettent de
  l'information à zéro**, aucun inverse matriciel exact n'existe.
- Le pseudo-inverse de Moore-Penrose `G⁺ · b5` reconstruit le plateau **sauf les
  pièces capturées** (2 cases fausses sur la démo) : c'est la meilleure
  reconstruction purement linéaire possible.
- L'inversion **exacte** est affine : en remontant coup par coup,
  `b_{k−1} = R_k · b_k + correction_k`, où `R = I − E_tt − E_ff + (1/k)·E_ft`
  inverse le déplacement et la correction réinjecte la pièce écrasée
  (mémoire des prises, en dehors de l'algèbre). `invert_game` reconstruit `b0`
  exactement, prises, roques, promotions et en passant compris.

C'est le théorème d'information du jeu : **les échecs ne sont pas réversibles**,
et la matrice le dit d'elle-même par son rang.

### 4 bis. La méthode pour rendre G inversible : augmenter l'espace d'état

`invert_full_game.py` démontre la solution complète : on étend le vecteur
d'état de 64 à **94 composantes** (64 cases + 30 places de « cimetière »).
Une prise ne détruit alors plus d'information — elle **déplace** la pièce
capturée vers le cimetière :

| Coup | Matrice 94×94 |
|---|---|
| simple | transposition `(f t)` |
| prise | 3-cycle `f → t → cimetière → f` |
| en passant | `(f t)(case_capturée cimetière)` |
| roque | `(e1 g1)(h1 f1)` |
| promotion | permutation × k sur la ligne d'arrivée |

Chaque coup est une **matrice monomiale** (permutation à coefficients ±k),
donc inversible ; le produit `G̃ = T̃_n···T̃_1` a `det = ±∏k ≠ 0` et **une seule
matrice inverse** `G̃⁻¹ = T̃_1⁻¹···T̃_n⁻¹` ramène l'état final à l'état initial,
sans mémoire externe. Démonstration sur une vraie partie (Stockfish, mat en
23 demi-coups, 3 prises) : `rang(G) = 49/64` sur les cases seules, mais
`det(G̃) = +1` et `G̃⁻¹·x_final = x_0` exactement.

> L'irréversibilité des échecs n'est pas dans le jeu, elle est dans la
> représentation : il suffit de donner une adresse aux morts.

### 4 ter. Sans cimetière : l'inverse est une relation — analyse rétrograde

Dans la réalité on ne dispose que de l'état final. L'inverse d'un coup n'est
alors plus une *fonction* mais une **relation** (`retro.py`) :

```
b_prev = R · b + c · e_t        c ∈ {0} ∪ {pièces restaurables}
```

`R` défait le déplacement ; le paramètre libre `c` est exactement
l'information détruite par la prise, qui réapparaît en arrière sous forme de
**branches de l'arbre rétrograde** (dé-coup simple, dé-prise ×5, dé-promotion,
dé-roque, dé-en-passant). `predecessors(b, mover)` énumère ces préimages et
ne garde que celles dont le coup avant est légal (filtres matriciels :
adversaire non en échec dans la préimage, joueur non en échec après).

`connect(départ, final, p)` relie deux états distants de `p` demi-coups par
**rencontre au milieu** : BFS avant sur les coups légaux ∩ BFS arrière sur
les préimages — et compte les *histoires distinctes* compatibles.

Démonstration (`demo_retro.py`) sur une vraie partie (mat en 23 demi-coups) :

- état final seul : 33 préimages immédiates ; le rembobinage de 4 demi-coups
  retrouve **exactement les coups réellement joués** (une seule histoire
  reliait les deux états) ;
- partie entière remontée jusqu'à `b0` en 12 segments (« en plusieurs fois ») ;
- **mat imposé** : prédicats matriciels `is_smothered_mate` (échec de cavalier
  seul ∧ voisinage du roi tout ami) et `is_pawn_mate` ; depuis la position du
  legs de Philidor, 26 suites de 3 demi-coups aboutissent à un mat étouffé,
  et la recherche rétrograde reconnecte chaque mat trouvé à la position de
  départ. Idem pour un mat par pion (10 suites).

Tests : `test_retro.py` (complétude sur parties aléatoires, soundness,
prédicats, reconnexion rejouable).

### 4 quater. Imposition générale d'une condition finale (`impose.py`)

Le rembobinage devient un **planificateur générique** : `impose(position,
prédicat, p)` fonctionne depuis n'importe quelle position et pour n'importe
quel prédicat (boîte noire `Game → bool`) :

1. **Synthèse des buts** : le plateau final n'est pas connu — il est inventé
   par éditions minimales du plateau courant (≤ k pièces déplacées vers la
   zone du roi cible, coups d'attente du défenseur, pièces sacrifiées),
   filtrées par conditions nécessaires matricielles (échec ∧ aucune fuite)
   puis par le prédicat lui-même. Matériel conservé : rien n'apparaît.
2. **Reliaison rétrograde asymétrique** : l'avant (coups légaux) est développé
   une seule fois par profondeur et partagé entre tous les buts ; chaque but
   ne coûte qu'un niveau de préimages `b_prev = R·b + c·e_t`. Rencontre à p−1.

Vérifié : le mécanisme **redécouvre** le mat étouffé de Philidor et le mat
par pion à partir du seul prédicat (aucun plateau final fourni), trouve les
mats en 1 génériques, et se scanne sur les positions d'une vraie partie
(`demo_impose.py`).

## 5. Transition plateau → plateau en calcul rapide

`transition_matrix(b1, b2, game1)` décide si `b2` est atteignable depuis `b1`
en un coup légal :

1. `Δ = b2 − b1` (une soustraction vectorielle) ; un coup touche **au plus 4
   cases** (roque) → si `|support(Δ)| ∉ [2, 4]`, rejet immédiat en O(64) ;
2. sinon, on compare `Δ` aux matrices des coups légaux de `b1` (candidats déjà
   filtrés par le support) et on retourne la matrice `T` du coup, ou `None`.

La démo rejette une dame téléportée et un roi supprimé, et retrouve `T` pour
tout successeur légal.

## 6. Validation empirique

- `test_engine.py` : 31 tests unitaires — masques de chaque pièce (centre, bord,
  coin, blocages), échec, clouage, mat du berger… pardon, du sot (fool's mate),
  pat, roque (dont interdit à travers échec), promotion, linéarité/composition/
  singularité des matrices, inversion, transitions.
- `validate.py` : parties aléatoires jouées en parallèle avec **python-chess** ;
  à chaque position, l'ensemble des coups légaux (UCI) et le verdict
  (mat/pat/en cours) doivent être **strictement identiques**.
  Résultat : 100 % d'accord sur des dizaines de milliers de positions.
- Vérification adversariale multi-agents : fuzzing ciblé par pièce/règle
  (en passant découvrant un échec, roque côté dame avec b1 attaquée, double
  échec, mats par promotion, inversion sur parties aléatoires…).

Bilan des campagnes : **31 tests unitaires + 38 113 positions (200 parties
aléatoires) + 45 530 tests adversariaux multi-agents = 0 divergence** avec
python-chess (coups légaux et verdicts strictement identiques).

## 7. Utilisation

```bash
venv/bin/python -m unittest src.matrix_chess.test_engine -v   # tests unitaires
venv/bin/python -m src.matrix_chess.validate 50 0              # vs python-chess
venv/bin/python -m src.matrix_chess.demo                       # 5 coups, G, inversion, transitions
venv/bin/python -m src.matrix_chess.webapp                     # UI web → http://127.0.0.1:8000
```

### UI web (`webapp.py`)

- **Partie** à 0/1/2 joueurs humains — l'autre camp est **Stockfish** (ou
  aléatoire) ; chaque coup s'affiche avec sa matrice `T` (formule + heatmap
  64×64), le panneau global montre `G = T_n···T_1`, son rang et la
  vérification `G·b0 = plateau courant`.
- **Undo matriciel** : applique `R + correction des prises` (formule affichée) ;
  « Retour à b₀ » remonte toute la partie par composition des R.
- **Bac à sable** : placer des pièces arbitraires sur un plateau vide et
  visualiser les masques matriciels de déplacement (bleu) / prise (rouge),
  avec la matrice T de chaque coup candidat.

> **Prochaine étape (GPU)** : tous les masques sont écrits sur les deux derniers
> axes `(..., 8, 8)` — remplacer `numpy` par `torch` et empiler N plateaux suffit
> pour explorer l'arbre de jeu en parallèle et chercher le meilleur chemin vers
> la victoire (le batch de `legal_moves` en est déjà la démonstration CPU).
