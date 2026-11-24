# Journal des études — chess-gpu

Archive chronologique des études du projet. Chaque étude garde son dossier daté, avec son rapport
complet, ses figures et ses données d'origine, intacts. Ce journal les relie : pour chacune, la
question posée, la méthode, les chiffres clés, ce qu'elle a appris et ce qu'elle a laissé ouvert.
Les documents de théorie, qui restent des références vivantes, demeurent dans `docs/`.

| Date | Étude | Dossier | Rapport |
|---|---|---|---|
| 2026-02-15 | Phase 1 : imitation des GM par un MLP | [`2026-02-15_phase1-mlp/`](2026-02-15_phase1-mlp/) | [`results.md`](2026-02-15_phase1-mlp/results.md) |
| 2026-11-23 | Rapport sur le moteur matriciel (générateur de figures seul) | [`2026-11-23_rapport-moteur-matriciel/`](2026-11-23_rapport-moteur-matriciel/) | source LaTeX absente du dépôt |
| 2026-11-24 | Incrément 1 : moteur matriciel sur GPU, alpha-bêta par lots, matchs contre Stockfish | [`2026-11-24_increment1-gpu/`](2026-11-24_increment1-gpu/) | [`rapport_gpu.pdf`](2026-11-24_increment1-gpu/rapport_gpu.pdf) |
| à venir | Incrément 2 : tri des coups, vagues adaptatives, temps, filtre de qualité, matchs statistiques | `2026-11-2x_increment2-…/` | en cours |

Documents de théorie, par date de création : [`docs/model.md`](../model.md) et
[`docs/king_matrix.md`](../king_matrix.md) (2026-02-15), [`docs/matrix_chess.md`](../matrix_chess.md)
(2026-07-13).

---

## 2026-02-15 — Phase 1 : imitation des Grands Maîtres par un MLP

**Question.** Un réseau entraîné à imiter les coups des GM peut-il jouer correctement aux échecs ?

**Méthode.** 208 588 parties de 10 grands joueurs téléchargées depuis l'API Chess.com, soit 17,7 M
positions. Chaque position est encodée en un vecteur one-hot de 832 dimensions, et le
dictionnaire compte 1 968 coups. Un MLP à 4 couches est entraîné en 2 min 48 s sur RTX 6000.
L'évaluation se fait contre Stockfish 14 bridé à 1350 Elo, à 0,1 s par coup, en deux modes :
coup instantané, et recherche α-β à profondeur 4 avec 1 000 nœuds.

**Résultats.**
- Précision top-1 de 24,8 %, top-5 de 52,5 %.
- Mode instantané : 0,5/10, Elo estimé à environ 838.
- Mode recherche α-β : 0/10, Elo estimé à environ 950.

**Ce qu'elle a appris.** L'imitation donne des coups plausibles, mais aucun plan. Sans évaluation
fiable, l'α-β amplifie les erreurs du réseau au lieu de les corriger.

**Laissé ouvert.** Une fonction d'évaluation et une recherche dignes de ce nom. C'est ce qui a
mené au moteur matriciel.

---

## 2026-02-15 → 2026-07-13 — Théorie du moteur matriciel

- `docs/king_matrix.md` : un mouvement est un décalage de matrice ; le roi est une convolution
  3×3.
- `docs/matrix_chess.md` : toutes les règles en opérations matricielles. Chaque coup est une
  matrice $T$ telle que $b' = T\,b$, son inversion est affine, et l'analyse rétrograde passe par
  la relation $b_{prev} = R\,b + c\,e_t$.

Ces documents restent la référence de `src/matrix_chess/`.

---

## 2026-11-23 — Rapport sur le moteur matriciel

Le dépôt ne contient que le générateur de figures et d'exemples chiffrés, calculés depuis le
vrai moteur. La source LaTeX `rapport.tex` et les sorties générées n'ont jamais été commitées. Le
générateur est archivé tel quel ; il écrit `figs/` et `exemples.tex` à côté de lui.

---

## 2026-11-24 — Incrément 1 : le moteur matriciel sur GPU

**Questions.** Combien de positions une RTX PRO 6000 peut-elle simuler, et peut-on aller plus vite
que Stockfish ? Comment obtenir un équivalent de l'alpha-bêta en pur calcul matriciel, avec un
critère de convergence ?

**Méthode.** La génération des coups légaux tourne entièrement par lots sur GPU : 70 plans de
décalages, un test d'échec vu depuis le roi, `torch.compile`, puis un noyau CUDA fusionné. Une
recherche negamax pleine largeur, puis un alpha-bêta par lots (fenêtres et vagues de fils), avec
un test d'égalité stricte avec le minimax. Les matchs se jouent contre Stockfish 16 local, d'abord
en 3+0, puis dans ses conditions de calibration (60 s + 0,6 s, 1 thread, Hash 16 Mo). Trois agents
indépendants ont été consultés : algébrique, par lots, et avocat du diable. Leurs hypothèses ont
été mesurées sur GPU.

**Résultats.**

| Mesure | Valeur |
|---|---|
| Perft(6) Kiwipete | 5,1e10 nœuds/s, soit 157 fois Stockfish sur 1 thread |
| Recherche pleine largeur | 2,5e8 nœuds/s, 4 à 6 demi-coups ; 5,5/14 en conditions de calibration |
| Alpha-bêta hybride | 0,05 % des nœuds sur Kiwipete en profondeur 4 ; +2 à +3 demi-coups à temps égal ; **7,5/14**, dont 1,5/2 contre SF 2200 |
| Contre Stockfish sur toute la machine (30 threads) | débit brut à 1,3×, 7 demi-coups contre 65, 0/2 |

**Ce qu'elle a appris.**
- Le débit ne fait pas la force. L'élagage domine, et la profondeur est le facteur limitant : la
  perte Stockfish par coup passe de 71 cp en profondeur 4 à 20 cp en profondeur 8.
- Sur GPU, moins de nœuds ne veut pas dire plus rapide. Il faut des lots assez gros.
- Un bug de choix du coup racine n'a été attrapé que par les matchs. Il est désormais couvert par
  un test.

**Laissé ouvert, et repris par l'incrément 2.**
1. Un filtre de qualité déterministe.
2. Le tri des coups : killer et historique.
3. Des vagues adaptatives.
4. Une gestion du temps qui garde le résultat partiel.
5. Le refus de répétition quand on mène.
6. Des matchs statistiques de 200 parties.
7. Plus tard : une table de transposition, et le volet rétrograde sur GPU (imposer une condition
   finale comme le mat étouffé).

**Données figées.** `donnees/summary.csv` résume 70 parties, et `donnees/games.txt` contient 62
parties rejouées et vérifiées. Les coups des 8 parties v3 ont été perdus avec le disque du pod.
`ab_gpu.csv` et `perft_gpu.csv` complètent les mesures. Pour régénérer :
`python docs/etudes/2026-11-24_increment1-gpu/gen_figures.py`.

Les nouvelles mesures repartent dans `results/matrix_chess/`, écrites par `match.py`.
