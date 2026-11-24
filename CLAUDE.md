# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Projet en français (docs, commentaires, README) : garder cette langue. Trois volets indépendants sous `src/` :

- `src/common/` — téléchargement des parties (API chess.com) et préparation des données ; partagé par les phases 1 et 2. Données dans `data/` (ignorées par git : `*.json/pgn/npz/png`).
- `src/phase1_mlp/` (MLP numpy, `.npz`) et `src/phase2_transformer/` (Transformer torch, `.pt`) — imitation des GMs, évaluation contre Stockfish. Pensés pour Google Colab/GPU.
- `src/matrix_chess/` — moteur d'échecs 100 % matriciel (numpy), sans lien avec les réseaux. Théorie dans `docs/king_matrix.md` et `docs/matrix_chess.md`.

## Commandes

Dépendances : `pip install -r requirements.txt` (torch, python-chess, numpy, matplotlib, fastapi, uvicorn ; cupy optionnel). Stockfish doit être installé (`/usr/games/stockfish` ou `~/.local/bin/stockfish`) pour l'évaluation et l'UI web.

```bash
# Moteur matriciel (lancer depuis la racine, en modules : imports relatifs)
python -m unittest src.matrix_chess.test_engine -v          # tests moteur
python -m unittest src.matrix_chess.test_retro -v           # tests rétrograde
python -m unittest src.matrix_chess.test_reduced -v         # tests moteur réduit (équivalence avec les matrices 64×64)
python -m unittest src.matrix_chess.test_reduced_gpu -v     # idem en PyTorch (cuda si disponible, sinon cpu)
python -m src.matrix_chess.reduced_gpu [n_candidats]        # débit du moteur réduit PyTorch sur la machine
python -m unittest src.matrix_chess.test_engine.TestKing -v # une classe de tests
python -m unittest src.matrix_chess.test_gpu_engine -v      # génération GPU : perft de référence, égalité python-chess, recherche
python -m unittest src.matrix_chess.test_search_control -v # pendule, résultat partiel, répétitions, journal par coup (CPU : OMP_NUM_THREADS=1)
python -m src.matrix_chess.bench_gpu [profondeur] [chunk]   # perft GPU contre `go perft` de Stockfish
python -m src.matrix_chess.match --elos 1320,1500,max       # matchs 3+0 contre Stockfish bridé (results/matrix_chess/)
python -m src.matrix_chess.duel --a "prune=1,waves=hybride" --b "…" -n 100 --clock 10 --inc 0.1  # duel ancien/nouveau, 2N parties, Elo ± IC 95 %
python -m src.matrix_chess.validate [n_games] [seed]        # validation croisée vs python-chess (coups légaux, nulles)
python -m unittest src.matrix_chess.test_analyse -v         # analyse Stockfish : formules, partie courte de bout en bout
python -m src.matrix_chess.analyse parties games.txt --sortie DOSSIER   # évaluation Stockfish de chaque coup (coups.csv, parties.csv)
python -m src.matrix_chess.analyse filtre games.txt -n 30 --depth 3 [--prune --waves hybride] --sortie f.csv  # filtre de qualité du Searcher
python docs/etudes/2026-11-24_increment1-gpu/gen_figures.py # régénère figures et tableaux du rapport de l'incrément 1
cd docs/etudes/2026-11-24_increment1-gpu && latexmk -pdf rapport_gpu.tex  # compile ce rapport (latexmk -c pour nettoyer)
python -m src.matrix_chess.webapp                           # UI web → http://127.0.0.1:8000

# Phases réseaux (scripts directs, pas des modules)
python src/phase1_mlp/main.py --top10                       # télécharge + prépare + entraîne
python src/phase1_mlp/evaluate.py data/top_players_model.npz --games 10 [--max-depth 4 --max-nodes 1000]
python src/phase2_transformer/main.py --top10
python run_colab.py [--phase2 | --all]                      # pipeline complet (Colab)
```

## Architecture du moteur matriciel (`src/matrix_chess/`)

Lire plusieurs fichiers pour comprendre le tout :

- `engine.py` — cœur. Plateau `8×8` d'entiers signés (+1..+6 = pion..roi blancs, négatif = noirs, ligne 0 = rangée 8). Les masques d'attaque/mouvement sont faits uniquement de `shift` + opérations élémentaires, compatibles batch `(..., 8, 8)`. Chaque coup est une matrice `T ∈ Z^{64×64}` : `b' = T @ b` (`move_matrix`). `Game` porte l'état (trait, roque, en passant) ; `reverse_matrix` / `invert_game` défont les coups. Toute `T` est singulière (ligne `f` nulle) ; seule une prise perd de l'information. `Game.status()` ne connaît que mat/pat (retro et webapp en dépendent) ; les nulles automatiques (matériel insuffisant, 50 coups via `halfmove`, triple répétition via `history`) sont dans `Game.outcome()`, que la webapp n'utilise pas encore.
- `reduced.py` — forme réduite, mêmes résultats que `move_matrix` / `T @ b` sans matrice 64×64. Plateaux `(N, 65)` (colonne sentinelle `NONE = 64` toujours nulle = « aucune case »), coups `(N, 6)` = `pack(moves)`, primitive unique `apply_moves`. L'opérateur cumulé `G_k` tient en deux plateaux d'étiquettes `(src, coef)` (`compose`, `to_dense`). `engine.py` garde les matrices denses, qui restent la référence.
- `reduced_gpu.py` — même API que `reduced.py` en PyTorch (plateaux `int8`, un `gather` et un `scatter` par lot), plus le portage torch de `in_check` et `legal_mask`. Cible RTX PRO 6000 Blackwell (PyTorch ≥ 2.7 construit avec CUDA ≥ 12.8) ; jamais exécuté sur GPU à ce jour, seulement sur CPU. La génération pseudo-légale reste sur CPU.
- `gpu_engine.py` — génération de coups légaux par lots entièrement sur GPU (70 plans de coups par shifts, test d'échec depuis le roi), `perft`, et `Searcher` : negamax pleine largeur + quiescence (prises MVV/LVA), récursif par lots (mémoire bornée), interrompu à la pendule (1,5 × budget) en gardant le résultat partiel sûr ; un coup racine qui répète une position de `history` vaut 0. Avec `prune=True` : alpha-bêta exact par vagues de fils (`WAVES`, dont `adaptatif` qui choisit la découpe lot par lot selon `SEUILS`), fils triés par `tri` (`mvv` par défaut, `statique` = incrément 1) ; `Searcher.stats()` donne le taux de coupe par l'aîné. `expand_torch` est la référence et le repli CPU.
- `gpu_kernels.py` — noyau CUDA fusionné (CuPy/nvrtc, `pip install cupy-cuda12x`) pour `expand` : même résultat et même ordre que `expand_torch`, ~10× plus rapide. Sans CuPy le repli torch est silencieux.
- `analyse.py` — analyse à la Lichess par Stockfish (pourcentage de gain, précision, imprécision/erreur/gaffe) : mode `parties` (chaque demi-coup de parties déjà jouées) et mode `filtre` (perte du coup choisi par `Searcher` sur des positions tirées à graine fixe), à lancer avant tout match.
- `match.py` / `bench_gpu.py` — matchs contre Stockfish (`UCI_LimitStrength`) et banc perft ; résultats bruts dans `results/matrix_chess/matches.jsonl`.
- `retro.py` — analyse rétrograde : sans cimetière l'inverse d'un coup est une *relation* (`b_prev = R·b + c·e_t`), les prises réapparaissent comme branches. Recherche par rencontre au milieu (préimages arrière ∩ coups légaux avant). Contient aussi les prédicats de condition finale.
- `impose.py` — impose une condition finale arbitraire (prédicat `Game -> bool`) depuis n'importe quelle position : synthèse de plateaux finaux candidats autour du roi cible, puis reliaison par `retro.connect`.
- `invert_full_game.py` — inversion exacte d'une partie complète (espace étendu 94).
- `validate.py` — compare les coups légaux du moteur à python-chess (`ref_to_matrix` convertit un `chess.Board`) ; à relancer après toute modification des règles.
- `webapp.py` — FastAPI, parties 0/1/2 joueurs vs Stockfish (repli aléatoire), affichage des matrices T / opérateur global G, undo matriciel, bac à sable. `demo*.py` / `pieces_doc.py` : démos et génération de doc.

## Notes

- Les phases 1/2 ajoutent leur dossier au `sys.path` à la main (`REPO_ROOT`/`DATA_DIR`) : les lancer par chemin de script, pas en `-m`. `matrix_chess` au contraire utilise des imports relatifs : toujours `python -m` depuis la racine.
- `Game(...)` copié à la main (`Game(g.board.copy(), g.turn, ...)`) repart avec un historique et un `halfmove` vierges : passer `halfmove=` si le compteur compte.
- Pas de venv ni de lanceur de tests dans le dépôt : installer `requirements.txt` avant `unittest`.
- Études archivées par date dans `docs/etudes/` (journal chronologique : `docs/etudes/README.md`) : chaque dossier daté garde son rapport, ses figures et ses données figées (`donnees/`). Figures et tableaux sont générés par le `gen_figures.py` du dossier, ne pas les éditer à la main. Une nouvelle étude = un nouveau dossier `AAAA-MM-JJ_nom/` et une entrée dans le journal.
- Mesures en cours (matchs) : `results/matrix_chess/`, écrit par `match.py` ; on les fige dans le dossier de l'étude quand elle se clôt.
