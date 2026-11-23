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
python -m unittest src.matrix_chess.test_engine.TestKing -v # une classe de tests
python -m src.matrix_chess.validate [n_games] [seed]        # validation croisée vs python-chess (coups légaux, nulles)
python docs/rapport/gen_figures.py                          # régénère figs/ et exemples.tex du rapport (matplotlib)
cd docs/rapport && latexmk -pdf rapport.tex                 # compile le rapport LaTeX (latexmk -c pour nettoyer)
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
- `retro.py` — analyse rétrograde : sans cimetière l'inverse d'un coup est une *relation* (`b_prev = R·b + c·e_t`), les prises réapparaissent comme branches. Recherche par rencontre au milieu (préimages arrière ∩ coups légaux avant). Contient aussi les prédicats de condition finale.
- `impose.py` — impose une condition finale arbitraire (prédicat `Game -> bool`) depuis n'importe quelle position : synthèse de plateaux finaux candidats autour du roi cible, puis reliaison par `retro.connect`.
- `invert_full_game.py` — inversion exacte d'une partie complète (espace étendu 94).
- `validate.py` — compare les coups légaux du moteur à python-chess (`ref_to_matrix` convertit un `chess.Board`) ; à relancer après toute modification des règles.
- `webapp.py` — FastAPI, parties 0/1/2 joueurs vs Stockfish (repli aléatoire), affichage des matrices T / opérateur global G, undo matriciel, bac à sable. `demo*.py` / `pieces_doc.py` : démos et génération de doc.

## Notes

- Les phases 1/2 ajoutent leur dossier au `sys.path` à la main (`REPO_ROOT`/`DATA_DIR`) : les lancer par chemin de script, pas en `-m`. `matrix_chess` au contraire utilise des imports relatifs : toujours `python -m` depuis la racine.
- `Game(...)` copié à la main (`Game(g.board.copy(), g.turn, ...)`) repart avec un historique et un `halfmove` vierges : passer `halfmove=` si le compteur compte.
- Pas de venv ni de lanceur de tests dans le dépôt : installer `requirements.txt` avant `unittest`. Le rapport (`docs/rapport/`) est écrit en LaTeX ; `exemples.tex` et `figs/` sont générés, ne pas les éditer à la main.
- Résultats/articles de la phase 1 dans `results/phase1_mlp/`.
