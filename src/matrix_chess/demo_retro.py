"""Démo d'analyse rétrograde : rembobiner une vraie partie, imposer un mat.

1. Joue une partie rapide (Stockfish vs aléatoire) jusqu'au mat.
2. Ne garde que l'ÉTAT FINAL : le classifie (prédicats matriciels), puis
   rembobine par préimages b_prev = R·b + c·e_t jusqu'à l'état « actuel »
   (position 4 demi-coups plus tôt), en comptant les histoires possibles.
3. Rembobine TOUTE la partie en plusieurs fois (segments), du mat jusqu'à b0.
4. Mat imposé : depuis une position donnée, trouve les suites qui terminent
   par un MAT ÉTOUFFÉ (Philidor) ou un MAT PAR PION, puis vérifie que la
   recherche rétrograde reconnecte le mat trouvé à la position de départ.

Lancer : venv/bin/python -m src.matrix_chess.demo_retro
"""

import numpy as np

from .engine import Game, to_uci
from .invert_full_game import play_quick_game
from .retro import connect, is_pawn_mate, is_smothered_mate, predecessors


def replay(ucis):
    g = Game()
    boards = [g.board.copy()]
    for u in ucis:
        g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
        boards.append(g.board.copy())
    return boards, g


def mates_matching(start, pred, plies):
    """BFS avant : toutes les suites de `plies` demi-coups menant à un mat
    vérifiant `pred`. Retourne [(ucis, Game_final)]."""
    frontier = [(start, [])]
    for _ in range(plies):
        nxt = []
        for g, path in frontier:
            for m in g.legal_moves():
                h = Game(g.board.copy(), g.turn, g.castling, g.ep_square)
                h.push(m)
                nxt.append((h, path + [to_uci(m)]))
        frontier = nxt
    return [(path, g) for g, path in frontier if pred(g)]


def main():
    # ---------- 1. partie rapide ----------
    ucis, result, term = play_quick_game()
    boards, g_final = replay(ucis)
    n = len(ucis)
    print(f"Partie : {n} demi-coups, {result} ({term})")

    # ---------- 2. depuis l'état final SEUL ----------
    b_final, turn_final = boards[-1], g_final.turn
    kind = ("mat étouffé" if is_smothered_mate(g_final)
            else "mat par pion" if is_pawn_mate(g_final) else "mat (autre pièce)")
    preds = predecessors(b_final, -turn_final)
    print(f"\nÉtat final seul : {kind} ; {len(preds)} préimages immédiates")
    print("(l'inverse n'est pas une fonction : chaque prise possible = une branche)")

    cur = Game(boards[n - 4], turn=1 if (n - 4) % 2 == 0 else -1)
    path, hist = connect(cur, b_final, turn_final, 4)
    print(f"Rembobinage mat → état 'actuel' (−4 demi-coups) : chemin {path}")
    print(f"  chemin réel : {ucis[n-4:]} ; histoires distinctes reliant les deux états : {hist}")

    # ---------- 3. toute la partie, en plusieurs fois ----------
    print(f"\nRembobinage complet {n} demi-coups → b0, par segments :")
    k, total_hist = n, 1
    while k > 0:
        step = min(2, k)
        a = Game(boards[k - step], turn=1 if (k - step) % 2 == 0 else -1)
        path, hist = connect(a, boards[k], 1 if k % 2 == 0 else -1, step)
        assert path is not None
        total_hist *= max(hist, 1)
        k -= step
    print(f"  ✓ remonté jusqu'à b0 ({(n + 1) // 2} segments) ; produit des branches "
          f"rencontrées : ≥ {total_hist} histoires compatibles")

    # ---------- 4a. imposer un MAT ÉTOUFFÉ (legs de Philidor) ----------
    philidor = np.array([
        [0, 0, 0, 0, 0, -4, 0, -6],
        [0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 5, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1 * 6, 0],
    ])
    X = Game(philidor, turn=1, castling="")
    found = mates_matching(X, is_smothered_mate, 3)
    print(f"\nMat étouffé imposé depuis la position de Philidor : "
          f"{len(found)} suite(s) en 3 demi-coups")
    line, gm = found[0]
    print(f"  ex. : {line}")
    path, hist = connect(X, gm.board, gm.turn, 3)
    print(f"  reconnexion rétrograde mat → position de départ : {path} "
          f"({hist} histoires)")

    # ---------- 4b. imposer un MAT PAR PION ----------
    Y = Game(np.array([
        [-4, 0, 0, 0, 0, 0, 0, -6],
        [0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 6, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]), turn=1, castling="")
    found = mates_matching(Y, is_pawn_mate, 3)
    print(f"\nMat par pion imposé : {len(found)} suite(s) en 3 demi-coups")
    line, gm = found[0]
    print(f"  ex. : {line}")
    path, hist = connect(Y, gm.board, gm.turn, 3)
    print(f"  reconnexion rétrograde : {path} ({hist} histoires)")


if __name__ == "__main__":
    main()
