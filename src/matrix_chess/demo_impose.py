"""Démo d'imposition générale : sur une vraie partie, à chaque position
testée, on impose une condition finale (prédicat pur) et le système
synthétise ses propres buts puis les relie par rembobinage.

Lancer : venv/bin/python -m src.matrix_chess.demo_impose
"""

import time

from .engine import Game, to_uci
from .impose import impose
from .invert_full_game import play_quick_game
from .retro import is_pawn_mate, is_smothered_mate


def main():
    ucis, result, term = play_quick_game(seed=7)
    print(f"Partie support : {len(ucis)} demi-coups, {result} ({term})\n")
    g = Game()
    states = [Game(g.board.copy(), g.turn, g.castling, g.ep_square)]
    for u in ucis:
        g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
        states.append(Game(g.board.copy(), g.turn, g.castling, g.ep_square))

    conditions = [("mat (quelconque)", lambda x: x.status() == "checkmate"),
                  ("mat par pion", is_pawn_mate),
                  ("mat étouffé", is_smothered_mate)]
    for k in range(6, len(ucis), 4):
        print(f"— position au demi-coup {k} (trait aux "
              f"{'blancs' if states[k].turn == 1 else 'noirs'}) :")
        for name, pred in conditions:
            t0 = time.time()
            path, _ = impose(states[k], pred, max_plies=3, max_changes=2)
            dt = time.time() - t0
            print(f"    {name:18s} → " +
                  (f"IMPOSABLE en {len(path)} demi-coups : {path}" if path
                   else "pas trouvé (≤3 demi-coups, ≤2 éditions)") + f"  [{dt:.0f}s]")
        print()


if __name__ == "__main__":
    main()
