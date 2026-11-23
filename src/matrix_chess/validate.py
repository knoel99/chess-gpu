"""Validation croisée : à chaque position de parties aléatoires, l'ensemble des
coups légaux du moteur matriciel doit être identique à celui de python-chess.

Lancer : venv/bin/python -m src.matrix_chess.validate [n_games] [seed]
"""

import random
import sys

import chess
import numpy as np

from .engine import Game, to_uci


def ref_to_matrix(ref):
    """Convertit un chess.Board python-chess vers notre représentation."""
    b = np.zeros((8, 8), dtype=np.int64)
    for square, piece in ref.piece_map().items():
        r, c = 7 - square // 8, square % 8
        b[r, c] = piece.piece_type * (1 if piece.color else -1)
    castling = "".join(ch for ch in "KQkq" if ch in ref.castling_xfen())
    ep = None
    if ref.ep_square is not None and ref.has_legal_en_passant():
        ep = (7 - ref.ep_square // 8, ref.ep_square % 8)
    return Game(b, turn=1 if ref.turn else -1, castling=castling, ep_square=ep,
                halfmove=ref.halfmove_clock)


def validate(n_games=50, seed=0, max_plies=200):
    rng = random.Random(seed)
    positions = 0
    for i in range(n_games):
        ref = chess.Board()
        live = Game()  # suit la partie coup par coup : compteur et historique
        for _ in range(max_plies):
            g = ref_to_matrix(ref)
            ours = {to_uci(m) for m in g.legal_moves()}
            theirs = {m.uci() for m in ref.legal_moves}
            if ours != theirs:
                print(f"MISMATCH partie {i}, fen {ref.fen()}")
                print(f"  nous seulement : {sorted(ours - theirs)}")
                print(f"  eux seulement  : {sorted(theirs - ours)}")
                return False
            # verdict fin de partie identique
            status = g.status()
            if ref.is_checkmate() != (status == "checkmate") or \
               ref.is_stalemate() != (status == "stalemate"):
                print(f"VERDICT DIFFÉRENT partie {i}, fen {ref.fen()}: {status}")
                return False
            # nulles automatiques identiques
            if (live.halfmove, live.threefold(), live.insufficient_material()) != \
               (ref.halfmove_clock, ref.is_repetition(3), ref.is_insufficient_material()):
                print(f"NULLE DIFFÉRENTE partie {i}, fen {ref.fen()}")
                return False
            positions += 1
            if not theirs:
                break
            uci = rng.choice(sorted(theirs))
            ref.push(chess.Move.from_uci(uci))
            live.push(next(m for m in live.legal_moves() if to_uci(m) == uci))
    print(f"OK — {n_games} parties, {positions} positions, "
          f"ensembles de coups légaux 100% identiques à python-chess")
    return True


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sys.exit(0 if validate(n, s) else 1)
