"""Tests du moteur réduit (reduced.py) : mêmes résultats que les matrices 64×64 du moteur.

Lancer : venv/bin/python -m unittest src.matrix_chess.test_reduced -v
"""

import collections
import random
import unittest

import numpy as np

from . import reduced
from .engine import QUEEN, Game, move_matrix, to_uci

# lignes construites qui finissent par un coup spécial : petit roque, grand roque, en passant,
# sous-promotion cavalier avec prise
LINES = ["g1f3 g8f6 e2e4 e7e5 f1c4 f8c5 e1g1",
         "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1",
         "e2e4 a7a6 e4e5 d7d5 e5d6",
         "a2a4 b7b5 a4b5 a7a6 b5a6 g8f6 a6a7 h7h6 a7b8n"]


def T(m):
    return move_matrix(m[0], m[1], promo=m[2], extra_clear=m[3], also=m[4])


def snapshot(g):
    return Game(g.board.copy(), g.turn, g.castling, g.ep_square, g.halfmove)


def play(ucis=None, seed=None):
    """Coups UCI imposés, ou 60 demi-coups aléatoires. Retourne (positions, coups) : positions[i] avant coups[i]."""
    rng, g, states, moves = random.Random(seed), Game(), [], []
    for i in range(len(ucis) if ucis else 60):
        legal = g.legal_moves()
        if not legal:
            break
        m = next(m for m in legal if to_uci(m) == ucis[i]) if ucis else rng.choice(legal)
        states.append(snapshot(g))
        moves.append(m)
        g.push(m)
    return states + [snapshot(g)], moves


GAMES = [play(seed=s) for s in range(3)] + [play(line.split()) for line in LINES]
POSITIONS = [g for states, _ in GAMES for g in states]


def candidates(positions):
    """Tous les coups pseudo-légaux des positions en un seul lot : plateaux (N, 65) et coups du moteur."""
    pseudo = [g.pseudo_legal() for g in positions]
    boards = np.concatenate([np.repeat(reduced.extend(g.board), len(p), axis=0) for g, p in zip(positions, pseudo)])
    return boards, [m for p in pseudo for m in p]


class TestReduced(unittest.TestCase):
    def test_shapes(self):
        b = Game().board
        for x in (b, b.reshape(64), b[None], b.reshape(1, 64)):
            np.testing.assert_array_equal(reduced.extend(x), np.append(b.reshape(64), 0)[None])
        self.assertEqual(reduced.pack([]).shape, (0, 6))

    def test_apply_moves_equals_T_b(self):
        boards, moves = candidates(POSITIONS)
        ref = np.stack([T(m) @ row[:64] for row, m in zip(boards, moves)])
        np.testing.assert_array_equal(reduced.apply_moves(boards, reduced.pack(moves)), reduced.extend(ref))

    def test_legal_moves(self):
        for g in POSITIONS:
            self.assertEqual(reduced.legal_moves(g), g.legal_moves())

    def test_compose_equals_product(self):
        for states, moves in GAMES:
            b0 = reduced.extend(states[0].board)[0]
            G, (src, coef) = np.eye(64, dtype=np.int64), reduced.identity()
            for m, after in zip(moves, states[1:]):
                G = T(m) @ G
                src, coef = reduced.compose(src, coef, reduced.pack([m]))
                np.testing.assert_array_equal(reduced.to_dense(src[0], coef[0]), G)
                np.testing.assert_array_equal(coef[0] * b0[src[0]], reduced.extend(after.board)[0])

    def test_coverage(self):
        seen = collections.Counter()
        for states, moves in GAMES:
            for g, (f, t, promo, ec, also) in zip(states, moves):
                seen["petit roque"] += also is not None and t % 8 == 6
                seen["grand roque"] += also is not None and t % 8 == 2
                seen["en passant"] += ec is not None
                seen["sous-promotion avec prise"] += promo not in (None, QUEEN) and g.board.reshape(64)[t] != 0
        for kind in ("petit roque", "grand roque", "en passant", "sous-promotion avec prise"):
            self.assertGreater(seen[kind], 0, kind)


if __name__ == "__main__":
    unittest.main()
