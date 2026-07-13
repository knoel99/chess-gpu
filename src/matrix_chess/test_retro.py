"""Tests de l'analyse rétrograde : préimages, prédicats, reconnexion.

Lancer : venv/bin/python -m unittest src.matrix_chess.test_retro -v
"""

import random
import unittest

import numpy as np

from .engine import Game, to_uci
from .retro import (checker_kinds, connect, is_pawn_mate, is_smothered_mate,
                    predecessors)


def play_random(n, seed):
    rng = random.Random(seed)
    g = Game()
    boards = [g.board.copy()]
    for _ in range(n):
        ms = g.legal_moves()
        if not ms:
            break
        g.push(rng.choice(sorted(ms, key=to_uci)))
        boards.append(g.board.copy())
    return boards


class TestPredecessors(unittest.TestCase):
    def test_completeness_random_game(self):
        """Le vrai prédécesseur figure toujours parmi les préimages."""
        boards = play_random(40, seed=11)
        for i in range(1, len(boards)):
            mover = 1 if (i - 1) % 2 == 0 else -1
            preds = predecessors(boards[i], mover)
            self.assertTrue(any(np.array_equal(p, boards[i - 1]) for p, _ in preds),
                            f"demi-coup {i}")

    def test_soundness(self):
        """Chaque préimage rejoue bien vers le plateau courant via le coup fourni."""
        boards = play_random(20, seed=5)
        b = boards[-1]
        mover = 1 if (len(boards) - 2) % 2 == 0 else -1
        for prev, mv in predecessors(b, mover):
            # en passant : la case ep est celle d'arrivée du coup
            ep = (mv[1] // 8, mv[1] % 8) if mv[3] is not None else None
            g = Game(prev.copy(), turn=mover, castling="KQkq", ep_square=ep)
            legal = g.legal_moves()
            self.assertIn(mv, legal)
            g.push(next(m for m in legal if m == mv))
            self.assertTrue(np.array_equal(g.board, b))


class TestPredicates(unittest.TestCase):
    def board(self, pieces):
        b = np.zeros((8, 8), dtype=np.int64)
        for r, c, v in pieces:
            b[r, c] = v
        return b

    def test_smothered_mate(self):
        # legs de Philidor : Rg8 Kh8 / pions g7 h7, Nf7#
        b = self.board([(0, 6, -4), (0, 7, -6), (1, 6, -1), (1, 7, -1),
                        (1, 5, 2), (7, 6, 6)])
        g = Game(b, turn=-1, castling="")
        self.assertTrue(is_smothered_mate(g))
        self.assertFalse(is_pawn_mate(g))

    def test_pawn_mate(self):
        # Kf7 + pion g7 contre Kh8 + pion h7 : mat par pion
        b = self.board([(0, 7, -6), (1, 7, -1), (1, 6, 1), (1, 5, 6)])
        g = Game(b, turn=-1, castling="")
        self.assertTrue(is_pawn_mate(g))
        self.assertFalse(is_smothered_mate(g))

    def test_checker_kinds(self):
        b = self.board([(4, 4, 6), (4, 0, -4), (2, 3, -2)])  # tour + cavalier
        self.assertEqual(sorted(checker_kinds(b, 1)), [2, 4])


class TestImpose(unittest.TestCase):
    """Le mécanisme général (prédicat seul, aucun plateau final fourni)
    doit redécouvrir les mats connus."""

    def test_philidor_smothered_rediscovered(self):
        from .impose import impose
        X = Game(np.array([
            [0, 0, 0, 0, 0, -4, 0, -6],
            [0, 0, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 5, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 6, 0]]), turn=1, castling="")
        path, board = impose(X, is_smothered_mate, max_plies=3)
        self.assertIsNotNone(path)
        # la ligne rejouée aboutit bien à un mat étouffé
        for u in path:
            X.push(next(m for m in X.legal_moves() if to_uci(m) == u))
        self.assertTrue(is_smothered_mate(X))

    def test_pawn_mate_rediscovered(self):
        from .impose import impose
        Y = Game(np.array([
            [-4, 0, 0, 0, 0, 0, 0, -6],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 6, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]), turn=1, castling="")
        path, board = impose(Y, is_pawn_mate, max_plies=3)
        self.assertIsNotNone(path)
        for u in path:
            Y.push(next(m for m in Y.legal_moves() if to_uci(m) == u))
        self.assertTrue(is_pawn_mate(Y))

    def test_mate_in_one_generic(self):
        from .impose import impose
        # dame + roi contre roi : mat en 1 (Qb7# ou similaire)
        b = np.zeros((8, 8), dtype=np.int64)
        b[0, 0], b[2, 2], b[2, 1] = -6, 6, 5  # Ka8, Kc6, Qb6
        g = Game(b, turn=1, castling="")
        path, board = impose(g, lambda x: x.status() == "checkmate", max_plies=1)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)


class TestConnect(unittest.TestCase):
    def test_reconnects_real_game(self):
        boards = play_random(12, seed=2)
        start = Game(boards[8], turn=1)
        path, hist = connect(start, boards[12], 1, 4)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(hist, 1)
        # le chemin trouvé est rejouable et aboutit au bon plateau
        g = Game(boards[8].copy(), turn=1, castling="")
        for u in path:
            g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
        self.assertTrue(np.array_equal(g.board, boards[12]))


if __name__ == "__main__":
    unittest.main()
