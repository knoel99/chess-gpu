"""Tests unitaires : équivalent matriciel des mouvements de chaque pièce, puis règles.

Lancer : venv/bin/python -m unittest src.matrix_chess.test_engine -v
"""

import unittest

import numpy as np

from .engine import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
                     Game, in_check, invert_game, move_matrix, piece_moves,
                     reverse_matrix, sq, transition_matrix)


def board_with(*pieces):
    b = np.zeros((8, 8), dtype=np.int64)
    for r, c, v in pieces:
        b[r, c] = v
    return b


def targets(board, r, c, ep=None):
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(piece_moves(board, r, c, ep)))}


class TestKing(unittest.TestCase):
    def test_center_8_squares(self):
        b = board_with((4, 3, KING))
        self.assertEqual(len(targets(b, 4, 3)), 8)

    def test_corner_3_squares(self):
        b = board_with((0, 0, KING))
        self.assertEqual(targets(b, 0, 0), {(0, 1), (1, 0), (1, 1)})

    def test_blocked_by_own_capture_enemy(self):
        b = board_with((4, 3, KING), (4, 4, PAWN), (3, 3, -PAWN))
        t = targets(b, 4, 3)
        self.assertNotIn((4, 4), t)  # pièce amie bloque
        self.assertIn((3, 3), t)     # pièce ennemie prenable


class TestKnight(unittest.TestCase):
    def test_center_8_squares(self):
        b = board_with((4, 3, KNIGHT))
        self.assertEqual(targets(b, 4, 3),
                         {(2, 2), (2, 4), (3, 1), (3, 5), (5, 1), (5, 5), (6, 2), (6, 4)})

    def test_corner_2_squares(self):
        b = board_with((0, 0, KNIGHT))
        self.assertEqual(targets(b, 0, 0), {(1, 2), (2, 1)})

    def test_jumps_over_pieces(self):
        b = board_with((4, 3, KNIGHT), (3, 3, PAWN), (4, 4, -PAWN), (3, 2, PAWN), (5, 3, -PAWN))
        self.assertEqual(len(targets(b, 4, 3)), 8)  # rien ne bloque un cavalier


class TestRook(unittest.TestCase):
    def test_empty_board_14_squares(self):
        b = board_with((4, 3, ROOK))
        self.assertEqual(len(targets(b, 4, 3)), 14)

    def test_blocking(self):
        b = board_with((4, 3, ROOK), (4, 5, PAWN), (1, 3, -PAWN))
        t = targets(b, 4, 3)
        self.assertIn((4, 4), t)      # jusqu'à la pièce amie exclue
        self.assertNotIn((4, 5), t)
        self.assertNotIn((4, 6), t)   # derrière le blocage
        self.assertIn((1, 3), t)      # prise de la pièce ennemie
        self.assertNotIn((0, 3), t)   # mais pas au-delà


class TestBishop(unittest.TestCase):
    def test_center_13_squares(self):
        b = board_with((4, 3, BISHOP))
        self.assertEqual(len(targets(b, 4, 3)), 13)

    def test_blocking(self):
        b = board_with((4, 3, BISHOP), (2, 1, -PAWN))
        t = targets(b, 4, 3)
        self.assertIn((2, 1), t)
        self.assertNotIn((1, 0), t)


class TestQueen(unittest.TestCase):
    def test_center_27_squares(self):
        b = board_with((4, 3, QUEEN))
        self.assertEqual(len(targets(b, 4, 3)), 27)  # 14 tour + 13 fou


class TestPawn(unittest.TestCase):
    def test_single_and_double_push(self):
        b = board_with((6, 3, PAWN))
        self.assertEqual(targets(b, 6, 3), {(5, 3), (4, 3)})

    def test_double_blocked_at_distance(self):
        b = board_with((6, 3, PAWN), (4, 3, -PAWN))
        self.assertEqual(targets(b, 6, 3), {(5, 3)})

    def test_fully_blocked(self):
        b = board_with((6, 3, PAWN), (5, 3, -PAWN))
        self.assertEqual(targets(b, 6, 3), set())

    def test_captures_diagonal_only(self):
        b = board_with((4, 3, PAWN), (3, 2, -PAWN), (3, 4, -PAWN), (3, 3, -PAWN))
        self.assertEqual(targets(b, 4, 3), {(3, 2), (3, 4)})

    def test_black_moves_down(self):
        b = board_with((1, 3, -PAWN))
        self.assertEqual(targets(b, 1, 3), {(2, 3), (3, 3)})

    def test_en_passant(self):
        b = board_with((3, 3, PAWN), (3, 4, -PAWN))
        self.assertIn((2, 4), targets(b, 3, 3, ep=(2, 4)))


class TestRules(unittest.TestCase):
    def test_check_detection(self):
        b = board_with((4, 4, KING), (4, 0, -ROOK))
        self.assertTrue(in_check(b, WHITE))
        b2 = board_with((4, 4, KING), (4, 0, -ROOK), (4, 2, PAWN))  # blocage
        self.assertFalse(in_check(b2, WHITE))

    def test_pinned_piece_cannot_move(self):
        # tour blanche clouée entre son roi et une tour noire
        b = board_with((4, 4, KING), (4, 2, ROOK), (4, 0, -ROOK), (0, 0, -KING))
        g = Game(b, turn=WHITE, castling="")
        rook_moves = [m for m in g.legal_moves() if m[0] == sq(4, 2)]
        self.assertTrue(all(m[1] in {sq(4, 0), sq(4, 1), sq(4, 3)} for m in rook_moves))

    def test_king_cannot_move_into_check(self):
        b = board_with((4, 4, KING), (3, 0, -ROOK), (0, 7, -KING))
        g = Game(b, turn=WHITE, castling="")
        king_targets = {m[1] for m in g.legal_moves() if m[0] == sq(4, 4)}
        self.assertTrue(all(t // 8 != 3 for t in king_targets))  # la rangée 3 est balayée

    def test_fools_mate(self):
        g = Game()
        for uci_move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            g.push(next(m for m in g.legal_moves() if _uci(m) == uci_move))
        self.assertEqual(g.status(), "checkmate")

    def test_stalemate(self):
        b = board_with((0, 0, KING), (2, 1, -QUEEN), (7, 7, -KING))
        g = Game(b, turn=WHITE, castling="")
        self.assertEqual(g.status(), "stalemate")

    def test_castling_kingside(self):
        b = Game.START.copy()
        for r, c in [(7, 5), (7, 6)]:
            b[r, c] = 0
        g = Game(b, turn=WHITE)
        self.assertIn("e1g1", {_uci(m) for m in g.legal_moves()})
        g.push(next(m for m in g.legal_moves() if _uci(m) == "e1g1"))
        self.assertEqual(g.board[7, 6], KING)
        self.assertEqual(g.board[7, 5], ROOK)
        self.assertEqual(g.board[7, 7], 0)

    def test_no_castling_through_check(self):
        b = Game.START.copy()
        for r, c in [(7, 5), (7, 6), (6, 5), (1, 5)]:
            b[r, c] = 0
        b[3, 5] = -ROOK  # contrôle la colonne f traversée par le roi
        g = Game(b, turn=WHITE)
        self.assertNotIn("e1g1", {_uci(m) for m in g.legal_moves()})

    def test_promotion(self):
        b = board_with((1, 0, PAWN), (7, 4, KING), (0, 7, -KING))
        g = Game(b, turn=WHITE, castling="")
        g.push(next(m for m in g.legal_moves() if _uci(m) == "a7a8q"))
        self.assertEqual(g.board[0, 0], QUEEN)


class TestMoveMatrices(unittest.TestCase):
    def test_move_is_linear_operator(self):
        g = Game()
        flat = g.board.reshape(64)
        T = move_matrix(sq(6, 4), sq(4, 4))  # e2e4
        after = (T @ flat).reshape(8, 8)
        self.assertEqual(after[6, 4], 0)
        self.assertEqual(after[4, 4], PAWN)

    def test_composition_equals_sequence(self):
        g1, g2 = Game(), Game()
        mats = []
        for uci_move in ["e2e4", "e7e5", "g1f3"]:
            mv = next(m for m in g1.legal_moves() if _uci(m) == uci_move)
            mats.append(g1.push(mv))
        G = mats[2] @ mats[1] @ mats[0]  # opérateur global de la séquence
        self.assertTrue(np.array_equal((G @ g2.board.reshape(64)).reshape(8, 8), g1.board))

    def test_capture_matrix_is_singular(self):
        T = move_matrix(sq(4, 3), sq(3, 3))
        self.assertEqual(np.linalg.matrix_rank(T), 63)  # l'information de la prise est détruite

    def test_reverse_restores_quiet_move(self):
        flat = Game.START.reshape(64).astype(np.float64)
        T = move_matrix(sq(6, 4), sq(4, 4))
        R = reverse_matrix(T)
        self.assertTrue(np.array_equal(np.rint(R @ (T @ flat)), flat))

    def test_invert_game_with_capture(self):
        g = Game()
        mats, caps = [], []
        for uci_move in ["e2e4", "d7d5", "e4d5"]:  # 3e coup : prise
            mv = next(m for m in g.legal_moves() if _uci(m) == uci_move)
            overwritten = [(mv[1], int(g.board.reshape(64)[mv[1]]))]
            if mv[3] is not None:
                overwritten.append((mv[3], int(g.board.reshape(64)[mv[3]])))
            mats.append(g.push(mv))
            caps.append(overwritten)
        self.assertTrue(np.array_equal(invert_game(g.board, mats, caps), Game.START))

    def test_transition_legal_vs_illegal(self):
        g = Game()
        b1 = g.board.copy()
        b2 = b1.copy()
        b2[6, 4], b2[4, 4] = 0, PAWN  # e2e4 : légal
        self.assertIsNotNone(transition_matrix(b1, b2, Game()))
        b3 = b1.copy()
        b3[6, 4], b3[3, 4] = 0, PAWN  # e2 → e5 : impossible en un coup
        self.assertIsNone(transition_matrix(b1, b3, Game()))
        b4 = b1.copy()
        b4[7, 3], b4[4, 0] = 0, QUEEN  # dame traverse ses propres pions : illégal
        self.assertIsNone(transition_matrix(b1, b4, Game()))


def _uci(move):
    from .engine import to_uci
    return to_uci(move)


if __name__ == "__main__":
    unittest.main()


def _play(g, *ucis):
    for u in ucis:
        g.push(next(m for m in g.legal_moves() if _uci(m) == u))


class TestAutomaticDraws(unittest.TestCase):
    def test_insufficient_material_cases(self):
        K, k = KING, -KING
        cases = [
            ([(7, 4, K), (0, 4, k)], True),                                  # R vs R
            ([(7, 4, K), (0, 4, k), (4, 4, KNIGHT)], True),                  # R+C vs R
            ([(7, 4, K), (0, 4, k), (4, 4, -BISHOP)], True),                 # R vs R+F
            ([(7, 4, K), (0, 4, k), (4, 4, BISHOP), (3, 3, -BISHOP)], True),  # fous même couleur
            ([(7, 4, K), (0, 4, k), (4, 4, BISHOP), (3, 4, -BISHOP)], False),  # couleurs opposées
            ([(7, 4, K), (0, 4, k), (4, 4, KNIGHT), (3, 3, -KNIGHT)], False),
            ([(7, 4, K), (0, 4, k), (4, 4, PAWN)], False),
            ([(7, 4, K), (0, 4, k), (4, 4, ROOK)], False),
            ([(7, 4, K), (0, 4, k), (4, 4, QUEEN)], False),
        ]
        for pieces, expected in cases:
            g = Game(board_with(*pieces), castling="")
            self.assertEqual(g.insufficient_material(), expected, pieces)

    def test_halfmove_counter_resets(self):
        g = Game()
        _play(g, "g1f3", "g8f6", "f3g1")
        self.assertEqual(g.halfmove, 3)
        _play(g, "e7e5")          # poussée de pion
        self.assertEqual(g.halfmove, 0)
        _play(g, "e2e4", "f6e4")  # e2e4 remet à 0, la prise aussi
        self.assertEqual(g.halfmove, 0)

    def test_fifty_move_rule(self):
        b = board_with((7, 4, KING), (0, 4, -KING), (7, 0, ROOK), (0, 7, -ROOK))
        g = Game(b, castling="", halfmove=99)
        self.assertFalse(g.fifty_moves())
        _play(g, "a1a2")
        self.assertTrue(g.fifty_moves())
        self.assertEqual(g.outcome(), "fifty")

    def test_checkmate_beats_fifty(self):
        g = Game(halfmove=99)
        _play(g, "f2f3", "e7e5", "g2g4", "d8h4")
        g.halfmove = 100
        self.assertEqual(g.outcome(), "checkmate")

    def test_threefold_repetition(self):
        g = Game()
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        _play(g, *cycle)
        self.assertFalse(g.threefold())   # 2e occurrence
        _play(g, *cycle)
        self.assertTrue(g.threefold())    # 3e occurrence
        self.assertEqual(g.outcome(), "threefold")

    def test_castling_rights_distinguish_positions(self):
        # même plateau mais droit de roque perdu puis retrouvé impossible : pas de 3e occurrence
        g = Game()
        _play(g, "g1f3", "g8f6", "h1g1", "h8g8", "g1h1", "g8h8")
        self.assertEqual(g.history.count(g.history[-1]), 1)

    def test_ep_square_ignored_when_not_capturable(self):
        a, b = Game(), Game()
        _play(a, "e2e4")                                  # ep_square e3 mais aucun pion noir adjacent
        b.board = a.board.copy(); b.turn = a.turn; b.castling = a.castling
        self.assertEqual(a.position_key(), b.position_key())

    def test_matches_python_chess_on_random_games(self):
        import random

        import chess

        from .engine import to_uci
        rng = random.Random(3)
        for _ in range(6):
            ref, g = chess.Board(), Game()
            for _ in range(200):
                moves = sorted(m.uci() for m in ref.legal_moves)
                if not moves:
                    break
                self.assertEqual(
                    (g.halfmove, g.threefold(), g.insufficient_material()),
                    (ref.halfmove_clock, ref.is_repetition(3), ref.is_insufficient_material()))
                u = rng.choice(moves)
                ref.push(chess.Move.from_uci(u))
                g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
