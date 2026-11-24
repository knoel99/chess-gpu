"""Tests du pilotage de la recherche : pendule, résultat partiel, répétitions, journal par coup des matchs.

Sur CPU, lancer avec un seul thread : OMP_NUM_THREADS=1 python -m unittest src.matrix_chess.test_search_control -v
"""

import os
import unittest

import chess
import chess.engine
import torch

from . import gpu_engine as ge
from . import match
from .engine import to_uci
from .validate import ref_to_matrix

DEV = "cuda" if torch.cuda.is_available() else "cpu"

FENS = ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 4 5",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r3k1/pp3ppp/4p3/3pP3/3P4/1q3N2/1P3PPP/2RQ2K1 b - - 0 22",
        "8/5pk1/6p1/1R5p/7P/6P1/r4PK1/8 w - - 0 40"]


def state(fen):
    g = ref_to_matrix(chess.Board(fen))
    return (*ge.from_game(g, DEV), g.turn)


def interrupt(s, depth, k):
    """Fait lever Timeout à `s` après k groupes de coups racine de l'itération `depth`."""
    real, calls = s.value, [0]

    def value(B, cr, ep, color, ply, remaining, *rest):
        if ply == 1 and remaining == depth - 1:
            calls[0] += 1
            if calls[0] > k:
                raise ge.Timeout
        return real(B, cr, ep, color, ply, remaining, *rest)

    s.value = value


class TestSearchControl(unittest.TestCase):
    def test_partial_result_is_safe(self):
        """Itération d interrompue après k groupes : le coup rendu vaut, au minimax exact de profondeur d,
        au moins le meilleur coup de la profondeur d - 1, et le score annoncé est son score exact."""
        full = ge.Searcher(DEV, qdepth=2)
        changed = 0
        for fen in FENS:
            B, cr, ep, color = state(fen)
            _, m, child, ccr, cep = ge.expand(B, cr, ep, color)
            moves = [ge.to_move(x) for x in m.tolist()]
            for d in (2, 3):
                exact = full.root_scores(child, ccr, cep, color, d, None)
                for prune in (False, True):
                    before, _ = ge.Searcher(DEV, qdepth=2, prune=prune).search(B, cr, ep, color, 1e9, max_depth=d - 1)
                    for k in (0, 1, 2, 4):
                        s = ge.Searcher(DEV, qdepth=2, prune=prune)
                        interrupt(s, d, k)
                        move, info = s.search(B, cr, ep, color, 1e9, min_depth=d - 1, max_depth=d)
                        msg = (fen, d, prune, k, info)
                        self.assertEqual((info["depth"], info["aborted"]), (d - 1, d), msg)
                        if k == 0:  # premier coup inachevé : rien n'est sûr, on garde la profondeur d - 1
                            self.assertEqual(move, before, msg)
                            self.assertNotIn("partial", info, msg)
                            continue
                        self.assertEqual(info["partial"], d, msg)
                        got = float(exact[moves.index(move)])
                        self.assertGreaterEqual(got, float(exact[moves.index(before)]), msg)
                        self.assertAlmostEqual(info["score"], got, places=3, msg=msg)
                        changed += move != before
        self.assertGreater(changed, 0, "aucun cas où le résultat partiel change le coup : test vide")

    def test_time_stays_within_budget(self):
        s = ge.Searcher(DEV, chunk=2000, qdepth=2, prune=True)  # petits lots : pendule consultée souvent
        budget = 0.4
        for fen in FENS:
            B, cr, ep, color = state(fen)
            move, info = s.search(B, cr, ep, color, budget)
            self.assertIsNotNone(move)
            self.assertGreaterEqual(info["depth"], 1, fen)
            self.assertLessEqual(info["time"], 1.5 * budget + 0.1, (fen, info))  # + un lot en cours à l'échéance

    def test_keys_match_python_chess(self):
        for fen in [chess.STARTING_FEN] + FENS:
            board = chess.Board(fen)
            B, cr, ep, color = state(fen)
            _, _, child, ccr, cep = ge.expand(B, cr, ep, color)
            want = set()
            for mv in list(board.legal_moves):
                board.push(mv)
                want.add(match.key(board))
                board.pop()
            self.assertEqual(set(ge.position_keys(child, ccr, cep, -color)), want, fen)

    def search_after(self, fen, repeated):
        """Coup et info de la recherche quand les positions après les coups UCI `repeated` ont déjà été vues."""
        board, history = chess.Board(fen), set()
        for uci in repeated:
            board.push_uci(uci)
            history.add(match.key(board))
            board.pop()
        B, cr, ep, color = state(fen)
        move, info = ge.Searcher(DEV, qdepth=2, prune=True).search(B, cr, ep, color, 1e9, max_depth=2, history=history)
        return to_uci(move), info

    def test_repetition_avoided_when_winning(self):
        fen = "6k1/8/8/8/8/8/PPP5/1K1R4 w - - 0 1"
        best, info = self.search_after(fen, [])
        self.assertGreater(info["score"], 500)
        other, info = self.search_after(fen, [best])
        self.assertNotEqual(other, best)
        self.assertGreater(info["score"], 500)

    def test_repetition_sought_when_losing(self):
        fen = "1k1r4/ppp5/8/8/8/8/8/6K1 w - - 0 1"
        _, info = self.search_after(fen, [])
        self.assertLess(info["score"], -500)
        move, info = self.search_after(fen, ["g1h1"])
        self.assertEqual((move, info["score"]), ("g1h1", 0.0))

    def test_all_moves_repeat(self):
        board = chess.Board("1k1r4/ppp5/8/8/8/8/8/6K1 w - - 0 1")
        move, info = self.search_after(board.fen(), [m.uci() for m in board.legal_moves])
        self.assertIn(chess.Move.from_uci(move), board.legal_moves)
        self.assertEqual(info["score"], 0.0)

    @unittest.skipUnless(os.path.exists(match.STOCKFISH), "Stockfish absent")
    def test_match_logs_every_move(self):
        recs = []
        sf = chess.engine.SimpleEngine.popen_uci(match.STOCKFISH)
        try:
            match.play(ge.Searcher(DEV, qdepth=2, prune=True), sf, "1320", True, 3.0, 0.0, recs.append)
        finally:
            sf.quit()
        rec = recs[0]
        for k in ("result", "our_score", "plies", "clock_left", "ours_depth", "sf_depth", "pgn_moves"):
            self.assertIn(k, rec)  # champs d'avant le journal par coup
        rows = rec["moves_info"]
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(" ".join(r["uci"] for r in rows), rec["pgn_moves"])
        self.assertEqual([r["ply"] for r in rows], list(range(1, len(rows) + 1)))
        self.assertEqual([r["side"] for r in rows[:2]], ["ours", "sf"])
        for r in rows:
            for k in ("score", "depth", "nodes", "time"):
                self.assertIsInstance(r[k], (int, float), r)
            self.assertEqual(set(r["clock"]), {"ours", "sf"})
        self.assertLess(rows[-1]["clock"]["ours"], 3.0)


if __name__ == "__main__":
    unittest.main()
