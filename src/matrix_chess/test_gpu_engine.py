"""Tests de gpu_engine.py : perft de référence, coups légaux identiques à python-chess, recherche.

Tourne sur cuda si disponible, sinon sur cpu. Lancer : python -m unittest src.matrix_chess.test_gpu_engine -v
"""

import random
import unittest

import chess
import torch

from . import gpu_engine as ge
from .engine import to_uci
from .validate import ref_to_matrix

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# positions et nombres de feuilles de référence (chessprogramming.org, « Perft Results »)
PERFT = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
]


def state(fen):
    g = ref_to_matrix(chess.Board(fen))
    return (*ge.from_game(g, DEV), g.turn)


class TestGPUEngine(unittest.TestCase):
    def test_perft(self):
        for fen, counts in PERFT:
            B, cr, ep, color = state(fen)
            for d, want in enumerate(counts, 1):
                self.assertEqual(ge.perft(B, cr, ep, color, d), want, f"{fen} profondeur {d}")

    def test_legal_moves_match_python_chess(self):
        rng = random.Random(1)
        for _ in range(20):
            ref = chess.Board()
            for _ in range(rng.randint(0, 120)):
                if ref.is_game_over():
                    break
                ref.push(rng.choice(list(ref.legal_moves)))
            B, cr, ep, color = state(ref.fen())
            _, m, *_ = ge.expand(B, cr, ep, color)
            self.assertEqual(sorted(to_uci(ge.to_move(x)) for x in m.tolist()),
                             sorted(x.uci() for x in ref.legal_moves), ref.fen())

    @unittest.skipUnless(ge.gpu_kernels.available(DEV), "noyau fusionné : cuda + CuPy")
    def test_fused_kernel_matches_torch_path(self):
        rng, lots = random.Random(2), {1: [], -1: []}
        fens = [fen for fen, _ in PERFT] + ["8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 1",       # e.p. qui pare l'échec
                                           "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1"]         # e.p. cloué sur la rangée
        for _ in range(300):
            ref = chess.Board()
            for _ in range(rng.randint(0, 150)):
                if ref.is_game_over():
                    break
                ref.push(rng.choice(list(ref.legal_moves)))
                fens.append(ref.fen())
        for fen in fens:
            *st, color = state(fen)
            lots[color].append(st)
        for color, lot in lots.items():
            B, cr, ep = (torch.cat(x) for x in zip(*lot))
            for capt in (False, True):
                want = ge.expand_torch(B, cr, ep, color, capt)
                got = ge.gpu_kernels.expand(B, cr, ep, color, capt)
                for w, g, name in zip(want, got, ("parent", "coups", "B'", "cr'", "ep'")):
                    self.assertTrue(torch.equal(w, g), (name, color, capt))  # mêmes enfants, même ordre
                self.assertEqual(ge.gpu_kernels.expand(B, cr, ep, color, capt, count_only=True), len(want[0]))

    def test_search_mate_in_one_and_free_queen(self):
        s = ge.Searcher(DEV)
        for fen, want in [("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),        # mat du couloir
                          ("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "d2d5")]:     # dame en prise
            B, cr, ep, color = state(fen)
            move, info = s.search(B, cr, ep, color, budget=5.0, min_depth=2, max_depth=3)
            self.assertEqual(to_uci(move), want, (fen, info))


if __name__ == "__main__":
    unittest.main()
