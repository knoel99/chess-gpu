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

# toute variante exacte de l'alpha-bêta par lots : (vagues, tri)
VARIANTES = [(w, "mvv") for w in ge.Searcher.WAVES] + [("ybw", "statique"), ("hybride", "historique"), ("adaptatif", "historique")]


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

    def test_search_matches_naive_negamax(self):
        s = ge.Searcher(DEV, qdepth=1)

        def naive(fen, depth, q):  # negamax position par position, même évaluation et quiescence
            b = chess.Board(fen)
            B, cr, ep, color = state(fen)
            if depth == 0:
                stand = float(ge.evaluate(B, color, s.E)[0])
                if q == 0:
                    return stand
                moves = [m for m in b.legal_moves if b.is_capture(m) or m.promotion]
            else:
                moves = list(b.legal_moves)
            _, mv, *_ = ge.expand(B, cr, ep, color, captures_only=depth == 0)
            kept = {to_uci(ge.to_move(x)) for x in mv.tolist()}  # filtre MVV/LVA de la quiescence
            vals = []
            for m in moves:
                if m.uci() in kept:
                    b.push(m)
                    vals.append(-naive(b.fen(), max(depth - 1, 0), q - (depth == 0)))
                    b.pop()
            if depth == 0:
                return max([stand] + vals)
            if not vals:
                return -ge.MATE + 1 if b.is_check() else 0.0  # ply 1 : enfants de la racine
            return max(vals)

        for fen in ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"]:
            B, cr, ep, color = state(fen)
            _, info = s.search(B, cr, ep, color, budget=1e9, min_depth=2, max_depth=2)
            b = chess.Board(fen)
            want = max(-naive((b.push(m), b.fen(), b.pop())[1], 1, 1) for m in list(b.legal_moves))
            self.assertAlmostEqual(info["score"], want, places=3, msg=fen)

    def test_pruned_search_same_score_as_full_width(self):
        full = ge.Searcher(DEV, qdepth=2)
        pruned = [ge.Searcher(DEV, qdepth=2, prune=True, waves=w, tri=t) for w, t in VARIANTES]
        for fen, _ in PERFT + [("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8", None)]:
            B, cr, ep, color = state(fen)
            for d in (2, 3):
                _, f = full.search(B, cr, ep, color, budget=1e9, min_depth=d, max_depth=d)
                for ab in pruned:
                    _, p = ab.search(B, cr, ep, color, budget=1e9, min_depth=d, max_depth=d)
                    self.assertAlmostEqual(p["score"], f["score"], places=3, msg=(fen, d, ab.waves, ab.tri))
                    self.assertLessEqual(p["nodes"], f["nodes"], (fen, d, ab.waves, ab.tri))

    def test_pruned_search_plays_a_best_move(self):
        """Le coup joué, pas seulement le score, doit valoir le minimax exact : les autres coups racine
        n'ont que des bornes hautes, une égalité de borne ne doit pas faire choisir un coup réfuté."""
        full = ge.Searcher(DEV, qdepth=2)
        pruned = [ge.Searcher(DEV, qdepth=2, prune=True, waves=w, tri=t) for w, t in VARIANTES]
        fens = [fen for fen, _ in PERFT] + ["r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 4 5",
                                            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
                                            "8/5pk1/6p1/1R5p/7P/6P1/r4PK1/8 w - - 0 40"]
        for fen in fens:
            B, cr, ep, color = state(fen)
            _, m, child, ccr, cep = ge.expand(B, cr, ep, color)
            for d in (2, 3):
                exact = full.root_scores(child, ccr, cep, color, d, None)
                for ab in pruned:
                    move, info = ab.search(B, cr, ep, color, budget=1e9, min_depth=d, max_depth=d)
                    i = [ge.to_move(x) for x in m.tolist()].index(move)
                    self.assertAlmostEqual(float(exact[i]), float(exact.max()), places=3, msg=(fen, d, ab.waves, ab.tri))

    def test_search_mate_in_one_and_free_queen(self):
        cases = [("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),     # mat du couloir
                 ("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "d2d5")]     # dame en prise
        for s in (ge.Searcher(DEV), ge.Searcher(DEV, prune=True)):
            for fen, want in cases:
                B, cr, ep, color = state(fen)
                move, info = s.search(B, cr, ep, color, budget=5.0, min_depth=2, max_depth=3)
                self.assertEqual(to_uci(move), want, (fen, info))


if __name__ == "__main__":
    unittest.main()
