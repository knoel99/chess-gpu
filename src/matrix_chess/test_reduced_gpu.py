"""Tests du moteur réduit PyTorch (reduced_gpu.py) : mêmes résultats que reduced.py et que le moteur.

Tourne sur cuda si disponible, sinon sur cpu. Lancer : venv/bin/python -m unittest src.matrix_chess.test_reduced_gpu -v
"""

import unittest

import numpy as np

from . import reduced
from .engine import in_check
from .test_reduced import GAMES, POSITIONS, candidates

try:
    import torch

    from . import reduced_gpu
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch non installé")
class TestReducedGPU(unittest.TestCase):
    def test_apply_moves_and_in_check(self):
        boards, moves = candidates(POSITIONS)
        m = reduced.pack(moves)
        ref = reduced.apply_moves(boards, m)
        got = reduced_gpu.apply_moves(torch.from_numpy(boards).to(DEV, torch.int8), torch.from_numpy(m).to(DEV))
        np.testing.assert_array_equal(got.cpu().numpy(), ref)
        cand = got[:, :64].reshape(-1, 8, 8)
        for color in (1, -1):
            np.testing.assert_array_equal(reduced_gpu.in_check(cand, color).cpu().numpy(),
                                          in_check(ref[:, :64].reshape(-1, 8, 8), color))

    def test_legal_mask(self):
        for g in POSITIONS:
            boards, moves = candidates([g])
            ok = reduced_gpu.legal_mask(torch.from_numpy(boards).to(DEV, torch.int8),
                                        torch.from_numpy(reduced.pack(moves)).to(DEV), g.turn)
            self.assertEqual([m for m, keep in zip(moves, ok.tolist()) if keep], g.legal_moves())

    def test_compose(self):
        for _, moves in GAMES:
            src, coef = reduced.identity()
            ts, tc = (torch.from_numpy(a).to(DEV, torch.int8) for a in (src, coef))
            for mv in moves:
                m = reduced.pack([mv])
                src, coef = reduced.compose(src, coef, m)
                ts, tc = reduced_gpu.compose(ts, tc, torch.from_numpy(m).to(DEV))
                np.testing.assert_array_equal(ts.cpu().numpy(), src)
                np.testing.assert_array_equal(tc.cpu().numpy(), coef)


if __name__ == "__main__":
    unittest.main()
