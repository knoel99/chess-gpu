"""Tests de duel.py : lecture des réglages et écart d'Elo. Lancer : python -m unittest src.matrix_chess.test_duel -v"""

import unittest

from .duel import elo, reglage


class TestDuel(unittest.TestCase):
    def test_reglage(self):
        self.assertEqual(reglage("prune=1,waves=hybride,qdepth=4"), dict(prune=True, waves="hybride", qdepth=4))
        self.assertEqual(reglage(""), {})

    def test_elo(self):
        e, lo, hi = elo(100, 200, 0.2)
        self.assertAlmostEqual(e, 0.0, places=6)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        e75, _, _ = elo(150, 200, 0.2)
        self.assertAlmostEqual(e75, 190.8, places=1)  # 75 % de score ≈ +191 Elo
        _, lo2, hi2 = elo(100, 800, 0.2)
        self.assertLess(hi2 - lo2, hi - lo)  # plus de parties, intervalle plus étroit


if __name__ == "__main__":
    unittest.main()
