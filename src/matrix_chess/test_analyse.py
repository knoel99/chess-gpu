"""Tests de l'analyse Stockfish : formules sur des cas connus, mat du lion analysé de bout en bout."""

import unittest

from . import analyse as an


class TestFormules(unittest.TestCase):
    def test_gain(self):
        self.assertEqual(an.gain(0), 50)
        self.assertAlmostEqual(an.gain(100), 59.1, places=1)
        self.assertAlmostEqual(an.gain(100) + an.gain(-100), 100)
        self.assertEqual(an.gain(5000), an.gain(an.CP_MAX))

    def test_precision(self):
        self.assertEqual(an.precision(0), 100)
        self.assertEqual(an.precision(-3), 100)
        self.assertAlmostEqual(an.precision(10), 64.6, places=1)
        self.assertEqual(an.precision(100), 0)

    def test_classe(self):
        self.assertEqual([an.classe(x) for x in (4.9, 5, 10, 15, 40)], ["", "imprecision", "erreur", "gaffe", "gaffe"])

    def test_juger_du_camp_qui_joue(self):
        noir = an.juger(-200, 100, blanc=False)
        self.assertEqual(noir["perte_cp"], 300)
        self.assertEqual(noir["classe"], "gaffe")
        self.assertEqual(an.juger(-200, 100, blanc=True)["perte_cp"], 0)


class TestPartie(unittest.TestCase):
    def test_mat_du_lion(self):
        p = dict(partie=0, tag="t", elo="max", ours="white", resultat="0-1", raison="checkmate",
                 coups="f2f3 e7e5 g2g4 d8h4".split())
        coups, (s,) = an.analyser_parties([p], {"depth": 8}, procs=1)
        self.assertEqual([r["joueur"] for r in coups], ["ours", "sf", "ours", "sf"])
        g4, dh4 = coups[2], coups[3]
        self.assertEqual((g4["classe"], g4["mat_apres"], g4["eval_apres"]), ("gaffe", -1, -an.CP_MAX))
        self.assertEqual((dh4["meilleur"], dh4["perte_cp"], dh4["precision"], dh4["mat_apres"]), ("d8h4", 0, 100.0, 0))
        self.assertEqual((s["ours_gaffes"], s["sf_gaffes"], s["decisif_demi_coup"], s["decisif_joueur"]),
                         (1, 0, 3, "ours"))


if __name__ == "__main__":
    unittest.main()
