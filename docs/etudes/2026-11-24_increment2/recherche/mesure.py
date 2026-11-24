"""Tri des coups et vagues : nœuds, temps, branchement effectif et taux de coupe par l'aîné, score identique exigé.

python mesure.py prof D VARIANTE…     profondeurs 1..D, une ligne JSON par (position, variante, d)
python mesure.py temps S VARIANTE…    profondeur atteinte en S secondes
VARIANTE = vagues:tri[:w_hist[:s0,s1,s2]]   (ex. ybw:statique, adaptatif:historique:400:3e4,3e6,3e8)
À lancer depuis la racine du dépôt (PYTHONPATH=.).
"""
import json
import sys
import time

import chess
import torch

from src.matrix_chess import gpu_engine as ge
from src.matrix_chess.validate import ref_to_matrix

FENS = {"départ": chess.STARTING_FEN,
        "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "italienne": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 4 5",
        "gambit-dame": "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "finale-tours": "8/5pk1/6p1/1R5p/7P/6P1/r4PK1/8 w - - 0 40"}


def searcher(dev, spec):
    w, t, *rest = spec.split(":")
    s = ge.Searcher(dev, qdepth=4, prune=True, waves=w, tri=t)
    if rest:
        s.w_hist, s.journal = abs(float(rest[0])), float(rest[0]) >= 0  # négatif : sans compteurs de coupe
    if len(rest) > 1:
        s.seuils = tuple((float(x), sch) for x, (_, sch) in zip(rest[1].split(","), ge.Searcher.SEUILS))
    return s


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    mode, x, specs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    searchers = {spec: searcher(dev, spec) for spec in specs}
    for name, fen in FENS.items():
        g = ref_to_matrix(chess.Board(fen))
        B, cr, ep = ge.from_game(g, dev)
        if mode == "temps":
            for spec, s in searchers.items():
                t = time.perf_counter()
                _, info = s.search(B, cr, ep, g.turn, budget=x, hard=3 * x)
                print(json.dumps(dict(pos=name, var=spec, budget=x, wall=round(time.perf_counter() - t, 3),
                                      **{k: info.get(k) for k in ("depth", "nodes", "score", "aborted")})), flush=True)
            continue
        ref, prev = {}, {}
        for d in range(1, int(x) + 1):
            for spec, s in searchers.items():
                sync()
                t = time.perf_counter()
                mv, info = s.search(B, cr, ep, g.turn, budget=1e9, min_depth=d, max_depth=d)
                sync()
                dt, st = time.perf_counter() - t, s.stats()
                assert abs(info["score"] - ref.setdefault(d, info["score"])) < 1e-3, (name, d, spec, info, ref)
                nd, nd1 = prev.get(spec, (0, 0))  # recherche reproductible : N_d = cumul(d) − cumul(d − 1)
                n_iter = info["nodes"] - nd
                prev[spec] = (info["nodes"], n_iter)
                print(json.dumps(dict(pos=name, var=spec, d=d, nodes=info["nodes"], t=round(dt, 3), score=info["score"],
                                      n_iter=n_iter, ebf=round(n_iter / nd1, 2) if nd1 else None,
                                      first_rate=round(st["first_rate"], 4), by_ply=st["first_rate_by_ply"])), flush=True)
