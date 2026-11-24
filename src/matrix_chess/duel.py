"""Duel entre deux réglages de notre moteur (ancien contre nouveau), ouvertures appariées.

N ouvertures équilibrées (P demi-coups tirés au hasard à graine fixe, gardées si Stockfish les
juge à moins de --equilibre cp) sont jouées deux fois, couleurs inversées : 2N parties. Pendule
--clock + --inc par camp, arbitrage par python-chess, journal coup par coup en JSON Lines dans
results/matrix_chess/duel.jsonl. Le résumé donne le score du moteur B et l'écart d'Elo avec un
intervalle de confiance à 95 %.

python -m src.matrix_chess.duel --a "prune=1,waves=hybride" --b "prune=1,waves=adaptatif" -n 100 --clock 10 --inc 0.1
"""

import argparse
import json
import math
import os
import random
import time

import chess
import chess.engine
import torch

from . import gpu_engine as ge
from .engine import to_uci
from .match import STOCKFISH
from .validate import ref_to_matrix

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "matrix_chess", "duel.jsonl")


def reglage(texte):
    """« prune=1,waves=hybride,qdepth=4 » -> kwargs du Searcher."""
    out = {}
    for kv in filter(None, texte.split(",")):
        k, v = kv.split("=")
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    if "prune" in out:
        out["prune"] = bool(out["prune"])
    return out


def ouvertures(n, graine, plies, equilibre, profondeur=12):
    """n listes de coups UCI, distinctes, jugées équilibrées par Stockfish."""
    rng, vues, out = random.Random(graine), set(), []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH) as sf:
        while len(out) < n:
            b = chess.Board()
            for _ in range(plies):
                b.push(rng.choice(list(b.legal_moves)))
            if b.is_game_over() or b.fen() in vues:
                continue
            cp = sf.analyse(b, chess.engine.Limit(depth=profondeur))["score"].white().score(mate_score=10_000)
            if abs(cp) <= equilibre:
                vues.add(b.fen())
                out.append([m.uci() for m in b.move_stack])
    return out


def jouer(moteurs, debut, clock, inc):
    """moteurs : {chess.WHITE: Searcher, chess.BLACK: Searcher}. Renvoie (résultat, raison, plateau, journal)."""
    board = chess.Board()
    for u in debut:
        board.push_uci(u)
    clocks = {chess.WHITE: clock, chess.BLACK: clock}
    journal = []
    while not board.is_game_over(claim_draw=True):
        s = moteurs[board.turn]
        g = ref_to_matrix(board)
        B, cr, ep = ge.from_game(g, s.dev)
        t0 = time.perf_counter()
        move, info = s.search(B, cr, ep, g.turn, clocks[board.turn] / 30 + 0.8 * inc,
                              hard=clocks[board.turn] / 10 + inc)
        dt = time.perf_counter() - t0
        clocks[board.turn] -= dt
        if clocks[board.turn] <= 0:
            return ("0-1" if board.turn == chess.WHITE else "1-0"), "temps", board, journal
        clocks[board.turn] += inc
        uci = to_uci(move)
        journal.append(dict(coup=uci, score=info.get("score"), depth=info.get("depth"), nodes=info.get("nodes"),
                            temps=round(dt, 3), pendule=round(clocks[board.turn], 2)))
        board.push_uci(uci)
    out = board.outcome(claim_draw=True)
    return out.result(), out.termination.name.lower(), board, journal


def elo(score, n, var):
    """Écart d'Elo (B contre A) et intervalle à 95 % par la méthode delta sur le score moyen."""
    def f(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / p - 1)
    p = score / n
    se = math.sqrt(var / n)
    return f(p), f(p - 1.96 * se), f(p + 1.96 * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="prune=1,waves=hybride", help="réglage du moteur A (référence)")
    ap.add_argument("--b", default="prune=1,waves=hybride", help="réglage du moteur B (candidat)")
    ap.add_argument("-n", type=int, default=100, help="ouvertures (2 parties chacune)")
    ap.add_argument("--clock", type=float, default=10.0)
    ap.add_argument("--inc", type=float, default=0.1)
    ap.add_argument("--plies", type=int, default=6, help="demi-coups aléatoires de l'ouverture")
    ap.add_argument("--equilibre", type=int, default=60, help="|éval Stockfish| max de l'ouverture (cp)")
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--tag", default="duel")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, Bm = ge.Searcher(dev, **reglage(a.a)), ge.Searcher(dev, **reglage(a.b))
    ouv = ouvertures(a.n, a.graine, a.plies, a.equilibre)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    scores = []
    for i, debut in enumerate(ouv):
        for b_blanc in (False, True):
            moteurs = {chess.WHITE: Bm if b_blanc else A, chess.BLACK: A if b_blanc else Bm}
            res, raison, board, journal = jouer(moteurs, debut, a.clock, a.inc)
            s = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
            sb = s if b_blanc else 1 - s
            scores.append(sb)
            rec = dict(tag=a.tag, ouverture=i, b_blanc=b_blanc, resultat=res, raison=raison, score_b=sb,
                       demi_coups=board.ply(), debut=debut, coups=[m.uci() for m in board.move_stack],
                       journal=journal, a=a.a, b=a.b, clock=a.clock, inc=a.inc,
                       device=torch.cuda.get_device_name() if dev == "cuda" else "cpu")
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n, tot = len(scores), sum(scores)
            var = sum((x - tot / n) ** 2 for x in scores) / max(n - 1, 1)
            e, lo, hi = elo(tot, n, var)
            print(f"[{n}/{2 * len(ouv)}] ouv {i} B {'blancs' if b_blanc else 'noirs'} {res} ({raison}) | "
                  f"B : {tot:g}/{n} = {100 * tot / n:.1f} % | Elo B-A {e:+.0f} [{lo:+.0f}, {hi:+.0f}]", flush=True)


if __name__ == "__main__":
    main()
