"""Analyse de parties par Stockfish, à la manière de l'analyse de partie de Lichess.

Deux modes, lancés depuis la racine :

    python -m src.matrix_chess.analyse parties games.txt --sortie DOSSIER [--profondeur 16 | --noeuds N]
    python -m src.matrix_chess.analyse filtre games.txt -n 30 --depth 3 [--prune --waves hybride] --sortie f.csv

`parties` évalue chaque demi-coup de chaque partie (une ligne `tag elo couleur résultat raison
coups…`, couleur = celle de notre moteur) et écrit `coups.csv` et `parties.csv`. `filtre` tire N
positions des parties (graine fixe), fait choisir un coup à `gpu_engine.Searcher` et mesure la
perte Stockfish de ce coup : c'est le filtre de qualité déterministe à lancer avant tout match.

Formules (celles, publiques, de Lichess) :
  - pourcentage de gain  W(cp) = 50 + 50·(2 / (1 + exp(−0,00368208·cp)) − 1), cp borné à ±1000 ;
  - un mat annoncé vaut ±1000 cp, comme dans le calcul de précision de Lichess ;
  - précision d'un coup  103,1668·exp(−0,04354·ΔW) − 3,1669 + 1, bornée à [0, 100], ΔW = perte de W ;
  - imprécision / erreur / gaffe : ΔW ≥ 5 / 10 / 15 points (0,1 / 0,2 / 0,3 sur l'échelle [−1, 1]).
Écarts assumés : pas de règles spéciales « mat manqué / mat retardé » (un mat manqué n'est puni
que par sa perte de W), et la précision d'une partie est la moyenne simple des précisions par coup,
pas la moyenne pondérée par la volatilité de Lichess.
"""

import argparse
import csv
import math
import os
import random
import shutil
import time
from multiprocessing import Pool
from statistics import mean, median

import chess
import chess.engine

STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"
CP_MAX = 1000
SEUILS = ((15.0, "gaffe"), (10.0, "erreur"), (5.0, "imprecision"))
DECISIF = 60.0  # W du vainqueur à partir duquel la partie est dite jouée, voir coup_decisif


# --- formules ---

def gain(cp):
    """Pourcentage de gain (0..100) du camp pour lequel `cp` est compté."""
    cp = max(-CP_MAX, min(CP_MAX, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def precision(perte_gain):
    """Précision (0..100) d'un coup qui perd `perte_gain` points de pourcentage de gain."""
    if perte_gain <= 0:
        return 100.0
    return max(0.0, min(100.0, 103.1668100711649 * math.exp(-0.04354415386753951 * perte_gain) - 3.166924740191411 + 1))


def classe(perte_gain):
    return next((nom for seuil, nom in SEUILS if perte_gain >= seuil), "")


def juger(avant, apres, blanc):
    """Perte du camp qui joue entre deux évaluations (cp, point de vue des blancs)."""
    s = 1 if blanc else -1
    wa, wb = gain(s * avant), gain(s * apres)
    dw = max(0.0, wa - wb)
    return dict(perte_cp=max(0, s * (avant - apres)), gain_avant=round(wa, 2), gain_apres=round(wb, 2),
                perte_gain=round(dw, 2), precision=round(precision(dw), 2), classe=classe(dw))


# --- Stockfish : un moteur par processus ---

_SF = None


def _init(hash_mo):
    global _SF
    _SF = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _SF.configure({"Threads": 1, "Hash": hash_mo})


def evaluer(board, limite, game=None):
    """(cp borné point de vue des blancs, mat signé ou None, meilleur coup UCI) d'une position."""
    if board.is_checkmate():
        return (-CP_MAX if board.turn == chess.WHITE else CP_MAX), 0, ""
    if board.is_stalemate() or board.is_insufficient_material():
        return 0, None, ""
    info = _SF.analyse(board, chess.engine.Limit(**limite), game=game)
    s = info["score"].white()
    cp = max(-CP_MAX, min(CP_MAX, s.score(mate_score=CP_MAX)))
    if s.is_mate():
        cp = CP_MAX if s.mate() > 0 else -CP_MAX
    return cp, s.mate(), (info["pv"][0].uci() if info.get("pv") else "")


def evaluer_ligne(args):
    """Évalue les positions atteintes après coups[:i], i = debut..len(coups). L'historique complet
    est transmis à Stockfish, qui voit donc les répétitions ; table de hachage vidée par ligne."""
    coups, debut, limite = args
    board, game, out = chess.Board(), object(), []
    for i in range(len(coups) + 1):
        if i >= debut:
            out.append(evaluer(board, limite, game))
        if i < len(coups):
            board.push_uci(coups[i])
    return out


def evaluer_lignes(taches, procs, hash_mo=64):
    with Pool(procs, _init, (hash_mo,)) as pool:
        return pool.map(evaluer_ligne, taches, chunksize=1)


# --- mode parties ---

def lire_parties(chemin):
    parties = []
    with open(chemin) as f:
        for i, ligne in enumerate(f):
            x = ligne.split()
            if x:
                parties.append(dict(partie=i, tag=x[0], elo=x[1], ours=x[2], resultat=x[3], raison=x[4], coups=x[5:]))
    return parties


def lignes_coups(p, evals):
    """Une ligne par demi-coup, à partir des évaluations des len(coups) + 1 positions."""
    rows = []
    for i, coup in enumerate(p["coups"]):
        blanc = i % 2 == 0
        (cp0, mat0, best), (cp1, mat1, _) = evals[i], evals[i + 1]
        camp = "white" if blanc else "black"
        rows.append(dict(partie=p["partie"], tag=p["tag"], elo=p["elo"], ours=p["ours"], demi_coup=i + 1, camp=camp,
                         joueur="ours" if camp == p["ours"] else "sf", coup=coup, meilleur=best,
                         eval_avant=cp0, mat_avant="" if mat0 is None else mat0,
                         eval_apres=cp1, mat_apres="" if mat1 is None else mat1, **juger(cp0, cp1, blanc)))
    return rows


def coup_decisif(p, rows):
    """Premier demi-coup après lequel le W du vainqueur reste ≥ DECISIF jusqu'à la fin ('' si nulle)."""
    if p["resultat"] not in ("1-0", "0-1") or not rows:
        return ""
    s = 1 if p["resultat"] == "1-0" else -1
    k = len(rows)
    while k > 0 and gain(s * rows[k - 1]["eval_apres"]) >= DECISIF:
        k -= 1
    return rows[k]["demi_coup"] if k < len(rows) else ""


def synthese(p, rows):
    out = {k: p[k] for k in ("partie", "tag", "elo", "ours", "resultat", "raison")}
    out["demi_coups"] = len(rows)
    for j in ("ours", "sf"):
        r = [x for x in rows if x["joueur"] == j]
        out[f"{j}_coups"] = len(r)
        out[f"{j}_perte_cp"] = round(mean(x["perte_cp"] for x in r), 1) if r else ""
        out[f"{j}_perte_gain"] = round(mean(x["perte_gain"] for x in r), 2) if r else ""
        out[f"{j}_precision"] = round(mean(x["precision"] for x in r), 1) if r else ""
        for _, nom in reversed(SEUILS):
            out[f"{j}_{nom}s"] = sum(x["classe"] == nom for x in r)
    d = coup_decisif(p, rows)
    out["decisif_demi_coup"] = d
    out["decisif_joueur"] = rows[d - 1]["joueur"] if d else ""
    out["decisif_perte_cp"] = rows[d - 1]["perte_cp"] if d else ""
    return out


def ecrire(chemin, rows):
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
    with open(chemin, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyser_parties(parties, limite, procs):
    """(lignes par coup, lignes par partie)."""
    evals = evaluer_lignes([(p["coups"], 0, limite) for p in parties], procs)
    coups, synth = [], []
    for p, e in zip(parties, evals):
        rows = lignes_coups(p, e)
        coups += rows
        synth.append(synthese(p, rows))
    return coups, synth


# --- mode filtre ---

def tirer_positions(parties, n, graine):
    """N couples (partie, nombre de demi-coups joués) où le trait a au moins deux coups légaux."""
    cand = []
    for p in parties:
        board = chess.Board()
        for i, c in enumerate(p["coups"]):
            if board.legal_moves.count() >= 2:
                cand.append((p, i))
            board.push_uci(c)
    return random.Random(graine).sample(cand, min(n, len(cand)))


def filtrer(parties, n, graine, limite, procs, prune=False, waves="ybw", qdepth=4, depth=None, budget=1.0, tri="mvv"):
    """Lignes du filtre : le coup du Searcher sur chaque position tirée, jugé par Stockfish."""
    import torch

    from . import gpu_engine as ge
    from .engine import to_uci
    from .validate import ref_to_matrix

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(1)  # les petits lots CPU sont bien plus lents avec plusieurs threads
    searcher = ge.Searcher(dev, qdepth=qdepth, prune=prune, waves=waves, tri=tri)
    rows, taches = [], []
    for p, i in tirer_positions(parties, n, graine):
        board = chess.Board()
        for c in p["coups"][:i]:
            board.push_uci(c)
        g = ref_to_matrix(board)
        B, cr, ep = ge.from_game(g, dev)
        t0 = time.perf_counter()
        if depth:
            mv, info = searcher.search(B, cr, ep, g.turn, 0.0, hard=float("inf"), min_depth=depth, max_depth=depth)
        else:
            mv, info = searcher.search(B, cr, ep, g.turn, budget)
        coup = to_uci(mv)
        assert chess.Move.from_uci(coup) in board.legal_moves, (board.fen(), coup)
        rows.append(dict(partie=p["partie"], tag=p["tag"], demi_coup=i + 1, fen=board.fen(), coup=coup,
                         profondeur=info["depth"], noeuds=info["nodes"], temps=round(time.perf_counter() - t0, 3)))
        taches.append((p["coups"][:i] + [coup], i, limite))
    for r, ((cp0, _, best), (cp1, _, _)) in zip(rows, evaluer_lignes(taches, procs)):
        r.update(meilleur=best, eval_avant=cp0, eval_apres=cp1, **juger(cp0, cp1, r["demi_coup"] % 2 == 1))
    return rows


def resume_filtre(rows):
    pertes = [r["perte_cp"] for r in rows]
    return dict(positions=len(rows), perte_moyenne=round(mean(pertes), 1), perte_mediane=median(pertes),
                part_100cp=round(sum(x >= 100 for x in pertes) / len(pertes), 3),
                profondeur_moyenne=round(mean(r["profondeur"] for r in rows), 2),
                temps_total=round(sum(r["temps"] for r in rows), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("parties", "filtre"))
    ap.add_argument("fichier", help="parties : tag elo couleur résultat raison coups UCI…")
    ap.add_argument("--sortie", required=True, help="dossier (parties) ou fichier CSV (filtre)")
    ap.add_argument("--profondeur", type=int, default=16, help="profondeur Stockfish par position")
    ap.add_argument("--noeuds", type=int, help="nœuds Stockfish par position, à la place de la profondeur")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1, help="processus Stockfish (1 thread chacun)")
    f = ap.add_argument_group("filtre")
    f.add_argument("-n", type=int, default=30, help="positions tirées")
    f.add_argument("--graine", type=int, default=0)
    f.add_argument("--prune", action="store_true", help="alpha-bêta par lots au lieu de la pleine largeur")
    f.add_argument("--waves", default="ybw", help="avec --prune : vagues de fils (Searcher.WAVES)")
    f.add_argument("--tri", default="mvv", help="ordre des fils (Searcher.TRIS ; statique = incrément 1)")
    f.add_argument("--qdepth", type=int, default=4)
    f.add_argument("--depth", type=int, help="profondeur fixe du Searcher (sinon budget de temps)")
    f.add_argument("--budget", type=float, default=1.0, help="secondes par coup du Searcher")
    a = ap.parse_args()
    limite = {"nodes": a.noeuds} if a.noeuds else {"depth": a.profondeur}
    parties = lire_parties(a.fichier)
    t0 = time.perf_counter()
    if a.mode == "parties":
        coups, synth = analyser_parties(parties, limite, a.procs)
        ecrire(os.path.join(a.sortie, "coups.csv"), coups)
        ecrire(os.path.join(a.sortie, "parties.csv"), synth)
        print(f"{len(parties)} parties, {len(coups)} demi-coups, {limite}, {a.procs} processus, "
              f"{time.perf_counter() - t0:.0f} s")
    else:
        rows = filtrer(parties, a.n, a.graine, limite, a.procs, a.prune, a.waves, a.qdepth, a.depth, a.budget, a.tri)
        ecrire(a.sortie, rows)
        print(resume_filtre(rows))


if __name__ == "__main__":
    main()
