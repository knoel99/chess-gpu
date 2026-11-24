"""Matchs : recherche GPU pleine largeur (gpu_engine.Searcher) contre Stockfish à Elo croissant.

python-chess sert d'arbitre (nulles, répétitions, mat) ; chaque camp a sa pendule (défaut 3+0), un
dépassement perd la partie. Stockfish est bridé par UCI_LimitStrength / UCI_Elo (minimum 1320),
« max » = pleine force. Résultats ajoutés en JSON Lines dans results/matrix_chess/matches.jsonl.

Préréglages de ressources Stockfish (--preset) :
- uci-elo : conditions de calibration de UCI_Elo (wiki UCI de Stockfish : 60 s + 0,6 s, ancré sur
  CCRL 40/4) avec les réglages par défaut de Stockfish, ceux de sa commande `bench` (1 thread, Hash 16) ;
- machine : pleine capacité de la machine, cœurs séparés : Stockfish sur tous les cœurs sauf deux
  (taskset), notre moteur sur les deux restants, Hash 8 Go.

python -m src.matrix_chess.match [--preset uci-elo] [--elos 1320,1600,2000,max] [--games 2]
"""

import argparse
import json
import os
import shutil
import time

import chess
import chess.engine
import torch

from . import gpu_engine as ge
from .engine import to_uci
from .validate import ref_to_matrix

STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "matrix_chess", "matches.jsonl")


PRESETS = {
    "uci-elo": dict(clock=60.0, inc=0.6, threads=1, hash=16),
    "machine": dict(threads=max(1, (os.cpu_count() or 4) - 2), hash=8192),
}


def our_move(searcher, board, clock, inc):
    g = ref_to_matrix(board)
    B, cr, ep = ge.from_game(g, searcher.dev)
    # ~30 coups restants supposés ; une itération est interrompue à clock / 10 (+ incrément)
    move, info = searcher.search(B, cr, ep, g.turn, clock / 30 + 0.8 * inc, hard=clock / 10 + inc)
    return chess.Move.from_uci(to_uci(move)), info


def play(searcher, sf, elo, ours_white, clock, inc, log):
    board, clocks = chess.Board(), {chess.WHITE: clock, chess.BLACK: clock}
    sf.configure({"UCI_LimitStrength": elo != "max", **({} if elo == "max" else {"UCI_Elo": int(elo)})})
    ours = chess.WHITE if ours_white else chess.BLACK
    stats = {"ours": [], "sf": []}
    result, reason = None, None
    while not board.is_game_over(claim_draw=True):
        t0 = time.perf_counter()
        if board.turn == ours:
            move, info = our_move(searcher, board, clocks[ours], inc)
            stats["ours"].append({k: info[k] for k in ("depth", "nodes", "nps")})
        else:
            r = sf.play(board, chess.engine.Limit(white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK],
                                                  white_inc=inc, black_inc=inc), info=chess.engine.INFO_BASIC)
            move = r.move
            stats["sf"].append({"depth": r.info.get("depth", 0), "nodes": r.info.get("nodes", 0),
                                "nps": r.info.get("nps", 0)})
        clocks[board.turn] -= time.perf_counter() - t0
        if clocks[board.turn] <= 0:
            result, reason = ("0-1" if board.turn == chess.WHITE else "1-0"), "temps"
            break
        clocks[board.turn] += inc
        board.push(move)
    if result is None:
        out = board.outcome(claim_draw=True)
        result, reason = out.result(), out.termination.name.lower()
    score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    rec = {
        "elo": elo, "ours": "white" if ours_white else "black", "result": result, "reason": reason,
        "our_score": score if ours_white else 1 - score, "plies": board.ply(),
        "clock_left": {"ours": round(clocks[ours], 1), "sf": round(clocks[not ours], 1)},
        **{f"{who}_{k}": _mean(stats[who], k) for who in ("ours", "sf") for k in ("depth", "nps")},
        "ours_nodes_total": sum(s["nodes"] for s in stats["ours"]),
        "sf_nodes_total": sum(s["nodes"] for s in stats["sf"]),
        "pgn_moves": " ".join(m.uci() for m in board.move_stack),
        "device": torch.cuda.get_device_name() if searcher.dev == "cuda" else "cpu",
    }
    log(rec)
    return rec


def _mean(rows, k):
    return round(sum(r[k] for r in rows) / len(rows), 1) if rows else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elos", default="1320,1600,2000,2400,max")
    ap.add_argument("--games", type=int, default=2, help="parties par niveau, couleurs alternées")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="ressources Stockfish d'un banc de référence")
    ap.add_argument("--clock", type=float, default=180.0, help="secondes par camp")
    ap.add_argument("--inc", type=float, default=0.0, help="incrément par coup (s)")
    ap.add_argument("--threads", type=int, default=1, help="threads Stockfish")
    ap.add_argument("--hash", type=int, default=256, help="table de hachage Stockfish (Mo)")
    ap.add_argument("--qdepth", type=int, default=4)
    ap.add_argument("--tag", default="", help="étiquette de la série, recopiée dans chaque résultat")
    ap.add_argument("--stop-after", type=int, default=0, help="arrêter après N niveaux sans point marqué")
    a = ap.parse_args()
    for k, v in PRESETS.get(a.preset, {}).items():
        setattr(a, k, v)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    searcher = ge.Searcher(dev, qdepth=a.qdepth)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def log(rec):
        rec.update(tag=a.tag, preset=a.preset, sf_threads=a.threads, sf_hash=a.hash, clock=a.clock, inc=a.inc)
        print(json.dumps({k: v for k, v in rec.items() if k != "pgn_moves"}), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")

    cmd = STOCKFISH
    if a.preset == "machine" and hasattr(os, "sched_setaffinity") and (os.cpu_count() or 0) > 2:
        os.sched_setaffinity(0, {0, 1})  # notre moteur : cœurs 0 et 1 ; Stockfish : tous les autres
        cmd = ["taskset", "-c", f"2-{os.cpu_count() - 1}", STOCKFISH]
    sf = chess.engine.SimpleEngine.popen_uci(cmd)
    sf.configure({"Threads": a.threads, "Hash": a.hash})
    blank = 0
    try:
        for elo in a.elos.split(","):
            pts = sum(play(searcher, sf, elo, i % 2 == 0, a.clock, a.inc, log)["our_score"] for i in range(a.games))
            print(f"== Elo {elo} : {pts}/{a.games}", flush=True)
            blank = blank + 1 if pts == 0 else 0
            if a.stop_after and blank >= a.stop_after:
                break
    finally:
        sf.quit()


if __name__ == "__main__":
    main()
