"""Matchs en cadence 3+0 : recherche GPU pleine largeur (gpu_engine.Searcher) contre Stockfish à Elo croissant.

python-chess sert d'arbitre (nulles, répétitions, mat) ; chaque camp a sa pendule de 180 s, un
dépassement perd la partie. Stockfish est bridé par UCI_LimitStrength / UCI_Elo (minimum 1320),
« max » = pleine force. Résultats ajoutés en JSON Lines dans results/matrix_chess/matches.jsonl.

python -m src.matrix_chess.match [--elos 1320,1600,2000,max] [--games 2] [--clock 180] [--threads 1]
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


def our_move(searcher, board, clock):
    g = ref_to_matrix(board)
    B, cr, ep = ge.from_game(g, searcher.dev)
    # ~30 coups restants supposés, pas d'incrément ; une itération est interrompue à clock / 10
    move, info = searcher.search(B, cr, ep, g.turn, clock / 30, hard=clock / 10)
    return chess.Move.from_uci(to_uci(move)), info


def play(searcher, sf, elo, ours_white, clock, log):
    board, clocks = chess.Board(), {chess.WHITE: clock, chess.BLACK: clock}
    sf.configure({"UCI_LimitStrength": elo != "max", **({} if elo == "max" else {"UCI_Elo": int(elo)})})
    ours = chess.WHITE if ours_white else chess.BLACK
    stats = {"ours": [], "sf": []}
    result, reason = None, None
    while not board.is_game_over(claim_draw=True):
        t0 = time.perf_counter()
        if board.turn == ours:
            move, info = our_move(searcher, board, clocks[ours])
            stats["ours"].append({k: info[k] for k in ("depth", "nodes", "nps")})
        else:
            r = sf.play(board, chess.engine.Limit(white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK]),
                        info=chess.engine.INFO_BASIC)
            move = r.move
            stats["sf"].append({"depth": r.info.get("depth", 0), "nodes": r.info.get("nodes", 0),
                                "nps": r.info.get("nps", 0)})
        clocks[board.turn] -= time.perf_counter() - t0
        if clocks[board.turn] <= 0:
            result, reason = ("0-1" if board.turn == chess.WHITE else "1-0"), "temps"
            break
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
    ap.add_argument("--clock", type=float, default=180.0)
    ap.add_argument("--threads", type=int, default=1, help="threads Stockfish")
    ap.add_argument("--qdepth", type=int, default=4)
    ap.add_argument("--tag", default="", help="étiquette de la série, recopiée dans chaque résultat")
    ap.add_argument("--stop-after", type=int, default=0, help="arrêter après N niveaux sans point marqué")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    searcher = ge.Searcher(dev, qdepth=a.qdepth)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def log(rec):
        rec["tag"] = a.tag
        print(json.dumps({k: v for k, v in rec.items() if k != "pgn_moves"}), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")

    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    sf.configure({"Threads": a.threads, "Hash": 256})
    blank = 0
    try:
        for elo in a.elos.split(","):
            pts = sum(play(searcher, sf, elo, i % 2 == 0, a.clock, log)["our_score"] for i in range(a.games))
            print(f"== Elo {elo} : {pts}/{a.games}", flush=True)
            blank = blank + 1 if pts == 0 else 0
            if a.stop_after and blank >= a.stop_after:
                break
    finally:
        sf.quit()


if __name__ == "__main__":
    main()
