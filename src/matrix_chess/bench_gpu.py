"""Débit brut : perft du moteur GPU contre `go perft` de Stockfish (même position, même profondeur).

perft = nombre exact de feuilles de l'arbre des coups légaux : c'est « combien de positions on peut
simuler ». Stockfish compte en masse au dernier niveau, sur un seul thread ; nous aussi (sur GPU).

python -m src.matrix_chess.bench_gpu [profondeur] [chunk]
"""

import subprocess
import sys
import time

import chess
import torch

from . import gpu_engine as ge
from .match import STOCKFISH
from .validate import ref_to_matrix

POSITIONS = {
    "départ": chess.STARTING_FEN,
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
}


def stockfish_perft(fen, depth):
    t0 = time.perf_counter()
    out = subprocess.run([STOCKFISH], input=f"position fen {fen}\ngo perft {depth}\nquit\n",
                         capture_output=True, text=True).stdout
    dt = time.perf_counter() - t0
    return int(out.split("Nodes searched:")[1].split()[0]), dt


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else (300_000 if dev == "cuda" else 20_000)
    name = torch.cuda.get_device_name() if dev == "cuda" else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    for label, fen in POSITIONS.items():
        g = ref_to_matrix(chess.Board(fen))
        B, cr, ep = ge.from_game(g, dev)
        ge.perft(B, cr, ep, g.turn, 2, chunk)  # chauffe (allocations, noyaux)
        sync()
        t0 = time.perf_counter()
        ours = ge.perft(B, cr, ep, g.turn, depth, chunk)
        sync()
        dt = time.perf_counter() - t0
        ref, sdt = stockfish_perft(fen, depth)
        assert ours == ref, (label, ours, ref)
        print(f"{label} perft({depth}) = {ours:,} | {name} : {dt:.2f} s, {ours / dt:.3e} nœuds/s | "
              f"Stockfish 1 thread : {sdt:.2f} s, {ref / sdt:.3e} nœuds/s | ratio {sdt / dt:.2f}", flush=True)
    if dev == "cuda":
        print(f"pic mémoire GPU : {torch.cuda.max_memory_allocated() / 2**30:.1f} Go")


if __name__ == "__main__":
    main()
