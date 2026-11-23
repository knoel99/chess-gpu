"""Moteur réduit en PyTorch : même calcul que reduced.py, sur des tenseurs, pour GPU NVIDIA
(cible : RTX PRO 6000 Blackwell, 96 Go). Tout reste sur le périphérique des tenseurs d'entrée
(cuda ou cpu) : le même code passe ses tests sur CPU.

- plateaux int8 (N, 65) : 65 octets par plateau (64 cases + la sentinelle NONE = 64, toujours nulle) ;
  les étiquettes de G_k tiennent aussi en int8 (src ≤ 64, |coef| ≤ 5) ;
- coups int64 (N, 6) au format de reduced.pack : f, t, k, ec, f2, t2 ;
- la génération pseudo-légale reste sur CPU (Game.pseudo_legal) : le GPU applique les coups
  candidats et écarte, par lots, ceux qui laissent le roi en échec.

Tests  : python -m unittest src.matrix_chess.test_reduced_gpu -v
Débit  : python -m src.matrix_chess.reduced_gpu [n_candidats]   (défaut 1 000 000)
"""

import torch

from .engine import (BISHOP, DIAG_DIRS, KING, KING_DIRS, KNIGHT, KNIGHT_DIRS, ORTHO_DIRS, PAWN,
                     QUEEN, ROOK, WHITE)


def apply_moves(boards, m):
    """b' = T @ b par ligne : un gather (b[f], b[f2]) puis un scatter hors place sur [t, t2, f, ec, f2]."""
    got = boards.gather(1, m[:, [0, 4]])
    zero = torch.zeros_like(got[:, 0])
    vals = torch.stack([m[:, 2].to(boards.dtype) * got[:, 0], got[:, 1], zero, zero, zero], 1)
    # indices en double seulement sur la colonne 64, toujours avec la valeur 0 : résultat déterministe
    return boards.scatter(1, m[:, [1, 5, 0, 3, 4]], vals)


def compose(src, coef, m):
    """G ← T @ G sur (src, coef), comme reduced.compose."""
    m1 = m.clone()
    m1[:, 2] = 1
    return apply_moves(src, m1), apply_moves(coef, m)


# --- portage torch de engine.shift / ray_attacks / step_attacks / attacked_by / in_check ---

def shift(m, dr, dc):
    """Décale les deux derniers axes de (dr, dc), remplissage à zéro (bords clippés)."""
    out = torch.zeros_like(m)
    out[..., max(0, dr):8 - max(0, -dr), max(0, dc):8 - max(0, -dc)] = \
        m[..., max(0, -dr):8 - max(0, dr), max(0, -dc):8 - max(0, dc)]
    return out


def ray_attacks(pos, empty, dirs):
    acc = torch.zeros_like(pos)
    for dr, dc in dirs:
        front = shift(pos, dr, dc)
        for _ in range(7):
            acc = acc | front
            front = shift(front & empty, dr, dc)
    return acc


def step_attacks(pos, dirs):
    acc = torch.zeros_like(pos)
    for dr, dc in dirs:
        acc = acc | shift(pos, dr, dc)
    return acc


def attacked_by(board, color):
    """Masque booléen (N, 8, 8) des cases attaquées par `color`."""
    empty = board == 0
    acc = step_attacks(board == color * PAWN, [(-color, -1), (-color, 1)])
    acc = acc | step_attacks(board == color * KNIGHT, KNIGHT_DIRS)
    acc = acc | step_attacks(board == color * KING, KING_DIRS)
    acc = acc | ray_attacks((board == color * BISHOP) | (board == color * QUEEN), empty, DIAG_DIRS)
    acc = acc | ray_attacks((board == color * ROOK) | (board == color * QUEEN), empty, ORTHO_DIRS)
    return acc


def in_check(board, color):
    """Booléen (N,) : le roi de `color` est-il attaqué ?"""
    return ((board == color * KING) & attacked_by(board, -color)).flatten(-2).any(-1)


def legal_mask(boards, m, color):
    """Booléen (N,) : le coup laisse-t-il hors d'échec le roi de `color` (le camp qui joue) ?"""
    return ~in_check(apply_moves(boards, m)[:, :64].reshape(-1, 8, 8), color)


if __name__ == "__main__":
    import random
    import statistics
    import sys
    import time

    import numpy as np

    from . import engine, reduced

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # lot réaliste : positions d'une partie aléatoire, chacune avec tous ses coups pseudo-légaux
    g, rng, rows, moves = engine.Game(), random.Random(0), [], []
    for _ in range(60):
        if g.turn == WHITE:  # legal_mask prend un seul camp par lot
            pseudo = g.pseudo_legal()
            rows += [reduced.extend(g.board)] * len(pseudo)
            moves += pseudo
        legal = g.legal_moves()
        if not legal:
            break
        g.push(rng.choice(legal))
    i = np.arange(n) % len(moves)
    b = torch.from_numpy(np.concatenate(rows)[i]).to(dev, torch.int8)
    m = torch.from_numpy(reduced.pack(moves)[i]).to(dev)

    ref = reduced.apply_moves(b[:1000].cpu().numpy().astype(np.int64), m[:1000].cpu().numpy())
    assert (apply_moves(b[:1000], m[:1000]).cpu().numpy() == ref).all()
    assert (legal_mask(b[:1000], m[:1000], WHITE).cpu().numpy()
            == ~engine.in_check(ref[:, :64].reshape(-1, 8, 8), WHITE)).all()

    def timed(fn):
        runs = []
        for _ in range(5):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            runs.append(time.perf_counter() - t0)
        return statistics.median(runs)

    print(f"{dev} : {n} candidats ({len(moves)} distincts, trait aux blancs)")
    print(f"apply_moves : {n / timed(lambda: apply_moves(b, m)):.3e} candidats/s (médiane de 5)")
    print(f"legal_mask  : {n / timed(lambda: legal_mask(b, m, WHITE)):.3e} candidats/s (médiane de 5)")
    if dev == "cuda":
        print(f"pic mémoire : {torch.cuda.max_memory_allocated() / n:.0f} octets par candidat")
