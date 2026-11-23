"""Moteur réduit : mêmes résultats que les matrices 64×64 de engine.py, par écritures d'indices seulement.

T = I − E_ff − E_tt + k·E_tf (+ case vidée en passant, + tour du roque) ne modifie que 2 à 4 lignes,
donc b' = T @ b se réduit à : out[t] = k·b[f], out[f] = 0, out[ec] = 0, out[t2] = b[f2], out[f2] = 0.

Données :
- plateaux (N, 65) : les 64 cases aplaties (idx = 8r + c) plus une colonne sentinelle NONE = 64,
  toujours nulle, qui tient lieu de « aucune case » (pas d'en passant, pas de roque) : aucun
  branchement dans le code par lot ;
- coups (N, 6) int64 : f, t, k, ec, f2, t2 — k = coefficient de promotion ou 1, ec = case vidée
  en passant ou NONE, (f2, t2) = déplacement de la tour du roque ou (NONE, NONE) ;
- opérateur cumulé G_k = T_k···T_1 : deux plateaux d'étiquettes (src, coef), b_k[i] = coef[i]·b_0[src[i]]
  (G_k reste une permutation partielle pondérée : au plus un coefficient non nul par ligne).
"""

import numpy as np

from .engine import in_check

NONE = 64


def pack(moves):
    """Coups du moteur (f, t, promo, extra_clear, also) -> tableau (N, 6) int64 : f, t, k, ec, f2, t2."""
    return np.array([(f, t, promo or 1, NONE if ec is None else ec, *(also or (NONE, NONE)))
                     for f, t, promo, ec, also in moves], dtype=np.int64).reshape(-1, 6)


def extend(boards):
    """Plateaux (8, 8), (64,), (N, 8, 8) ou (N, 64) -> (N, 65), colonne sentinelle nulle ajoutée."""
    b = np.asarray(boards).reshape(-1, 64)
    return np.concatenate([b, np.zeros((len(b), 1), dtype=b.dtype)], axis=1)


def apply_moves(boards, m):
    """b' = T @ b pour chaque ligne : plateaux (N, 65), coups (N, 6) -> (N, 65), lus sur l'entrée."""
    f, t, k, ec, f2, t2 = m.T
    r = np.arange(len(m))
    out = boards.copy()
    out[r, t] = k * boards[r, f]
    out[r, f] = 0
    out[r, ec] = 0
    out[r, t2] = boards[r, f2]
    out[r, f2] = 0
    return out


def identity(n=1):
    """G_0 = I pour n lots : src = arange(65), coef = 1, sentinelle nulle dans les deux."""
    src = np.tile(np.arange(65), (n, 1))
    coef = np.ones_like(src)
    src[:, NONE] = coef[:, NONE] = 0  # la sentinelle reste nulle partout : le scatter GPU reste déterministe
    return src, coef


def compose(src, coef, m):
    """G ← T @ G sur (src, coef) : la ligne t de G reçoit k × la ligne f, les lignes vidées passent à 0.

    src contient n'importe quoi là où coef == 0 (ligne nulle de G) : sans effet sur b_k ni sur to_dense.
    """
    m1 = m.copy()
    m1[:, 2] = 1  # l'étiquette de source se déplace sans coefficient
    return apply_moves(src, m1), apply_moves(coef, m)


def to_dense(src, coef):
    """Un couple (src, coef) de forme (65,) -> matrice G 64×64 int64."""
    G = np.zeros((64, 65), dtype=np.int64)
    G[np.arange(64), src[:64]] = coef[:64]
    return G[:, :64]


def legal_moves(game):
    """Même liste que Game.legal_moves() : candidats par apply_moves, puis in_check du moteur en un lot."""
    pseudo = game.pseudo_legal()
    m = pack(pseudo)
    cand = apply_moves(np.repeat(extend(game.board), len(m), axis=0), m)
    ok = ~in_check(cand[:, :64].reshape(-1, 8, 8), game.turn)
    return [mv for mv, keep in zip(pseudo, ok) if keep]
