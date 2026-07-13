"""Démo : partie de 5 coups en opérateurs matriciels, inversion, transitions.

1. Joue 5 coups (ouverture italienne) : chaque coup k est une matrice T_k ∈ Z^{64×64}
2. Compose l'opérateur global G = T_10 ··· T_1 et vérifie G @ b0 = b5
3. Inverse : montre que G est singulière (les prises détruisent l'information),
   puis reconstruit b0 exactement par retour affine b_{k-1} = R_k @ b_k + correction
4. Transition : décide par calcul vectoriel rapide si un plateau b' est atteignable
   en un coup légal depuis b, et donne la matrice T correspondante

Lancer : venv/bin/python -m src.matrix_chess.demo
"""

import numpy as np

from .engine import (Game, invert_game, move_matrix, reverse_matrix, to_uci,
                     transition_matrix)

SYMBOLS = {0: "·", 1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
           -1: "♟", -2: "♞", -3: "♝", -4: "♜", -5: "♛", -6: "♚"}


def show(board, title=""):
    if title:
        print(f"\n{title}")
    for i, row in enumerate(board):
        print(f"  {8 - i} " + " ".join(SYMBOLS[int(v)] for v in row))
    print("    a b c d e f g h")


def main():
    # --- 1. cinq coups complets (10 demi-coups), italienne avec une prise ---
    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "d7d5", "e4d5", "f6d5"]
    g = Game()
    b0 = g.board.copy()
    mats, caps = [], []
    print("Partie jouée (chaque coup = une matrice 64×64 appliquée au plateau) :")
    for uci in ucis:
        mv = next(m for m in g.legal_moves() if to_uci(m) == uci)
        flat = g.board.reshape(64)
        overwritten = [(mv[1], int(flat[mv[1]]))]  # ce que T écrase (prise ou 0)
        if mv[3] is not None:
            overwritten.append((mv[3], int(flat[mv[3]])))
        mats.append(g.push(mv))
        caps.append(overwritten)
        taken = [v for _, v in overwritten if v]
        print(f"  {uci}  T = I − E_ff − E_tt + E_tf" + (f"   (prise : {SYMBOLS[taken[0]]})" if taken else ""))
    b5 = g.board.copy()
    show(b5, "Position après 5 coups :")

    # --- 2. opérateur global ---
    G = np.eye(64, dtype=np.int64)
    for T in mats:
        G = T @ G
    assert np.array_equal((G @ b0.reshape(64)).reshape(8, 8), b5)
    rank = np.linalg.matrix_rank(G.astype(float))
    print(f"\nOpérateur global G = T10 ··· T1 : G @ b0 = b5 vérifié ✓")
    print(f"rang(G) = {rank}/64 → G est singulière : l'inverse matriciel exact n'existe pas.")
    print("Chaque prise (et chaque case vidée) projette de l'information à zéro ;")
    print("l'inversion est donc affine : b_(k-1) = R_k @ b_k + correction_k,")
    print("où R_k inverse le déplacement et la correction restaure les pièces écrasées.")

    # --- 3. inversion affine exacte ---
    b0_rebuilt = invert_game(b5, mats, caps)
    assert np.array_equal(b0_rebuilt, b0)
    show(b0_rebuilt, "Position initiale reconstruite depuis b5 (inversion affine) :")
    print("  b0 reconstruit == b0 exactement ✓")

    # pseudo-inverse pur (sans mémoire des prises) : ce qu'on récupère au mieux
    Gp = np.linalg.pinv(G.astype(float))
    b0_pinv = np.rint(Gp @ b5.reshape(64)).astype(int).reshape(8, 8)
    diff = int((b0_pinv != b0).sum())
    print(f"  pseudo-inverse de Moore-Penrose seul : {diff} cases fausses "
          f"(les pièces capturées sont irrécupérables sans mémoire externe)")

    # --- 4. transitions plateau → plateau par calcul vectoriel ---
    print("\nVérification de transition (diff vectoriel Δ = b2 − b1, ≤ 4 cases touchées) :")
    g5 = Game(b5, turn=g.turn, castling=g.castling, ep_square=g.ep_square)
    legal_next = g5.legal_moves()[0]
    g6 = Game(b5, turn=g.turn, castling=g.castling, ep_square=g.ep_square)
    g6.push(legal_next)
    T = transition_matrix(b5, g6.board, Game(b5, turn=g.turn, castling=g.castling,
                                             ep_square=g.ep_square))
    print(f"  b5 → b5+{to_uci(legal_next)} : transition légale, T retrouvée ✓"
          if T is not None else "  ERREUR")

    bad = b5.copy()
    bad[0, 3], bad[4, 4] = 0, -5  # la dame noire se téléporte
    T = transition_matrix(b5, bad, Game(b5, turn=g.turn, castling=g.castling,
                                        ep_square=g.ep_square))
    print("  b5 → dame téléportée : transition rejetée ✓" if T is None else "  ERREUR")

    bad2 = b5.copy()
    bad2[7, 4] = 0  # un roi disparaît (position cible illégale)
    T = transition_matrix(b5, bad2, Game(b5, turn=g.turn, castling=g.castling,
                                         ep_square=g.ep_square))
    print("  b5 → roi supprimé : transition rejetée ✓" if T is None else "  ERREUR")


if __name__ == "__main__":
    main()
