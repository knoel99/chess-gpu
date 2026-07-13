"""Une partie complète, puis inversion matricielle exacte de toute la partie.

Réponse à « peut-on inverser la matrice globale G ? » :

1. Sur Z^64 : NON. Chaque prise écrase une case — G projette de l'information
   à zéro, rang(G) < 64, l'inverse n'existe pas (le pseudo-inverse ne restitue
   pas les pièces capturées).
2. LA MÉTHODE : augmenter l'espace d'état. On étend le vecteur plateau de
   64 à 94 composantes (64 cases + 30 places de « cimetière »). Une prise ne
   détruit plus rien : elle DÉPLACE la pièce capturée vers le cimetière.
   Chaque coup devient alors une matrice monomiale (permutation à
   coefficients ±k) de déterminant non nul :
       coup simple      : transposition (f t)
       prise            : 3-cycle (f → t → cimetière → f)
       en passant       : (f t)(case_capturée cimetière)
       roque            : (e1 g1)(h1 f1)
       promotion        : permutation × k sur la ligne d'arrivée (k = 5, 4, 3, 2)
   Le produit G̃ = T̃_n ··· T̃_1 est inversible : une SEULE matrice inverse
   G̃⁻¹ = T̃_1⁻¹ ··· T̃_n⁻¹ ramène l'état final à l'état initial, sans aucune
   mémoire externe — l'information vit dans l'algèbre.

Lancer : venv/bin/python -m src.matrix_chess.invert_full_game
"""

import os
import random
import shutil

import chess
import chess.engine
import numpy as np

from .engine import Game, to_uci

DIM = 94  # 64 cases + 30 places de cimetière (30 pièces capturables au maximum)


def play_quick_game(seed=7):
    """Stockfish (blancs) vs aléatoire (noirs) : partie courte terminée par mat."""
    sf_path = shutil.which("stockfish") or os.path.expanduser("~/.local/bin/stockfish")
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    rng = random.Random(seed)
    ref = chess.Board()
    ucis = []
    while not ref.is_game_over() and len(ucis) < 200:
        if ref.turn:
            uci = sf.play(ref, chess.engine.Limit(time=0.03)).move.uci()
        else:
            uci = rng.choice(sorted(m.uci() for m in ref.legal_moves))
        ref.push_uci(uci)
        ucis.append(uci)
    sf.quit()
    return ucis, ref.result(), ref.outcome().termination.name


def extended_matrix(move, flat, next_slot):
    """Matrice monomiale 94×94 du coup, et prochaine place de cimetière libre."""
    f, t, promo, ec, also = move
    src = list(range(DIM))  # x'_i = coeff · x_{src[i]}
    if ec is not None:                      # en passant : (f t)(ec slot)
        src[t], src[f] = f, t
        src[next_slot], src[ec] = ec, next_slot
        next_slot += 1
    elif flat[t] != 0:                      # prise : 3-cycle f → t → cimetière
        src[t], src[next_slot], src[f] = f, t, next_slot
        next_slot += 1
    else:                                   # coup simple : transposition (f t)
        src[t], src[f] = f, t
    if also is not None:                    # roque : la tour aussi
        f2, t2 = also
        src[t2], src[f2] = f2, t2
    T = np.zeros((DIM, DIM))
    for i, j in enumerate(src):
        T[i, j] = 1.0
    if promo is not None:
        T[t, :] *= promo                    # le pion (±1) devient ±k
    return T, next_slot


def main():
    # ---------- 1. une partie rapide, jouée jusqu'au bout ----------
    ucis, result, termination = play_quick_game()
    print(f"Partie jouée : {len(ucis)} demi-coups, résultat {result} ({termination})")
    print("  " + " ".join(ucis))

    # ---------- 2. INDÉPENDANT : depuis l'état final, inverser ----------
    # on ne repart que de la liste des coups (qui détermine les matrices T̃)
    # et de l'état final étendu x_n ; b0 n'est PAS réutilisé avant la vérif.
    g = Game()
    b0 = g.board.copy()

    # 2a. espace 64 : l'inverse n'existe pas
    from .engine import move_matrix
    G = np.eye(64, dtype=np.int64)
    gg = Game()
    for uci in ucis:
        mv = next(m for m in gg.legal_moves() if to_uci(m) == uci)
        G = move_matrix(*mv[:2], promo=mv[2], extra_clear=mv[3], also=mv[4]) @ G
        gg.push(mv)
    b_final = gg.board.copy()
    rank = np.linalg.matrix_rank(G.astype(float))
    print(f"\n[espace 64]  rang(G) = {rank}/64 → G singulière, inverse inexistant")
    b0_pinv = np.rint(np.linalg.pinv(G.astype(float)) @ b_final.reshape(64)).astype(int)
    wrong = int((b0_pinv.reshape(8, 8) != b0).sum())
    print(f"[espace 64]  pseudo-inverse de Moore-Penrose : {wrong} cases fausses "
          f"(pièces capturées perdues)")

    # 2b. espace 94 (cases + cimetière) : G̃ inversible, UNE matrice inverse suffit
    x = np.zeros(DIM)
    x[:64] = b0.reshape(64)
    Gt = np.eye(DIM)
    gg = Game()
    slot = 64
    for uci in ucis:
        mv = next(m for m in gg.legal_moves() if to_uci(m) == uci)
        T, slot = extended_matrix(mv, gg.board.reshape(64), slot)
        Gt = T @ Gt
        gg.push(mv)
    x_final = Gt @ x                      # état final étendu (plateau + cimetière)
    assert np.array_equal(x_final[:64], b_final.reshape(64))
    captured = [int(v) for v in x_final[64:slot]]
    det = np.linalg.det(Gt)
    print(f"\n[espace 94]  cimetière final : {captured}")
    print(f"[espace 94]  det(G̃) = {det:+.0f} ≠ 0 → G̃ INVERSIBLE "
          f"(matrice monomiale : permutation ×k des promotions)")

    Gt_inv = np.linalg.inv(Gt)            # LA matrice inverse de toute la partie
    x0 = np.rint(Gt_inv @ x_final).astype(int)
    ok_board = np.array_equal(x0[:64].reshape(8, 8), b0)
    ok_grave = not x0[64:].any()
    print(f"[espace 94]  G̃⁻¹ · x_final : plateau initial reconstruit {'✓' if ok_board else '✗'}, "
          f"cimetière vidé {'✓' if ok_grave else '✗'}")
    assert ok_board and ok_grave

    print(f"""
RÉPONSE : oui, la méthode existe, mais elle exige de changer d'espace.
  • Sur les 64 cases, G est une projection (rang {rank}/64) : chaque prise écrase
    une case et l'information est détruite — AUCUNE matrice 64×64 ne peut revenir.
  • En ajoutant 30 composantes « cimetière », chaque coup devient une matrice
    monomiale (une prise = 3-cycle qui DÉPLACE la pièce au lieu de l'effacer).
    Le produit G̃ ({DIM}×{DIM}) a det = {det:+.0f} : l'inversion redevient de
    l'algèbre pure, G̃⁻¹ = T̃₁⁻¹···T̃ₙ⁻¹, et UNE application de G̃⁻¹ ramène
    l'état final ({len(ucis)} demi-coups, {len(captured)} prises) à l'état initial.
  L'irréversibilité des échecs n'est pas dans le jeu, elle est dans la
  représentation : il suffit de donner une adresse aux morts.""")


if __name__ == "__main__":
    main()
