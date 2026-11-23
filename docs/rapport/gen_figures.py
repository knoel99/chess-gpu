"""Génère figures (docs/rapport/figs/*.pdf) et exemples chiffrés (docs/rapport/exemples.tex)
à partir du vrai moteur : toutes les valeurs affichées sont calculées, jamais recopiées.

Lancer depuis la racine : venv/bin/python -m docs.rapport.gen_figures  (ou python docs/rapport/gen_figures.py)
"""

import os
import sys

import matplotlib
import matplotlib.patheffects  # noqa: F401
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from src.matrix_chess.engine import (Game, move_matrix, piece_moves,  # noqa: E402
                                     reverse_matrix, to_uci)

GLYPH = {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}
NAME = "abcdefgh"


def square_name(idx):
    return f"{NAME[idx % 8]}{8 - idx // 8}"


# (fichier, titre, coups joués avant, coup étudié, commentaire)
EXAMPLES = [
    ("pion", "Pion : \\texttt{e2e4}", [], "e2e4",
     "Le masque contient la poussée simple et la poussée double."),
    ("cavalier", "Cavalier : \\texttt{g1f3}", [], "g1f3",
     "Le cavalier saute : aucun blocage, huit sauts moins ceux hors plateau ou occupés."),
    ("fou", "Fou : \\texttt{c1g5}", ["d2d3", "e7e6"], "c1g5",
     "Le rayon diagonal s'arrête sur le pion d2 (case amie) et se prolonge librement de l'autre côté."),
    ("tour", "Tour : \\texttt{a1a3}", ["a2a4", "a7a5"], "a1a3",
     "Rayon vertical bloqué par le pion a4 ; la tour atteint a2 et a3 seulement."),
    ("dame", "Dame : \\texttt{d1h5}", ["e2e4", "e7e5"], "d1h5",
     "Réunion des rayons orthogonaux et diagonaux : la dame sort par la diagonale libérée."),
    ("roi", "Roi : \\texttt{e1e2}", ["e2e4", "e7e5"], "e1e2",
     "Un pas dans chacune des huit directions, filtré par les cases libres ou adverses."),
    ("prise", "Prise : \\texttt{e4d5}", ["e2e4", "d7d5"], "e4d5",
     "La destination est occupée : $T$ écrase la pièce adverse, que $R$ ne peut plus restituer."),
    ("roque", "Petit roque : \\texttt{e1g1}", ["g1f3", "g8f6", "e2e4", "e7e5", "f1c4", "f8c5"], "e1g1",
     "Deux pièces bougent dans une seule matrice : roi $e1\\to g1$ et tour $h1\\to f1$. Le masque "
     "\\texttt{piece\\_moves} du roi ne contient pas $g1$ : le roque est ajouté par \\texttt{Game}."),
    ("promotion", "Promotion avec prise : \\texttt{a7b8q}", ["a2a4", "b7b5", "a4b5", "a7a6", "b5a6", "g8f6",
                                                  "a6a7", "h7h6"], "a7b8q",
     "Le coefficient $k=5$ dépose directement une dame blanche (pion $=+1$, dame $=5\\cdot(+1)$)."),
    ("enpassant", "Prise en passant : \\texttt{e5d6}", ["e2e4", "a7a6", "e4e5", "d7d5"], "e5d6",
     "La case $d5$ du pion capturé est vidée en plus ($-E_{ee}$) : trois cases changent."),
]


def draw_board(ax, board, title, mask=None, arrow=None):
    for i in range(8):
        for j in range(8):
            ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1,
                                       color="#f0d9b5" if (i + j) % 2 == 0 else "#b58863"))
            v = board[i, j]
            if mask is not None and mask[i, j]:
                ax.add_patch(plt.Circle((j + .5, 7 - i + .5), .18, color="#2a7fff", alpha=.8))
            if v:
                ax.text(j + .5, 7 - i + .48, GLYPH[abs(v)], ha="center", va="center", fontsize=19,
                        color="black" if v < 0 else "white",
                        path_effects=[matplotlib.patheffects.withStroke(
                            linewidth=2.2, foreground="white" if v < 0 else "black")])
    if arrow:
        (fi, fj), (ti, tj) = arrow
        ax.annotate("", xy=(tj + .5, 7 - ti + .5), xytext=(fj + .5, 7 - fi + .5),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
    ax.set_xticks(np.arange(8) + .5); ax.set_xticklabels(list(NAME), fontsize=7)
    ax.set_yticks(np.arange(8) + .5); ax.set_yticklabels([str(8 - i) for i in range(8)][::-1], fontsize=7)
    ax.tick_params(length=0); ax.set_title(title, fontsize=9)


def draw_T(ax, T, special):
    img = np.ones((64, 64, 3))
    for i in range(64):
        for j in range(64):
            if i == j:
                img[i, j] = (.80, .80, .80) if T[i, j] else (.95, .45, .45)
            elif T[i, j]:
                img[i, j] = (.15, .45, 1.0)
    ax.imshow(img, interpolation="nearest")
    for idx in special:
        ax.annotate(square_name(idx), (idx, idx), fontsize=6, color="darkred", ha="center",
                    xytext=(0, -6), textcoords="offset points")
    ax.set_xticks(range(0, 64, 8)); ax.set_yticks(range(0, 64, 8))
    ax.tick_params(labelsize=6)
    ax.set_title("Matrice $T$ ($64\\times64$) : gris = 1, rouge = 0 sur la diagonale, bleu = coefficient\n"
                 "hors diagonale", fontsize=7)


def pm(M, fmt=lambda x: str(int(x))):
    rows = [" & ".join(fmt(x) for x in r) for r in M]
    return "\\begin{pmatrix}" + " \\\\ ".join(rows) + "\\end{pmatrix}"


def fnum(x):
    from fractions import Fraction
    q = Fraction(float(x)).limit_denominator(10)
    return str(q.numerator) if q.denominator == 1 else f"\\tfrac{{{q.numerator}}}{{{q.denominator}}}"


def example(key, title, before, uci, comment):
    g = Game()
    for u in before:
        g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
    move = next(m for m in g.legal_moves() if to_uci(m) == uci)
    f, t, promo, ec, also = move
    b = g.board.copy()
    mover = piece_moves(b, f // 8, f % 8, g.ep_square)
    T = move_matrix(f, t, promo=promo, extra_clear=ec, also=also)
    flat = b.reshape(64)
    after = (T @ flat).reshape(8, 8)
    R = reverse_matrix(T)
    S = sorted({f, t} | ({ec} if ec is not None else set()) | set(also or ()))
    names = [square_name(i) for i in S]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    arrow = ((f // 8, f % 8), (t // 8, t % 8))
    draw_board(axes[0], b, "Avant : $B$ (points = masque)", mask=mover, arrow=arrow)
    draw_board(axes[1], after, "Après : $B'=\\mathrm{mat}(Tb)$")
    draw_T(axes[2], T, S)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs", f"{key}.pdf")); plt.close(fig)

    bS, aS = flat[S], (T @ flat)[S]
    rS = (R @ (T @ flat))[S]
    idx_txt = ", ".join(f"{n}\\,{{=}}\\,{i}" for n, i in zip(names, S))
    lines = [
        "\\Needspace{22\\baselineskip}", f"\\subsection{{{title}}}",
        comment + "\\par",
        f"\\begin{{center}}\\includegraphics[width=\\textwidth]{{figs/{key}.pdf}}\\end{{center}}",
        f"\\textbf{{Masque}} $M$ des destinations de la pièce en ${square_name(f)}$ "
        f"(ligne $i=0$ en haut, valeurs de \\texttt{{piece\\_moves}}) :",
        "\\[ M = {\\scriptsize\\setlength\\arraycolsep{3pt}" + pm(mover.astype(int)) + "} \\]",
        f"\\textbf{{Matrice du coup.}} Indices $\\mathrm{{sq}}(i,j)=8i+j$ : ${idx_txt}$. "
        f"En dehors de ces lignes et colonnes, $T$ coïncide avec l'identité $I_{{64}}$ ; "
        f"restreinte aux cases concernées :",
        "\\[ T_{" + ",".join(names) + "} = " + pm(T[np.ix_(S, S)]) +
        ",\\qquad b_{S}=" + pm(bS.reshape(-1, 1)) + " \\;\\Longrightarrow\\; (Tb)_{S}=" +
        pm(aS.reshape(-1, 1)) + " \\]",
        f"\\textbf{{Retour.}} $R_{{S}} = {pm(R[np.ix_(S, S)], fnum)}$ et "
        f"$(R\\,b')_S = {pm(rS.astype(int).reshape(-1, 1))}$"
        ,
    ]
    if np.any(rS != bS):
        lines[-1] += (" : cette valeur diffère de $b_S$ car la case $t$ contenait une pièce écrasée "
                      "(information perdue, restituée par la pile des prises).")
    else:
        lines[-1] += " : on retrouve exactement $b_S$."
    return "\n\n".join(lines) + "\n"


if __name__ == "__main__":
    out = ["% Fichier généré par gen_figures.py — ne pas éditer à la main.\n"]
    out += [example(*e) for e in EXAMPLES]
    with open(os.path.join(HERE, "exemples.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"{len(EXAMPLES)} exemples générés")
