"""Figures (figs/*.pdf) et fragments LaTeX (gen/*.tex) du rapport de synthèse, calculés depuis le vrai moteur
(src/matrix_chess) et depuis les données figées des études précédentes : aucune valeur recopiée à la main,
sauf les pertes par profondeur de l'agent 3, citées depuis le rapport de l'incrément 1.

Lancer depuis la racine : python docs/etudes/2026-11-25_synthese/gen_figures.py
"""

import csv
import math
import os
import sys
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ETUDES = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import torch  # noqa: E402

from src.matrix_chess import gpu_engine as ge  # noqa: E402
from src.matrix_chess import reduced  # noqa: E402
from src.matrix_chess.engine import (KING, KING_DIRS, KNIGHT_DIRS, ORTHO_DIRS, WHITE, Game,  # noqa: E402
                                     attacked_by, in_check, move_matrix, piece_moves, ray_attacks,
                                     reverse_matrix, shift, step_attacks, to_uci)
from src.matrix_chess.impose import impose  # noqa: E402
from src.matrix_chess.invert_full_game import extended_matrix  # noqa: E402
from src.matrix_chess.retro import is_smothered_mate, predecessors  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POS, NEG = "#d9e7f8", "#fbe2d3"  # cases de valeur positive (blancs) / négative (noirs)
LIGHT, DARK = "#f0d9b5", "#b58863"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "legend.frameon": False, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans"})
GLYPH = {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}
FILES = "abcdefgh"
DEV = "cpu"


def name(idx):
    return f"{FILES[idx % 8]}{8 - idx // 8}"


def idx(s):
    return 8 * (8 - int(s[1])) + FILES.index(s[0])


def play(*ucis, g=None):
    g = g or Game()
    for u in ucis:
        g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
    return g


def find(g, uci):
    return next(m for m in g.legal_moves() if to_uci(m) == uci)


def save(fig, nom):
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs", nom + ".pdf"))
    plt.close(fig)


def write(nom, texte):
    os.makedirs(os.path.join(HERE, "gen"), exist_ok=True)
    open(os.path.join(HERE, "gen", nom + ".tex"), "w").write(texte.strip() + "\n")


def pm(M, fmt=lambda x: str(int(x))):
    return "\\begin{pmatrix}" + " \\\\ ".join(" & ".join(fmt(x) for x in r) for r in M) + "\\end{pmatrix}"


def fr(x, fmt):
    return format(float(x), fmt).replace(".", "{,}")


def lire(*chemin):
    return list(csv.DictReader(open(os.path.join(ETUDES, *chemin))))


# ----------------------------------------------------------------------------------------------- dessins

def board(ax, b, title="", mask=None, arrow=None, labels=None, colors=None, size=17, coords=True):
    """Échiquier : pièces de b, points sur mask, flèche (f, t) en indices, labels/colors par case."""
    for i in range(8):
        for j in range(8):
            ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=LIGHT if (i + j) % 2 == 0 else DARK, lw=0))
            if colors is not None and colors.get(8 * i + j):
                ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=colors[8 * i + j], alpha=.75, lw=0))
            if mask is not None and mask[i, j]:
                ax.add_patch(plt.Circle((j + .5, 7 - i + .5), .17, color=C[0], alpha=.9))
            v = int(b[i, j])
            if v:
                ax.text(j + .5, 7 - i + .47, GLYPH[abs(v)], ha="center", va="center", fontsize=size,
                        color="black" if v < 0 else "white",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white" if v < 0 else "black")])
            if labels is not None and labels.get(8 * i + j) is not None:
                ax.text(j + .95, 7 - i + .06, labels[8 * i + j], ha="right", va="bottom", fontsize=5.5,
                        color=INK, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=.85))
    if arrow:
        f, t = arrow
        ax.annotate("", xy=(t % 8 + .5, 7 - t // 8 + .5), xytext=(f % 8 + .5, 7 - f // 8 + .5),
                    arrowprops=dict(arrowstyle="-|>", color=C[1], lw=2.2, mutation_scale=12))
    ax.set_xlim(0, 8), ax.set_ylim(0, 8), ax.set_aspect("equal")
    if coords:
        ax.set_xticks(np.arange(8) + .5, list(FILES), fontsize=6)
        ax.set_yticks(np.arange(8) + .5, [str(r) for r in range(1, 9)], fontsize=6)
    else:
        ax.set_xticks([]), ax.set_yticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8)


def grid(ax, M, title="", fmt="{:d}", hi=None, size=7, ticks=True, cmap=None):
    """Vue matrice : entier par cellule, ligne i=0 en haut ; fond bleu si > 0, orange si < 0, ou hi."""
    M = np.asarray(M)
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            v = M[i, j]
            fc = "white"
            if hi is not None:
                fc = hi(i, j, v) or "white"
            elif v > 0:
                fc = POS
            elif v < 0:
                fc = NEG
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=GRID, lw=.5))
            if fmt:
                ax.text(j + .5, n - 1 - i + .5, fmt.format(int(v) if float(v).is_integer() else v),
                        ha="center", va="center", fontsize=size, color=INK if v else "#9a9893")
    ax.set_xlim(0, m), ax.set_ylim(0, n), ax.set_aspect("equal")
    if ticks:
        ax.set_xticks(np.arange(m) + .5, range(m), fontsize=5.5)
        ax.set_yticks(np.arange(n) + .5, range(n - 1, -1, -1), fontsize=5.5)
        ax.xaxis.tick_top()
    else:
        ax.set_xticks([]), ax.set_yticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8, pad=12 if ticks else 4)


def ones(color):
    return lambda i, j, v: color if v else None


def big(ax, T, special, title):
    """Image d'une matrice 64×64 : gris = 1 diagonal, rouge = 0 diagonal, bleu = coefficient hors diagonale."""
    n = T.shape[0]
    img = np.ones((n, n, 3))
    d = np.diag(T)
    img[np.arange(n), np.arange(n)] = np.where(d[:, None] != 0, (.80, .80, .80), (.92, .35, .30))
    off = (T != 0) & ~np.eye(n, dtype=bool)
    img[off] = (.16, .47, .84)
    ax.imshow(img, interpolation="nearest")
    for s in special:
        ax.annotate(name(s) if s < 64 else f"c{s - 64}", (s, s), fontsize=5.5, color="#a3261b", ha="left",
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xticks(range(0, n, 8)), ax.set_yticks(range(0, n, 8))
    ax.tick_params(labelsize=5.5, length=2)
    ax.set_title(title, fontsize=8)


# ----------------------------------------------------------------------------------------------- partie I

def plateau():
    g = Game()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    board(ax[0], g.board, "L'échiquier")
    grid(ax[1], g.board, "La matrice $B\\in\\mathbb{Z}^{8\\times 8}$ (ligne $i$, colonne $j$)")
    lab = np.arange(64).reshape(8, 8)
    grid(ax[2], lab, "Indice aplati $8i+j$ de chaque case", hi=lambda i, j, v: "#f3f2ee", size=6.5)
    for i in range(8):
        for j in range(8):
            ax[2].text(j + .5, 7 - i + .16, name(8 * i + j), ha="center", fontsize=4.5, color=MUTED)
    save(fig, "plateau")
    b = g.board.reshape(64)
    write("vecteur", f"b = (b_0,\\dots,b_{{63}}),\\quad b_{{{idx('a8')}}} = {b[idx('a8')]:+d}\\ (\\text{{a8}}),\\quad "
          f"b_{{{idx('e1')}}} = {b[idx('e1')]:+d}\\ (\\text{{e1}}),\\quad b_{{{idx('e2')}}} = {b[idx('e2')]:+d}"
          f"\\ (\\text{{e2}}),\\quad b_{{{idx('e4')}}} = {b[idx('e4')]:d}\\ (\\text{{e4}})")


def S(d):
    """Matrice de décalage 8×8 : (S(d) M)[i] = M[i − d], zéro hors plateau."""
    return np.eye(8, k=-d, dtype=int)


def decalage():
    pos = np.zeros((8, 8), int)
    pos[idx("d4") // 8, idx("d4") % 8] = 1
    for dr, dc in KING_DIRS + KNIGHT_DIRS:
        assert np.array_equal(shift(pos, dr, dc), S(dr) @ pos @ S(dc).T)
    g = Game()
    pawns = (g.board == 1).astype(int)
    fig, ax = plt.subplots(1, 4, figsize=(11, 3.1))
    grid(ax[0], pos, "$P$ : un 1 en d4", hi=ones(C[0] + "55"))
    grid(ax[1], shift(pos, -1, 0), "$\\mathrm{shift}(P,-1,0)$ : un pas vers le nord", hi=ones(C[0] + "55"))
    grid(ax[2], shift(pos, -2, 1), "$\\mathrm{shift}(P,-2,+1)$ : saut de cavalier", hi=ones(C[0] + "55"))
    grid(ax[3], shift(pawns, -1, 0), "$\\mathrm{shift}(\\text{pions blancs},-1,0)$", hi=ones(C[2] + "66"))
    save(fig, "decalage")
    write("shiftS", f"S_{{-1}} = {pm(S(-1))},\\qquad \\mathrm{{shift}}(M, d_r, d_c) = S_{{d_r}}\\, M\\, S_{{d_c}}^{{\\top}}")


def dist_roi(r, c):
    """Distance de Chebyshev par dilatations successives du masque du roi (docs/king_matrix.md)."""
    reach = np.zeros((8, 8), bool)
    reach[r, c] = True
    D = np.full((8, 8), -1)
    D[r, c] = 0
    for k in range(1, 8):
        reach = reach | step_attacks(reach, KING_DIRS)
        D[(D < 0) & reach] = k
    return D


def roi_cavalier():
    e4 = np.zeros((8, 8), bool)
    e4[4, 4] = True
    a1 = np.zeros((8, 8), bool)
    a1[7, 0] = True
    allb = np.ones((8, 8), bool)
    mob_k = sum(shift(allb, dr, dc).astype(int) for dr, dc in KING_DIRS)
    mob_n = sum(shift(allb, dr, dc).astype(int) for dr, dc in KNIGHT_DIRS)
    D = dist_roi(4, 4)
    assert np.array_equal(D, np.maximum(abs(np.arange(8)[:, None] - 4), abs(np.arange(8)[None] - 4)))
    fig, ax = plt.subplots(2, 4, figsize=(11, 6.0))
    z = np.zeros((8, 8), int)
    zk, zn = z.copy(), z.copy()
    zk[4, 4], zn[4, 3] = KING, 2
    board(ax[0, 0], zk, "Roi en e4 : 8 décalages", mask=step_attacks(e4, KING_DIRS))
    za = z.copy()
    za[7, 0] = KING
    board(ax[0, 1], za, "Roi en a1 : 3 cases", mask=step_attacks(a1, KING_DIRS))
    grid(ax[0, 2], mob_k, "Cases atteintes : $\\sum_d \\mathrm{shift}(\\mathbf{1}, d)$",
         hi=lambda i, j, v: {3: "#fbe2d3", 5: "#f3f2ee", 8: "#d9e7f8"}[v], ticks=False)
    grid(ax[0, 3], D, "Distance du roi depuis e4 (dilatations)", hi=lambda i, j, v: f"#2a78d6{max(0, 70 - 14 * v):02x}",
         ticks=False)
    d4 = np.zeros((8, 8), bool)
    d4[4, 3] = True
    board(ax[1, 0], zn, "Cavalier en d4 : 8 sauts", mask=step_attacks(d4, KNIGHT_DIRS))
    zn2 = z.copy()
    zn2[7, 0] = 2
    board(ax[1, 1], zn2, "Cavalier en a1 : 2 sauts", mask=step_attacks(a1, KNIGHT_DIRS))
    grid(ax[1, 2], mob_n, "Sauts possibles par case", ticks=False,
         hi=lambda i, j, v: f"#1baf7a{int(20 + 8 * v):02x}")
    K = np.ones((5, 5), int) * 0
    for dr, dc in KNIGHT_DIRS:
        K[2 + dr, 2 + dc] = 1
    grid(ax[1, 3], K, "Noyau $5\\times5$ du cavalier", hi=ones(C[2] + "66"), ticks=False, size=8)
    fig.tight_layout()
    save(fig, "roi_cavalier")
    Kr = np.ones((3, 3), int)
    Kr[1, 1] = 0
    write("noyau_roi", f"K_{{\\text{{roi}}}} = {pm(Kr)}")


def glisseur():
    b = np.zeros((8, 8), int)
    for s, v in (("d4", 4), ("b4", 1), ("d7", -2), ("g4", -1), ("e1", 6), ("e8", -6)):
        b[idx(s) // 8, idx(s) % 8] = v
    pos = b == 4
    empty = b == 0
    step = {}
    for dr, dc in ORTHO_DIRS:
        front = shift(pos, dr, dc)
        for k in range(1, 8):
            for i, j in zip(*np.nonzero(front)):
                step.setdefault(8 * i + j, k)
            front = shift(front & empty, dr, dc)
    acc = ray_attacks(pos, empty, ORTHO_DIRS)
    final = piece_moves(b, idx("d4") // 8, idx("d4") % 8)
    assert np.array_equal(final, acc & (empty | (b < 0)))
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    board(ax[0], b, "Étape de propagation où chaque case est atteinte",
          labels={s: str(k) for s, k in step.items()},
          colors={s: C[0] for s in step})
    board(ax[1], b, "$A = \\mathrm{ray}(P, \\mathrm{vide})$ : avant filtrage", mask=acc)
    board(ax[2], b, "$A \\wedge (\\mathrm{vide} \\vee \\mathrm{adverse})$ : coups", mask=final)
    save(fig, "glisseur")


def echec():
    g = play("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7")
    b = g.board
    att = attacked_by(b, WHITE)
    king = b == -KING
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
    board(ax[0], b, "Après 4.Dxf7, mat")
    grid(ax[1], att.astype(int), "$\\mathrm{att}_{\\text{blancs}}$ (masque attaqué)", hi=ones(C[1] + "55"))
    grid(ax[2], king.astype(int), "$[B = -6]$ : le roi noir", hi=ones(C[0] + "66"))
    grid(ax[3], (att & king).astype(int), "Produit élément par élément", hi=ones("#d6453d88"))
    save(fig, "echec")
    assert in_check(b, -1) and g.status() == "checkmate"
    kr, kc = np.argwhere(king)[0]
    k = np.zeros((8, 8), bool)
    k[kr, kc] = True
    fuites = step_attacks(k, KING_DIRS) & ~(b < 0) & ~att
    write("echec", f"\\sum_{{i,j}} (\\mathrm{{att}} \\odot [B=-6])_{{ij}} = {int((att & king).sum())},\\qquad "
          f"\\text{{cases de fuite}} = {int(fuites.sum())},\\qquad \\texttt{{status()}} = "
          f"\\text{{« {g.status()} »}}")


# ----------------------------------------------------------------------------------------------- partie III

COUPS = [  # (clé, coups avant, coup étudié)
    ("pion", [], "e2e4"),
    ("prise", ["e2e4", "d7d5"], "e4d5"),
    ("roque", ["g1f3", "g8f6", "e2e4", "e7e5", "f1c4", "f8c5"], "e1g1"),
    ("promotion", ["a2a4", "b7b5", "a4b5", "a7a6", "b5a6", "g8f6", "a6a7", "h7h6"], "a7b8q"),
    ("enpassant", ["e2e4", "a7a6", "e4e5", "d7d5"], "e5d6"),
]


def coup(key, before, uci):
    g = play(*before)
    mv = find(g, uci)
    f, t, promo, ec, also = mv
    b = g.board.copy()
    T = move_matrix(f, t, promo=promo, extra_clear=ec, also=also)
    flat = b.reshape(64)
    after = T @ flat
    h = play(uci, g=play(*before))
    assert np.array_equal(after.reshape(8, 8), h.board)
    R = reverse_matrix(T)
    Sq = sorted({f, t} | ({ec} if ec is not None else set()) | set(also or ()))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    board(ax[0], b, "Avant : $B$", mask=piece_moves(b, f // 8, f % 8, g.ep_square), arrow=(f, t))
    board(ax[1], after.reshape(8, 8), "Après : $B' = \\mathrm{mat}(T\\,b)$")
    big(ax[2], T, Sq, "$T\\in\\mathbb{Z}^{64\\times64}$")
    save(fig, "coup_" + key)
    names = [name(i) for i in Sq]
    rb = (R @ after)[Sq]
    ret = ("$R$ rend $b_S$ sans perte" if np.array_equal(rb, flat[Sq])
           else "$R$ ne rend pas la pièce prise, dont la valeur est perdue")
    fn = (lambda x: str(int(x)) if float(x).is_integer() else f"\\tfrac{{1}}{{{int(round(1 / x))}}}")
    write("coup_" + key,
          f"\\[ S = ({', '.join(f'{n}={i}' for n, i in zip(names, Sq))}),\\quad "
          f"T_S = {pm(T[np.ix_(Sq, Sq)])},\\quad b_S = {pm(flat[Sq].reshape(-1, 1))}"
          f"\\;\\longmapsto\\; (Tb)_S = {pm(after[Sq].reshape(-1, 1))} \\]\n"
          f"\\[ R_S = {pm(R[np.ix_(Sq, Sq)], fn)},\\qquad (R\\,b')_S = {pm(rb.reshape(-1, 1))}"
          f" \\]\n"
          f"Ici, {ret}. La matrice $T$ est de rang {np.linalg.matrix_rank(T.astype(float))} sur 64.")


def composition():
    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6"]
    g = Game()
    src, coef = reduced.identity()
    G = np.eye(64, dtype=np.int64)
    b0 = g.board.reshape(64).copy()
    for u in ucis:
        mv = find(g, u)
        G = move_matrix(mv[0], mv[1], promo=mv[2], extra_clear=mv[3], also=mv[4]) @ G
        src, coef = reduced.compose(src, coef, reduced.pack([mv]))
        g.push(mv)
    assert np.array_equal(reduced.to_dense(src[0], coef[0]), G)
    assert np.array_equal(G @ b0, g.board.reshape(64))
    src, coef = src[0], coef[0]
    lab = {}
    col = {}
    for i in range(64):
        if coef[i] == 0:
            lab[i], col[i] = "0", "#d6453d"
        elif src[i] != i:
            lab[i], col[i] = "←" + name(src[i]), C[0]
    perdues = [name(j) for j in range(64) if not G[:, j].any() and b0[j]]
    rank = np.linalg.matrix_rank(G.astype(float))
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1.05]))
    board(ax[0], g.board, "Après 4…dxc6 : étiquettes $(\\mathrm{src}, \\mathrm{coef})$ de $G$", labels=lab, colors=col)
    big(ax[1], G, [i for i in range(64) if i in lab], "$G = T_8 \\cdots T_1$")
    save(fig, "composition")
    write("composition", f"$G$ est de rang {rank} sur 64. Ses lignes nulles sont les cases vidées "
          f"({', '.join(name(i) for i in range(64) if coef[i] == 0)}). Parmi ses colonnes nulles, {len(perdues)} "
          f"portaient une pièce au départ, {' et '.join(perdues)}, les deux pièces prises.")
    write("partie_ruy", " ".join(ucis))


def retro():
    b = np.zeros((8, 8), int)
    for s, v in (("h1", 6), ("h8", -6), ("e5", 2), ("a7", -1)):
        b[idx(s) // 8, idx(s) % 8] = v
    pre = predecessors(b, WHITE)
    from_d3 = [(p, m) for p, m in pre if m[0] == idx("d3")]
    noms = {0: "rien", -1: "un pion", -2: "un cavalier", -3: "un fou", -4: "une tour", -5: "une dame"}
    fig, ax = plt.subplots(1, len(from_d3) + 1, figsize=(2.0 * (len(from_d3) + 1), 2.5))
    board(ax[0], b, "$b$ (blancs viennent\nde jouer Ce5)", size=11, coords=False,
          colors={idx("e5"): C[1]})
    for a, (p, m) in zip(ax[1:], from_d3):
        c = int(p[idx("e5") // 8, idx("e5") % 8])
        board(a, p, f"$c = {c}$ : Cd3 prend\n{noms[c]}" if c else "$c = 0$ : Cd3-e5\nsans prise", size=11,
              coords=False, colors={idx("d3"): C[0], idx("e5"): "#d6453d" if c else None}, arrow=(m[0], m[1]))
        q = Game(p, turn=WHITE, castling="")
        q.push(find(q, to_uci(m)))
        assert np.array_equal(q.board, b)
    fig.tight_layout()
    save(fig, "retro")
    orig = sorted({name(m[0]) for _, m in pre})
    write("retro", f"le moteur trouve {len(pre)} prédécesseurs légaux à un demi-coup, issus de {len(orig)} cases "
          f"d'origine ({', '.join(orig)}). Parmi eux, {len(from_d3)} partent de d3.")


def cimetiere():
    g = play("e2e4", "d7d5")
    mv = find(g, "e4d5")
    T, slot = extended_matrix(mv, g.board.reshape(64), 64)
    f, t = mv[0], mv[1]
    Sq = [f, t, 64]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    big(ax, T, Sq, "$\\tilde T \\in \\{0,1\\}^{94\\times 94}$ : e4xd5 avec cimetière")
    ax.axhline(63.5, color=C[1], lw=.8), ax.axvline(63.5, color=C[1], lw=.8)
    save(fig, "cimetiere")
    x = np.zeros(94, int)
    x[:64] = g.board.reshape(64)
    y = (T @ x).astype(int)
    det = round(np.linalg.det(T))
    write("cimetiere", f"\\tilde T_S = {pm(T[np.ix_(Sq, Sq)])},\\quad x_S = {pm(x[Sq].reshape(-1, 1))}"
          f"\\;\\longmapsto\\; (\\tilde T x)_S = {pm(y[Sq].reshape(-1, 1))},\\quad \\det \\tilde T = {det:+d},"
          f"\\quad \\tilde T^{{-1}} = \\tilde T^{{\\top}}")
    assert np.array_equal(np.linalg.inv(T), T.T)


def etouffe():
    X = Game(np.array([[0, 0, 0, 0, 0, -4, 0, -6], [0, 0, 0, 0, 0, 0, -1, -1], [0, 0, 0, 0, 5, 0, 0, 2],
                       [0] * 8, [0] * 8, [0] * 8, [0] * 8, [0, 0, 0, 0, 0, 0, 6, 0]]), turn=1, castling="")
    path, final = impose(Game(X.board.copy(), 1, ""), is_smothered_mate, max_plies=3)
    g = Game(X.board.copy(), 1, "")
    etats = [(g.board.copy(), "Départ (trait aux blancs)", None)]
    for u in path:
        mv = find(g, u)
        g.push(mv)
        etats.append((g.board.copy(), u, (mv[0], mv[1])))
    assert is_smothered_mate(g)
    fig, ax = plt.subplots(1, len(etats), figsize=(2.6 * len(etats), 2.9))
    for a, (b, t, arr) in zip(ax, etats):
        board(a, b, t, arrow=arr, size=12)
    ax[-1].set_title(path[-1] + " : mat étouffé", fontsize=8)
    save(fig, "etouffe")
    write("etouffe", ", ".join(f"\\texttt{{{u}}}" for u in path))


# ----------------------------------------------------------------------------------------------- partie IV

def lots():
    g = Game()
    ucis = ["e2e4", "d2d4", "g1f3", "b1c3"]
    m = reduced.pack([find(g, u) for u in ucis])
    par = reduced.extend(g.board)
    kids = reduced.apply_moves(np.repeat(par, len(m), 0), m)
    for u, k in zip(ucis, kids):
        assert np.array_equal(k[:64].reshape(8, 8), play(u).board)
    c0 = 32  # les rangées 8 à 5 ne changent pas : on montre a4..h1 et la sentinelle
    rows = np.vstack([par, kids])[:, c0:]
    fig, ax = plt.subplots(figsize=(8.5, 1.9))
    n, w = rows.shape
    for i in range(n):
        for j in range(w):
            v = rows[i, j]
            changed = i > 0 and v != par[0, c0 + j]
            fc = "#f3f2ee" if j == w - 1 else (POS if v > 0 else NEG if v < 0 else "white")
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=C[1] if changed else GRID,
                                       lw=1.4 if changed else .4, zorder=3 if changed else 1))
            if v:
                ax.text(j + .5, n - 1 - i + .5, str(v), ha="center", va="center", fontsize=6)
    ax.set_xlim(0, w), ax.set_ylim(0, n), ax.set_aspect("equal")
    ax.set_yticks(np.arange(n) + .5, [f"fils {u}" for u in ucis[::-1]] + ["parent"], fontsize=6)
    ax.set_xticks([i + .5 for i in range(0, w)], [name(c0 + i) for i in range(w - 1)] + ["NONE"], fontsize=5.5,
                  rotation=90)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "lots")
    write("lots", "\\texttt{coups} = " + pm(m))


PLANS_NOMS = ([f"{d}{k}" for d in ("NO", "NE", "SO", "SE", "N", "S", "O", "E") for k in range(1, 8)]
              + [f"C{k}" for k in range(1, 9)] + ["P1", "P2", "PxO", "PxE", "O-O", "O-O-O"])


def plans():
    g = play("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6")
    B, cr, ep = ge.from_game(g, DEV)
    mask = ge.pseudo_planes(B, cr, ep, g.turn)[0]
    par, mv, *_ = ge.expand(B, cr, ep, g.turn)
    assert sorted(to_uci(ge.to_move(x)) for x in mv.tolist()) == sorted(to_uci(x) for x in g.legal_moves())
    offs = ge.plane_offsets(WHITE)
    labels = PLANS_NOMS
    choix = [labels.index(x) for x in ("P1", "P2", "C3", "NE2", "N1", "O-O")]
    fig = plt.figure(figsize=(11, 5.3))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
    for k, p in enumerate(choix):
        a = fig.add_subplot(gs[0, k])
        m = mask[p].view(8, 8).numpy()
        dr, dc = offs[p]
        arrows = [(8 * i + j, 8 * (i + dr) + j + dc) for i, j in zip(*np.nonzero(m))]
        board(a, g.board, f"plan {p} « {labels[p]} »\ndécalage $({dr:+d},{dc:+d})$", mask=m, size=10, coords=False,
              arrow=arrows[0] if arrows else None)
    a = fig.add_subplot(gs[1, :])
    cnt = mask.sum(1).numpy()
    cols = [C[0]] * 56 + [C[2]] * 8 + [C[1]] * 4 + [MUTED] * 2
    a.bar(range(70), cnt, color=cols, width=.8)
    a.set_xticks(range(70), labels, rotation=90, fontsize=5)
    a.set_ylabel("cases de départ actives")
    a.spines[["top", "right"]].set_visible(False)
    a.set_xlim(-1, 70)
    for x0, x1, t, c in ((0, 55, "56 plans de rayons : 8 directions × 7 distances", C[0]),
                         (56, 63, "8 sauts", C[2]), (64, 67, "4 pions", C[1]), (68, 69, "2 roques", MUTED)):
        a.text((x0 + x1) / 2, cnt.max() + .6, t, ha="center", fontsize=6.5, color=c)
    a.set_ylim(0, cnt.max() + 1.6)
    fig.tight_layout()
    save(fig, "plans")
    nz = mask.nonzero().tolist()
    ex = [(p, f, f + 8 * offs[p][0] + offs[p][1]) for p, f in nz[:: max(1, len(nz) // 5)][:5]]
    write("plans_decode",
          "\\begin{tabular}{rlrlrl}\\toprule plan $p$ & nom & départ $f$ & case & $t = f + \\delta_p$ & coup\\\\\\midrule\n"
          + "\\\\\n".join(f"{p} & {labels[p]} & {f} & {name(f)} & {t} & \\texttt{{{name(f)}{name(t)}}}" for p, f, t in ex)
          + "\\\\\\bottomrule\\end{tabular}")
    write("plans_total", f"{int(cnt.sum())} coups pseudo-légaux" + (", tous légaux" if len(mv) == int(cnt.sum())
                                                                    else f", dont {len(mv)} légaux"))


def vue_roi():
    geo = ge._consts(DEV)[3]
    k = idx("e4")
    cols, n = {}, {}
    leg = (("pw", "#9a5bd0", "pions"), ("kg", C[1], "voisins"), ("kn", C[2], "sauts"), ("rays", C[0], "rayons"))
    for key, c, _ in leg:
        t = geo[key][1] if key == "pw" else geo[key]
        s = [x for x in t[k].tolist() if x < 64]
        n[key] = (t.shape[1], len(s))
        for x in s:
            cols.setdefault(x, c)
    b = np.zeros((8, 8), int)
    b[4, 4] = KING
    fig, ax = plt.subplots(figsize=(3.4, 3.5))
    board(ax, b, "Cases lues pour tester l'échec du roi e4", colors=cols)
    ax.legend(handles=[matplotlib.patches.Patch(color=c, alpha=.75, label=t) for _, c, t in leg], ncol=4,
              fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.06), handlelength=1)
    save(fig, "vue_roi")
    write("vue_roi", f"Pour le roi en e4, les tables donnent {n['rays'][0]} indices de rayons, dont {n['rays'][1]} "
          f"sur le plateau, {n['kn'][0]} sauts de cavalier, {n['kg'][0]} voisins et {n['pw'][0]} cases de pion, "
          f"soit {n['rays'][0] + n['kn'][0] + n['kg'][0] + n['pw'][0]} lectures par plateau.")


def bitboard():
    g = Game()
    m = (g.board == 1).reshape(64)
    bb = sum(1 << i for i in range(64) if m[i])
    push = (bb >> 8) & ~sum(1 << i for i in range(64) if g.board.reshape(64)[i] != 0)
    assert push == sum(1 << i for i in range(64) if shift(g.board == 1, -1, 0).reshape(64)[i])
    write("bitboard", f"\\texttt{{pions}} = \\texttt{{0x{bb:016X}}},\\qquad "
          f"(\\texttt{{pions}} \\gg 8)\\ \\&\\ \\lnot\\,\\texttt{{occ}} = \\texttt{{0x{push:016X}}}")


def perft():
    """Deux figures de natures différentes : les valeurs de référence recalculées ici sur processeur
    (calcul préparatoire) et le débit mesuré sur GPU à l'incrément 1 (résultat de l'étude)."""
    g = Game()
    B, cr, ep = ge.from_game(g, DEV)
    ref = [20, 400, 8902, 197281, 4865609]  # chessprogramming.org, « Perft Results »
    n = [ge.perft(B, cr, ep, WHITE, d) for d in range(1, 6)]
    assert n == ref
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.bar(range(1, 6), n, color=C[0])
    ax.set_yscale("log")
    for d, v in enumerate(n, 1):
        ax.text(d, v * 1.4, f"{v:,}".replace(",", " "), ha="center", fontsize=6.5)
    ax.set_ylim(10, 5e7)
    ax.set_xlabel("profondeur (demi-coups)"), ax.set_ylabel("positions (échelle log)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "perft_ref")
    write("perft_ref", ", ".join(f"{v:,}".replace(",", "\\,") for v in n))

    p = lire("2026-11-24_increment1-gpu", "donnees", "perft_gpu.csv")
    lab, vals, cs = [], [], []
    for r in p:
        lab += [f"{r['position']}, GPU", f"{r['position']}, Stockfish"]
        vals += [float(r["gpu_nps"]), float(r["sf_nps"])]
        cs += [C[0], C[1]]
    fig, ax = plt.subplots(figsize=(6.5, 2.1))
    ax.barh(range(len(vals)), vals, color=cs)
    ax.set_xscale("log"), ax.invert_yaxis()
    ax.set_yticks(range(len(vals)), lab, fontsize=7)
    for i, v in enumerate(vals):
        ax.text(v * 1.2, i, f"{v:.1e}".replace(".", ","), va="center", fontsize=6.5)
    ax.set_xlabel("positions par seconde, perft en profondeur 6")
    ax.set_xlim(1e7, 1e12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "exp_debit")
    k = next(x for x in p if x["position"] == "kiwipete")
    write("perft", f"{fr(float(k['gpu_nps']) / float(k['sf_nps']), '.0f')}")


# ----------------------------------------------------------------------------------------------- partie V

FEUILLES = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]


def alphabeta(tree, ordre):
    """Alpha-bêta d'un arbre MAX → MIN → feuilles ; renvoie la valeur et les feuilles lues (i, j)."""
    lues, alpha = [], -math.inf
    for i in range(len(tree)):
        beta = math.inf
        for j in ordre(i):
            lues.append((i, j))
            beta = min(beta, tree[i][j])
            if beta <= alpha:
                break
        alpha = max(alpha, beta)
    return alpha, lues


def arbre():
    full = [(i, j) for i in range(3) for j in range(3)]
    v_ab, ab = alphabeta(FEUILLES, lambda i: range(3))
    v_tri, tri = alphabeta(FEUILLES, lambda i: sorted(range(3), key=lambda j: FEUILLES[i][j]))
    v = max(min(r) for r in FEUILLES)
    assert v == v_ab == v_tri
    cas = [("Minimax complet", full, None), ("Alpha-bêta, ordre naturel", ab, None),
           ("Alpha-bêta, fils triés", tri, lambda i: sorted(range(3), key=lambda j: FEUILLES[i][j]))]
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9))
    for a, (t, lues, ordre) in zip(ax, cas):
        a.axis("off")
        a.set_xlim(-.6, 8.6), a.set_ylim(-.5, 2.5)
        a.set_title(f"{t} : {len(lues)} feuilles lues", fontsize=8)
        a.plot(4, 2.1, "o", ms=14, color=C[0])
        a.text(4, 2.1, str(v), ha="center", va="center", color="white", fontsize=8, weight="bold")
        a.text(4.45, 2.3, "MAX", fontsize=6, color=MUTED)
        for i in range(3):
            xi = 1 + 3 * i
            seen = [j for (ii, j) in lues if ii == i]
            val = min(FEUILLES[i][j] for j in seen)
            exact = len(seen) == 3
            a.plot([4, xi], [1.85, 1.3], color=INK, lw=.8)
            a.add_patch(plt.Rectangle((xi - .3, .95), .6, .4, fc=C[1] if exact else "white", ec=C[1]))
            a.text(xi, 1.15, str(val) if exact else f"≤{val}", ha="center", va="center", fontsize=7.5,
                   color="white" if exact else INK)
            pos = list(ordre(i)) if ordre else range(3)
            for r, j in enumerate(pos):
                xj = xi - .9 + .9 * r
                lu = (i, j) in lues
                a.plot([xi, xj], [.95, .35], color=INK if lu else "#bdbbb5", lw=.8, ls="-" if lu else "--")
                a.add_patch(plt.Rectangle((xj - .22, .0), .44, .35, fc="#f3f2ee" if lu else "white",
                                          ec=INK if lu else "#bdbbb5", lw=.6))
                a.text(xj, .17, str(FEUILLES[i][j]), ha="center", va="center", fontsize=7,
                       color=INK if lu else "#bdbbb5")
        a.text(-.5, 1.15, "MIN", fontsize=6, color=MUTED)
    save(fig, "arbre")
    ell = [x for r in FEUILLES for x in r]
    A = [["0" if k // 3 == i else "-\\infty" for k in range(9)] for i in range(3)]
    v1 = [max(-x for x in r) for r in FEUILLES]
    write("maxplus_A", f"A_1 = {pm(A, str)}")
    write("maxplus", f"(-\\ell)^{{\\top}} = ({', '.join(str(-x) for x in ell)}) \\\\ "
          f"v_1 = A_1 \\otimes (-\\ell) = ({', '.join(map(str, v1))})^{{\\top}},\\quad "
          f"v_0 = A_0 \\otimes (-v_1) = {max(-x for x in v1)}")


def intervalles():
    """[L, U] des trois fils de la racine quand les feuilles sont révélées dans l'ordre de l'alpha-bêta."""
    lo, hi = 0, 15
    _, nat = alphabeta(FEUILLES, lambda i: range(3))
    _, tri = alphabeta(FEUILLES, lambda i: sorted(range(3), key=lambda j: FEUILLES[i][j]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.7), sharey=True)
    decs = []
    for ax, (titre, ordre) in zip(axes, (("ordre naturel", nat), ("fils triés", tri))):
        n = len(ordre) + 1
        L, U = np.zeros((n, 3)), np.zeros((n, 3))
        for s in range(n):
            vu = set(ordre[:s])
            for i in range(3):
                L[s, i] = min(FEUILLES[i][j] if (i, j) in vu else lo for j in range(3))
                U[s, i] = min(FEUILLES[i][j] if (i, j) in vu else hi for j in range(3))
        dec = next(s for s in range(n) if L[s, 0] >= max(U[s, 1], U[s, 2]))
        decs.append(dec)
        for i in range(3):
            x = np.arange(n) + (i - 1) * .2
            ax.vlines(x, L[:, i], U[:, i], color=C[i], lw=4, alpha=.85,
                      label=f"fils {i + 1} : $[L_{i + 1}, U_{i + 1}]$")
            ax.plot(x, L[:, i], "_", color=C[i], ms=7, mew=2), ax.plot(x, U[:, i], "_", color=C[i], ms=7, mew=2)
        ax.axvline(dec, color=MUTED, lw=.8, ls="--")
        ax.text(dec - .15, 15.6, f"décidé après {dec} feuilles", fontsize=6.5, color=MUTED, ha="right")
        ax.set_xticks(range(n), ["0"] + [f"{FEUILLES[i][j]}" for i, j in ordre], fontsize=6.5)
        ax.set_xlabel(f"feuilles révélées ({titre}) : valeur de la feuille")
        ax.set_ylim(-.8, 17), ax.spines[["top", "right"]].set_visible(False), ax.grid(axis="y", color=GRID, lw=.6)
    axes[0].set_ylabel("valeur")
    axes[1].legend(fontsize=6.5, loc="center right")
    save(fig, "intervalles")
    write("decision", f"{decs[0]} feuilles dans l'ordre naturel, {decs[1]} avec les fils triés")


def vagues():
    rangs = 16
    schemas = [("ybw (0, 1)", (0, 1)), ("hybride (0, 1, 4)", (0, 1, 4)), ("1-2-4 (0, 1, 3, 7, 15)", (0, 1, 3, 7, 15, 31))]
    schemas = [("adaptatif, petit lot (0,)", (0,))] + schemas
    fig, ax = plt.subplots(figsize=(8, 2.2))
    pal = [C[0], C[1], C[2], "#9a5bd0", MUTED, INK]
    for r, (nom, w) in enumerate(schemas):
        bornes = list(w) + [1 << 30]
        for k in range(rangs):
            v = max(i for i, b in enumerate(bornes) if k >= b)
            ax.add_patch(plt.Rectangle((k, len(schemas) - 1 - r), .92, .8, fc=pal[v], alpha=.85))
            ax.text(k + .46, len(schemas) - 1 - r + .4, str(v + 1), ha="center", va="center", fontsize=6,
                    color="white")
    ax.set_xlim(0, rangs), ax.set_ylim(0, len(schemas))
    ax.set_yticks(np.arange(len(schemas)) + .4, [n for n, _ in schemas][::-1], fontsize=7)
    ax.set_xticks(np.arange(rangs) + .46, range(rangs), fontsize=6)
    ax.set_xlabel("rang du fils après tri (0 = meilleur candidat) ; chiffre = numéro de vague")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "vagues")


# ----------------------------------------------------------------------------------------------- partie VI

def experiences():
    ab = [r for r in lire("2026-11-24_increment1-gpu", "donnees", "ab_gpu.csv")
          if r["kind"] == "profondeur" and r["pos"] == "kiwipete" and r["qdepth"] == "4"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 2.7))
    for s, c, lab in (("pleine", C[1], "pleine largeur"), ("ybw", C[0], "alpha-bêta YBW"),
                      ("hybride", INK, "alpha-bêta hybride"), ("1-2-4", C[2], "alpha-bêta 1-2-4")):
        rr = [r for r in ab if r["searcher"] == s]
        d = [int(r["d_or_budget"]) for r in rr]
        ax[0].plot(d, [int(r["nodes"]) for r in rr], "o-", color=c, ms=3, label=lab)
        ax[1].plot(d, [max(float(r["seconds"]), 1e-3) for r in rr], "o-", color=c, ms=3, label=lab)
    ax[0].set_yscale("log"), ax[1].set_yscale("log")
    ax[0].set_ylabel("nœuds"), ax[1].set_ylabel("secondes sur GPU")
    for a in ax:
        a.set_xlabel("profondeur"), a.spines[["top", "right"]].set_visible(False), a.grid(color=GRID, lw=.6)
    ax[0].legend(fontsize=6.5)
    ax[0].set_title("Kiwipete : nœuds visités", fontsize=8)
    ax[1].set_title("Kiwipete : temps de calcul sur GPU", fontsize=8)
    save(fig, "exp_noeuds")

    s = lire("2026-11-24_increment1-gpu", "donnees", "summary.csv")
    tags = list(dict.fromkeys(r["tag"] for r in s))
    noms = {"v1-torch": "v1 torch (3+0)", "v2-noyau-temps": "v2 noyau CUDA (3+0)", "v3-recursif": "v3 récursif (3+0)",
            "v4-plein-ucielo": "v4 pleine largeur (calibration)", "v4-hybride-ucielo": "v4 hybride, bug racine",
            "v4b-hybride-ucielo-corrige": "v4b hybride corrigé (calibration)",
            "v4-hybride-machine": "v4 contre SF 30 threads"}
    fig, ax = plt.subplots(figsize=(7, 2.5))
    for k, t in enumerate(tags):
        rr = [r for r in s if r["tag"] == t]
        sc = sum(float(r["our_score"]) for r in rr)
        ax.barh(k, 100 * sc / len(rr), color=C[2] if t.startswith("v4b") else C[0])
        ax.text(100 * sc / len(rr) + 1, k, f"{sc:g}/{len(rr)}".replace(".", ","), va="center", fontsize=6.5)
    ax.set_yticks(range(len(tags)), [noms.get(t, t) for t in tags], fontsize=6.5)
    ax.invert_yaxis(), ax.set_xlim(0, 75), ax.set_xlabel("score contre Stockfish bridé 1320 → max (%)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "exp_versions")

    # pertes Stockfish du coup choisi par profondeur : agent 3, rapport de l'incrément 1 (agent3.tex)
    prof, perte = [4, 5, 6, 7, 8], [71, 57, 53, 23, 20]
    ph = lire("2026-11-24_increment2", "analyse", "phases.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 2.6))
    ax[0].plot(prof, perte, "o-", color=C[0], ms=4)
    for x, y in zip(prof, perte):
        ax[0].text(x, y + 3, f"{y} cp", ha="center", fontsize=6.5)
    ax[0].set_xlabel("profondeur de recherche"), ax[0].set_ylabel("perte moyenne du coup (cp)")
    ax[0].set_xticks(prof), ax[0].set_ylim(0, 85), ax[0].set_title("La profondeur est le facteur limitant", fontsize=8)
    x = np.arange(len(ph))
    ax[1].bar(x - .2, [float(r["ours_perte_cp"]) for r in ph], .4, color=C[0], label="notre moteur")
    ax[1].bar(x + .2, [float(r["sf_perte_cp"]) for r in ph], .4, color=C[1], label="Stockfish bridé")
    ax[1].set_xticks(x, [r["coups"] for r in ph])
    ax[1].set_xlabel("numéro du coup"), ax[1].set_ylabel("perte moyenne (cp)")
    ax[1].set_title("Où la partie se perd : la sortie d'ouverture", fontsize=8), ax[1].legend(fontsize=6.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False), a.grid(axis="y", color=GRID, lw=.6)
    save(fig, "exp_qualite")

    d = lire("2026-11-24_increment2", "donnees", "duel_synthese.csv")
    sc = [float(r["score_b"]) for r in d]

    def elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / p - 1)
    xs, e, lo, hi = [], [], [], []
    for n in range(10, len(sc) + 1):
        m = mean(sc[:n])
        se = math.sqrt(sum((v - m) ** 2 for v in sc[:n]) / (n - 1) / n)
        xs.append(n), e.append(elo(m)), lo.append(elo(m - 1.96 * se)), hi.append(elo(m + 1.96 * se))
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    ax.fill_between(xs, lo, hi, color=C[0], alpha=.18, lw=0, label="intervalle à 95 %")
    ax.plot(xs, e, color=C[0], lw=1.6, label="écart d'Elo, incrément 2 − incrément 1")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_ylim(-100, 600), ax.set_xlabel("parties jouées (10 s + 0,1 s)"), ax.set_ylabel("Elo")
    ax.legend(fontsize=6.5), ax.spines[["top", "right"]].set_visible(False), ax.grid(color=GRID, lw=.6)
    save(fig, "exp_duel")
    n, tot = len(sc), sum(sc)
    se = math.sqrt(sum((v - tot / n) ** 2 for v in sc) / (n - 1) / n)
    write("duel", f"{fr(tot, 'g')}/{n} ({fr(100 * tot / n, '.1f')}~\\%), soit ${elo(tot / n):+.0f}$ Elo "
          f"[${elo(tot / n - 1.96 * se):+.0f}$ ; ${elo(tot / n + 1.96 * se):+.0f}$]")


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    for f in (plateau, decalage, roi_cavalier, glisseur, echec, composition, retro, cimetiere, etouffe, lots, plans,
              vue_roi, bitboard, perft, arbre, intervalles, vagues, experiences):
        f()
        print("ok", f.__name__)
    for c in COUPS:
        coup(*c)
    print("figures dans figs/, fragments dans gen/")
