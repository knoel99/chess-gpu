"""Figures (docs/rapport_gpu/figs/*.pdf) et tableaux (docs/rapport_gpu/tables/*.tex) du rapport GPU,
calculés depuis results/matrix_chess/ : aucune valeur recopiée à la main.
Sources :
summary.csv (matchs), ab_gpu.csv (alpha-bêta contre pleine largeur), perft_gpu.csv (perft).

Lancer depuis la racine : python docs/rapport_gpu/gen_figures.py
"""

import csv
import json
import os
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "matrix_chess")
FIGS = os.path.join(HERE, "figs")
# palette catégorielle de référence (skill dataviz), trois premiers emplacements, ordre fixe
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("v1-torch", "v1 : torch"), ("v2-noyau-temps", "v2 : noyau CUDA"), ("v3-recursif", "v3 : récursive")]
ELOS = ["1320", "1500", "1700", "1900", "2200", "2500", "max"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)


def matches():
    return list(csv.DictReader(open(os.path.join(RES, "summary.csv"))))


def fig_scores(rows):
    pts = defaultdict(float)
    played = set()
    for r in rows:
        pts[(r["tag"], r["elo"])] += float(r["our_score"])
        played.add((r["tag"], r["elo"]))
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    w = 0.26
    for k, (tag, label) in enumerate(SERIES):
        xs = [i + (k - 1) * w for i, e in enumerate(ELOS) if (tag, e) in played]
        ys = [pts[(tag, e)] for e in ELOS if (tag, e) in played]
        ax.bar(xs, [max(y, 0.04) for y in ys], width=w - 0.03, color=C[k], label=label)
    ax.set_xticks(range(len(ELOS)), [f"SF {e}" for e in ELOS])
    ax.set_ylabel("points sur 2 parties")
    ax.set_ylim(0, 2.2)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "scores.pdf")


def fig_speed_depth(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for k, (tag, label) in enumerate(SERIES):
        g = [r for r in rows if r["tag"] == tag]
        axes[0].scatter([k - 0.12] * len(g), [float(r["ours_nps"]) for r in g], s=14, color=C[0],
                        label="notre moteur" if k == 0 else None)
        axes[0].scatter([k + 0.12] * len(g), [float(r["sf_nps"]) for r in g], s=14, color=C[1],
                        label="Stockfish 16, 1 thread" if k == 0 else None)
        axes[1].scatter([k - 0.12] * len(g), [float(r["ours_depth"]) for r in g], s=14, color=C[0])
        axes[1].scatter([k + 0.12] * len(g), [float(r["sf_depth"]) for r in g], s=14, color=C[1])
    for ax in axes:
        ax.set_xticks(range(3), [s[1].split(" : ")[0] for s in SERIES])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("nœuds par seconde (moyenne par partie)")
    axes[1].set_ylabel("profondeur moyenne (demi-coups)")
    axes[0].legend(loc="center right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "vitesse_profondeur.pdf")


def fig_perft():
    rows = list(csv.DictReader(open(os.path.join(RES, "perft_gpu.csv"))))
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    labels = [r["position"] for r in rows]
    y = range(len(rows))
    ax.barh([i + 0.18 for i in y], [float(r["gpu_nps"]) for r in rows], height=0.34, color=C[0],
            label="RTX PRO 6000, noyau fusionné")
    ax.barh([i - 0.18 for i in y], [float(r["sf_nps"]) for r in rows], height=0.34, color=C[1],
            label="Stockfish 16 go perft, 1 thread")
    for i, r in enumerate(rows):
        ax.text(float(r["gpu_nps"]) * 1.15, i + 0.18, f"{float(r['gpu_nps']):.1e}", va="center", fontsize=7, color=INK)
        ax.text(float(r["sf_nps"]) * 1.15, i - 0.18, f"{float(r['sf_nps']):.1e}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(y), labels)
    ax.set_xscale("log")
    ax.set_xlim(1e8, 3e11)
    ax.set_xlabel("nœuds par seconde")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    save(fig, "perft.pdf")


def fig_ab():
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "ab_gpu.csv")))
            if r["kind"] == "profondeur" and r["qdepth"] == "4"]
    pos = list(dict.fromkeys(r["pos"] for r in rows))
    variants = [("pleine", "pleine largeur"), ("ybw", "YBW"), ("hybride", "hybride"), ("1-2-4", "vagues 1-2-4")]
    colors = dict(zip([v for v, _ in variants], [C[0], C[1], C[2], "#6250d6"]))
    for key, ylabel, name in (("nodes", "nœuds générés", "alphabeta_noeuds.pdf"),
                              ("seconds", "temps mur (s)", "alphabeta_temps.pdf")):
        fig, axes = plt.subplots(1, len(pos), figsize=(7.2, 2.4), sharey=True)
        for ax, p in zip(axes, pos):
            for v, label in variants:
                g = sorted((int(r["d_or_budget"]), float(r[key])) for r in rows if r["pos"] == p and r["searcher"] == v)
                g = [(d, y) for d, y in g if y > 0]
                if g:
                    ax.plot(*zip(*g), marker="o", ms=2.5, lw=1.4, color=colors[v], label=label)
            ax.set_yscale("log")
            ax.set_title(p, fontsize=8.5, color=INK)
            ax.set_xlabel("profondeur")
            ax.set_xticks(range(1, 7))
        axes[0].set_ylabel(ylabel)
        fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   fontsize=7.5)
        fig.tight_layout()
        save(fig, name)


def sci(x, digits=2):
    """1.34e9 -> $1{,}34\\cdot 10^{9}$ (typographie française)."""
    m, e = f"{float(x):.{digits}e}".split("e")
    return f"${m.replace('.', '{,}')}\\cdot 10^{{{int(e)}}}$"


def fr(x, fmt):
    return format(float(x), fmt).replace(".", "{,}")


def write_table(name, header, rows, align):
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    body = "\\\\\n".join(" & ".join(r) for r in rows)
    tex = (f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(header)}\\\\\n\\midrule\n"
           f"{body}\\\\\n\\bottomrule\n\\end{{tabular}}\n")
    open(os.path.join(HERE, "tables", name), "w").write(tex)


def tables(rows):
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    pts, n = defaultdict(float), defaultdict(int)
    for r in rows:
        pts[(r["tag"], r["elo"])] += float(r["our_score"])
        n[(r["tag"], r["elo"])] += 1
    body = []
    for t in tags:
        cells = [f"{fr(pts[(t, e)], 'g')}/{n[(t, e)]}" if n[(t, e)] else "--" for e in ELOS]
        tot = sum(pts[(t, e)] for e in ELOS)
        g = [r for r in rows if r["tag"] == t]
        body.append([t.replace("_", "\\_")] + cells + [f"{fr(tot, 'g')}/{sum(n[(t, e)] for e in ELOS)}",
                     sci(median(float(r['ours_nps']) for r in g), 1), fr(median(float(r['ours_depth']) for r in g), '.1f'),
                     f"{median(float(r['sf_depth']) for r in g):.0f}"])
    write_table("scores.tex", ["série"] + ELOS + ["total", "nœuds/s", "prof.", "prof. SF"], body, "l" + "c" * 7 + "rrrr")

    ab = list(csv.DictReader(open(os.path.join(RES, "ab_gpu.csv"))))
    pos = list(dict.fromkeys(r["pos"] for r in ab))
    variants = ["pleine", "ybw", "hybride", "1-2-4"]

    def cell(p, v, kind, d, q="4"):
        m = [r for r in ab if (r["kind"], r["pos"], r["searcher"], r["d_or_budget"], r["qdepth"]) == (kind, p, v, d, q)]
        return m[-1] if m else None

    body = []
    for p in pos:
        for d in ("4", "6"):
            row = [p if d == "4" else "", d]
            for v in variants:
                c = cell(p, v, "profondeur", d)
                row.append(f"{sci(c['nodes'])} en {fr(c['seconds'], '.2f')}~s" if c else "--")
            body.append(row)
    write_table("ab_d6.tex", ["position", "d", "pleine largeur", "YBW", "hybride", "1-2-4"], body, "lccccc")

    body = []
    for p in pos:
        row = [p]
        for v in ["pleine", "ybw", "hybride", "1-2-4"]:
            c = cell(p, v, "temps", "3.0")
            row.append(f"{c['depth']} ({fr(c['seconds'], '.1f')}~s)" if c else "--")
        body.append(row)
    write_table("temps3s.tex", ["position", "pleine largeur", "YBW", "hybride", "1-2-4"], body, "lcccc")


if __name__ == "__main__":
    rows = matches()
    fig_scores(rows)
    fig_speed_depth(rows)
    fig_perft()
    fig_ab()
    tables(rows)
    print("figures et tableaux écrits dans", HERE)
