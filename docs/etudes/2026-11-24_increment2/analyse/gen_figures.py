"""Figures (figs/*.pdf) et tableaux (series.csv, phases.csv) de l'analyse coup par coup des 62
parties de l'incrément 1, calculés depuis coups.csv et parties.csv (écrits par
`python -m src.matrix_chess.analyse parties`) : aucune valeur recopiée à la main.

Lancer depuis la racine : python docs/etudes/2026-11-24_increment2/analyse/gen_figures.py
"""

import csv
import os
from collections import defaultdict
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
# même palette que l'incrément 1 : notre moteur en bleu, Stockfish en orange
OURS, SF, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("v1-torch", "v1 torch"), ("v2-noyau-temps", "v2 noyau"), ("v4-plein-ucielo", "v4 plein"),
          ("v4-hybride-ucielo", "v4 hybride\n(avant correctif)"), ("v4b-hybride-ucielo-corrige", "v4b hybride"),
          ("v4-hybride-machine", "v4 hybride\nSF machine")]
ELOS = ["1320", "1500", "1700", "1900", "2200", "2500", "max"]
PHASES = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 999)]
NOMS = {"ours": "notre moteur", "sf": "Stockfish"}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def lire(nom):
    with open(os.path.join(HERE, nom)) as f:
        return list(csv.DictReader(f))


def ecrire(nom, rows):
    with open(os.path.join(HERE, nom), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, nom):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, nom))
    plt.close(fig)


def tableau_series(coups, parties):
    rows = []
    for tag, _ in SERIES:
        n = sum(p["tag"] == tag for p in parties)
        row = dict(serie=tag, parties=n)
        for j in ("ours", "sf"):
            c = [x for x in coups if x["tag"] == tag and x["joueur"] == j]
            row[f"{j}_precision"] = round(mean(float(x["precision"]) for x in c), 1)
            row[f"{j}_perte_cp"] = round(mean(float(x["perte_cp"]) for x in c), 1)
            row[f"{j}_perte_gain"] = round(mean(float(x["perte_gain"]) for x in c), 2)
            for k in ("imprecision", "erreur", "gaffe"):
                row[f"{j}_{k}s_par_partie"] = round(sum(x["classe"] == k for x in c) / n, 2)
        rows.append(row)
    return rows


def tableau_phases(coups):
    rows = []
    for lo, hi in PHASES:
        row = dict(coups=f"{lo}-{hi}" if hi < 999 else f"{lo}+")
        for j in ("ours", "sf"):
            c = [x for x in coups if x["joueur"] == j and lo <= (int(x["demi_coup"]) + 1) // 2 <= hi]
            row[f"{j}_n"] = len(c)
            row[f"{j}_perte_cp"] = round(mean(float(x["perte_cp"]) for x in c), 1)
            row[f"{j}_perte_gain"] = round(mean(float(x["perte_gain"]) for x in c), 2)
            row[f"{j}_gaffes_pct"] = round(100 * sum(x["classe"] == "gaffe" for x in c) / len(c), 1)
        rows.append(row)
    return rows


def fig_courbes(coups, parties, tag, titre):
    """Petits multiples : pourcentage de gain de notre moteur après chaque demi-coup, une partie par case."""
    jeu = sorted((p for p in parties if p["tag"] == tag), key=lambda p: (ELOS.index(p["elo"]), p["ours"] != "white"))
    par = defaultdict(list)
    for x in coups:
        par[x["partie"]].append(x)
    ncol = 4
    nrow = -(-len(jeu) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 1.9 * nrow + 0.6), sharey=True, squeeze=False)
    for ax in axes.flat[len(jeu):]:
        ax.set_visible(False)
    for ax, p in zip(axes.flat, jeu):
        c = par[p["partie"]]
        x = [int(r["demi_coup"]) / 2 for r in c]
        w = [float(r["gain_apres"]) if r["camp"] == p["ours"] else 100 - float(r["gain_apres"]) for r in c]
        ax.axhline(50, color=MUTED, lw=0.6)
        ax.fill_between(x, 50, w, color=OURS, alpha=0.15, lw=0)
        ax.plot(x, w, color=OURS, lw=1.1)
        for j, col in (("ours", OURS), ("sf", SF)):
            g = [i for i, r in enumerate(c) if r["joueur"] == j and r["classe"] == "gaffe"]
            ax.plot([x[i] for i in g], [w[i] for i in g], "o", ms=3.2, color=col, mec="white", mew=0.4)
        score = {"1-0": "1" if p["ours"] == "white" else "0", "0-1": "0" if p["ours"] == "white" else "1"}
        ax.set_title(f"SF {p['elo']} · {'blancs' if p['ours'] == 'white' else 'noirs'} · "
                     f"{score.get(p['resultat'], '½')}", fontsize=8, loc="left", color=INK)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
    for ax in axes[:, 0]:
        ax.set_ylabel("gain, notre moteur (%)")
    fig.suptitle(f"{titre} : pourcentage de gain de notre moteur selon Stockfish, coup par coup "
                 "(points : gaffes, bleu = les nôtres, orange = celles de Stockfish ; titre : niveau, couleur, score)",
                 fontsize=9, x=0.01, ha="left")
    fig.supxlabel("coup", fontsize=8)
    fig.tight_layout()
    save(fig, f"courbes_{tag}.pdf")


def fig_precision(parties):
    """Précision moyenne par niveau Elo, un panneau par série."""
    fig, axes = plt.subplots(1, len(SERIES), figsize=(12, 2.6), sharey=True)
    for ax, (tag, titre) in zip(axes, SERIES):
        for j, col in (("ours", OURS), ("sf", SF)):
            pts = [(i, mean(float(p[f"{j}_precision"]) for p in parties if p["tag"] == tag and p["elo"] == e))
                   for i, e in enumerate(ELOS) if any(p["tag"] == tag and p["elo"] == e for p in parties)]
            ax.plot(*zip(*pts), "o-", color=col, ms=3.5, lw=1.2, label=NOMS[j])
        ax.set_title(titre.replace("\n", " "), fontsize=8, loc="left", color=INK)
        ax.set_xticks(range(len(ELOS)), ELOS, rotation=60)
        ax.set_xlim(-0.5, len(ELOS) - 0.5)
    axes[0].set_ylabel("précision moyenne (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.supxlabel("niveau de Stockfish (UCI_Elo)", fontsize=8)
    fig.tight_layout()
    save(fig, "precision.pdf")


def barres(ax, labels, ours, sf, ylabel):
    for k, (vals, col, j) in enumerate(((ours, OURS, "ours"), (sf, SF, "sf"))):
        xs = [i + (k - 0.5) * 0.38 for i in range(len(labels))]
        ax.bar(xs, vals, 0.36, color=col, label=NOMS[j])
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig_perte(series, phases):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [6, 5]})
    barres(a, [t for _, t in SERIES], [r["ours_perte_cp"] for r in series], [r["sf_perte_cp"] for r in series],
           "perte moyenne par coup (cp)")
    a.set_title("Par série", fontsize=9, loc="left", color=INK)
    a.legend(fontsize=7)
    barres(b, [r["coups"] for r in phases], [r["ours_perte_cp"] for r in phases], [r["sf_perte_cp"] for r in phases],
           "perte moyenne par coup (cp)")
    b.set_title("Par moment de la partie (numéro du coup, toutes séries)", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    save(fig, "perte.pdf")


def main():
    coups, parties = lire("coups.csv"), lire("parties.csv")
    series, phases = tableau_series(coups, parties), tableau_phases(coups)
    ecrire("series.csv", series)
    ecrire("phases.csv", phases)
    for tag, titre in SERIES:
        fig_courbes(coups, parties, tag, titre.replace("\n", " "))
    fig_precision(parties)
    fig_perte(series, phases)
    for r in series + phases:
        print(r)


if __name__ == "__main__":
    main()
