"""Figures (figs/*.pdf) et tableaux (tables/*.tex) du rapport de l'incrément 2, calculés depuis les
données figées de l'étude (donnees/, analyse/, recherche/) : aucune valeur recopiée à la main.

Lancer depuis la racine : python docs/etudes/2026-11-24_increment2/gen_figures.py
"""

import csv
import math
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C = ["#2a78d6", "#eb6834", "#1baf7a"]  # palette catégorielle de référence, ordre fixe
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def lire(*chemin):
    return list(csv.DictReader(open(os.path.join(HERE, *chemin))))


def fr(x, fmt):
    return format(float(x), fmt).replace(".", "{,}")


def table(nom, entete, lignes, align):
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    corps = "\\\\\n".join(" & ".join(map(str, r)) for r in lignes)
    open(os.path.join(HERE, "tables", nom), "w").write(
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(entete)}\\\\\n\\midrule\n{corps}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n")


def elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / p - 1)


def duel():
    s = lire("donnees", "duel_synthese.csv")
    sc = [float(r["score_b"]) for r in s]
    # courbe de l'écart d'Elo cumulé et de son intervalle à 95 %
    xs, e, lo, hi = [], [], [], []
    for n in range(10, len(sc) + 1):
        m = mean(sc[:n])
        var = sum((x - m) ** 2 for x in sc[:n]) / (n - 1)
        se = math.sqrt(var / n)
        xs.append(n), e.append(elo(m)), lo.append(elo(m - 1.96 * se)), hi.append(elo(m + 1.96 * se))
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.fill_between(xs, lo, hi, color=C[0], alpha=0.18, linewidth=0, label="intervalle à 95 %")
    ax.plot(xs, e, color=C[0], lw=1.6, label="écart d'Elo, nouveau − ancien")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-100, 600)
    ax.set_xlabel("parties jouées")
    ax.set_ylabel("Elo")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.savefig(os.path.join(HERE, "figs", "duel_elo.pdf"))
    plt.close(fig)

    n, tot = len(sc), sum(sc)
    var = sum((x - tot / n) ** 2 for x in sc) / (n - 1)
    se = math.sqrt(var / n)
    w, d = sc.count(1.0), sc.count(0.5)
    lignes = [["parties", n], ["score du nouveau moteur", f"{fr(tot, 'g')}/{n} ({fr(100 * tot / n, '.1f')}~\\%)"],
              ["victoires / nulles / défaites", f"{w} / {d} / {n - w - d}"],
              ["écart d'Elo (IC 95~\\%)", f"${elo(tot / n):+.0f}$ [${elo(tot / n - 1.96 * se):+.0f}$ ; "
                                          f"${elo(tot / n + 1.96 * se):+.0f}$]"],
              ["avec les blancs / les noirs", " / ".join(fr(sum(float(r["score_b"]) for r in s if r["b_blanc"] == c), "g")
                                                         for c in ("True", "False"))]]
    for k, lab in (("prof", "profondeur moyenne (demi-coups)"), ("temps", "temps moyen par coup (s)"),
                   ("noeuds", "nœuds par coup"), ("partiels", "coups au résultat partiel, par partie"),
                   ("pendule_min", "pendule minimale (s)")):
        fmt = ".0f" if k == "noeuds" else ".2f"
        lignes.append([lab, f"{fr(mean(float(r[k + '_b']) for r in s), fmt)} / {fr(mean(float(r[k + '_a']) for r in s), fmt)}"])
    table("duel.tex", ["mesure", "nouveau / ancien"], lignes, "lr")

    a = lire("donnees", "duel_analyse_parties.csv")
    lignes = [[lab, fr(mean(float(r[c + "_precision"]) for r in a), ".1f"), fr(mean(float(r[c + "_perte_cp"]) for r in a), ".1f"),
               fr(mean(int(r[c + "_erreurs"]) for r in a), ".2f"), fr(mean(int(r[c + "_gaffes"]) for r in a), ".2f"),
               sum(r["decisif_joueur"] == c for r in a if r["resultat"] != "1/2-1/2")]
              for c, lab in (("ours", "nouveau (adaptatif, MVV/LVA)"), ("sf", "ancien (hybride, statique)"))]
    table("duel_analyse.tex", ["moteur", "précision (\\%)", "perte (cp)", "erreurs/partie", "gaffes/partie",
                               "coups décisifs"], lignes, "lrrrrr")


def filtres():
    lignes = []
    for f, lab in (("plein", "pleine largeur"), ("hybride-statique", "hybride, tri statique (incr. 1)"),
                   ("adaptatif-mvv", "adaptatif, tri MVV/LVA (incr. 2)")):
        r = lire("donnees", f"filtre_{f}.csv")
        p = sorted(float(x["perte_cp"]) for x in r)
        lignes.append([lab, fr(mean(p), ".1f"), fr(p[len(p) // 2], ".0f"), f"{sum(x >= 100 for x in p)}~\\%",
                       fr(mean(float(x["profondeur"]) for x in r), ".2f"), fr(mean(float(x["temps"]) for x in r), ".2f")])
    table("filtres.tex", ["moteur", "perte moy. (cp)", "médiane", "$\\ge 100$~cp", "profondeur", "temps (s)"],
          lignes, "lrrrrr")


def recherche():
    r = [x for x in lire("recherche", "tri_vagues_gpu.csv") if x["mesure"] == "profondeur" and x["d_ou_budget"] == "6"]
    variantes = ["ybw:statique", "hybride:statique", "ybw:mvv", "adaptatif:mvv", "adaptatif:historique"]
    pos = list(dict.fromkeys(x["pos"] for x in r))
    lignes = []
    for p in pos:
        ligne = [p]
        for v in variantes:
            m = [x for x in r if x["pos"] == p and x["variante"] == v]
            ligne.append(f"{fr(m[0]['secondes'], '.2f')}~s" if m else "--")
        lignes.append(ligne)
    table("recherche_d6.tex", ["position", "YBW stat.", "hybride stat.", "YBW MVV", "adaptatif MVV", "adaptatif hist."],
          lignes, "lrrrrr")


def analyse():
    s = lire("analyse", "series.csv")
    lignes = [[x["serie"].replace("_", "\\_"), x["parties"], fr(x["ours_precision"], ".1f"), fr(x["sf_precision"], ".1f"),
               fr(x["ours_perte_cp"], ".1f"), fr(x["sf_perte_cp"], ".1f"), fr(x["ours_gaffes_par_partie"], ".2f")]
              for x in s]
    table("series.tex", ["série", "parties", "précision nous", "précision SF", "perte nous (cp)", "perte SF (cp)",
                         "gaffes/partie (nous)"], lignes, "lrrrrrr")
    p = lire("analyse", "phases.csv")
    lignes = [[x["coups"], fr(x["ours_perte_cp"], ".1f"), fr(x["ours_gaffes_pct"], ".1f") + "~\\%",
               fr(x["sf_perte_cp"], ".1f"), fr(x["sf_gaffes_pct"], ".1f") + "~\\%"] for x in p]
    table("phases.tex", ["coups", "perte nous (cp)", "gaffes nous", "perte SF (cp)", "gaffes SF"], lignes, "lrrrr")


if __name__ == "__main__":
    duel()
    filtres()
    recherche()
    analyse()
    print("figures et tableaux écrits dans", HERE)
