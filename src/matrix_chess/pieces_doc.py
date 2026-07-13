"""Page LaTeX des opérations matricielles de chaque pièce.

Les matrices d'exemple ne sont pas écrites à la main : elles sont calculées
par le moteur (piece_moves) au chargement — la page est donc une preuve
d'exécution autant qu'une documentation.
"""

import numpy as np

from .engine import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, piece_moves

SYM = {0: "\\cdot", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
       -1: "\\bar{P}", -2: "\\bar{N}", -3: "\\bar{B}", -4: "\\bar{R}",
       -5: "\\bar{Q}", -6: "\\bar{K}"}


def lat(M, sym=False):
    """Matrice 8×8 → pmatrix LaTeX (sym=True : symboles de pièces)."""
    rows = [" & ".join(SYM[int(v)] if sym else str(int(v)) for v in row) for row in np.asarray(M, dtype=int)]
    return "\\begin{pmatrix}" + " \\\\ ".join(rows) + "\\end{pmatrix}"


def mask_of(pieces, r, c, ep=None):
    b = np.zeros((8, 8), dtype=np.int64)
    for rr, cc, v in pieces:
        b[rr, cc] = v
    return b, piece_moves(b, r, c, ep).astype(int)


def build():
    # exemples calculés par le moteur
    b_king, m_king = mask_of([(4, 3, KING)], 4, 3)
    _, m_king_corner = mask_of([(0, 0, KING)], 0, 0)
    b_knight, m_knight = mask_of([(4, 3, KNIGHT)], 4, 3)
    b_rook, m_rook = mask_of([(4, 3, ROOK), (2, 3, PAWN), (4, 5, -PAWN)], 4, 3)
    b_bishop, m_bishop = mask_of([(4, 3, BISHOP), (2, 1, -PAWN), (6, 5, PAWN)], 4, 3)
    b_queen, m_queen = mask_of([(4, 3, QUEEN)], 4, 3)
    b_pawn, m_pawn = mask_of([(6, 4, PAWN)], 6, 4)
    b_pawnc, m_pawnc = mask_of([(4, 4, PAWN), (3, 3, -PAWN), (3, 4, -KNIGHT), (3, 5, -PAWN)], 4, 4)
    b_ep, m_ep = mask_of([(3, 3, PAWN), (3, 4, -PAWN)], 3, 3, ep=(2, 4))

    S = []  # sections (titre, html)

    def sec(title, body):
        S.append((title, body))

    sec("Conventions communes", rf"""
<p>Plateau $B \in \mathbb{{Z}}^{{8\times 8}}$ (+1..+6 blancs P N B R Q K, négatif noirs, 0 vide ;
ligne $i=0$ = rangée 8). Position de la pièce : matrice one-hot
$P_{{i,j}} = \mathbb{{1}}[(i,j)=(r,c)]$. Masques d'occupation :</p>
$$E = \mathbb{{1}}[B = 0] \quad\text{{(vide)}} \qquad
A = \mathbb{{1}}[\sigma B < 0] \quad\text{{(ennemi, }}\sigma=\pm 1\text{{ la couleur)}}$$
<p>Brique de base — le <b>shift</b> à remplissage nul (les bords clippent seuls) :</p>
$$\mathrm{{shift}}(P, d_r, d_c)_{{i,j}} = \begin{{cases}} P_{{i-d_r,\,j-d_c}} & 0 \le i-d_r < 8,\; 0 \le j-d_c < 8 \\ 0 & \text{{sinon}} \end{{cases}}$$
<p>Et le masque final de coups de toute pièce retire les cases amies :
$M = M_{{\text{{brut}}}} \wedge (E \vee A)$. Chaque coup retenu $f \to t$ devient l'opérateur
linéaire 64×64 appliqué au plateau aplati $b \in \mathbb{{Z}}^{{64}}$ :</p>
$$T = I - E_{{ff}} - E_{{tt}} + E_{{tf}}, \qquad b' = T\,b$$
""")

    sec("Roi ♔", rf"""
<p>Somme de 8 shifts unitaires — équivalente à la convolution par le noyau $3\times 3$,
ou à la distance de Tchebychev $=1$ (cf. <code>king_matrix.md</code>) :</p>
$$M_{{\text{{brut}}}} = \sum_{{(d_r,d_c) \in \mathcal{{D}}_8}} \mathrm{{shift}}(P, d_r, d_c)
= P * K,\qquad K = \begin{{pmatrix}} 1&1&1 \\ 1&0&1 \\ 1&1&1 \end{{pmatrix}}$$
$$\mathcal{{D}}_8 = \{{(\pm 1, \pm 1), (\pm 1, 0), (0, \pm 1)\}}$$
<p>Exemple calculé, roi en d4 puis en a8 (le bord clippe sans condition) :</p>
$$B = {lat(b_king, sym=True)} \;\Rightarrow\; M = {lat(m_king)}
\qquad M_{{a8}} = {lat(m_king_corner)}$$
""")

    sec("Cavalier ♘", rf"""
<p>Même principe avec les 8 sauts $(\pm 1, \pm 2)$ — le cavalier <b>saute par
construction</b> : aucun terme de la somme ne dépend de l'occupation :</p>
$$M_{{\text{{brut}}}} = \sum_{{(d_r,d_c) \in \mathcal{{D}}_N}} \mathrm{{shift}}(P, d_r, d_c) = P * K_N,
\qquad K_N = \begin{{pmatrix}} 0&1&0&1&0 \\ 1&0&0&0&1 \\ 0&0&0&0&0 \\ 1&0&0&0&1 \\ 0&1&0&1&0 \end{{pmatrix}}$$
<p>Exemple calculé, cavalier en d4 :</p>
$$B = {lat(b_knight, sym=True)} \;\Rightarrow\; M = {lat(m_knight)}$$
""")

    prop = r"""$$F^{(1)} = \mathrm{shift}(P, d), \qquad
F^{(k+1)} = \mathrm{shift}\big(F^{(k)} \wedge E,\; d\big), \qquad
A_d = \bigvee_{k=1}^{7} F^{(k)}$$
<p>Le front $F$ ne se propage qu'à travers les cases vides ($\wedge\, E$) mais le rayon
<b>inclut la première case occupée</b> rencontrée — c'est la cible de prise ;
le masquage final $\wedge (E \vee A)$ élimine le cas où elle est amie.</p>"""

    sec("Tour ♖", rf"""
<p>Pièce glissante : <b>propagation avec blocage</b>, 7 itérations de shift par direction
$d \in \mathcal{{D}}_+ = \{{(\pm 1, 0), (0, \pm 1)\}}$ :</p>
{prop}
$$M_{{\text{{brut}}}} = \bigvee_{{d \in \mathcal{{D}}_+}} A_d$$
<p>Exemple calculé : tour d4, pion ami d6 (bloque, case exclue), pion ennemi f4 (bloque, case prenable) :</p>
$$B = {lat(b_rook, sym=True)} \;\Rightarrow\; M = {lat(m_rook)}$$
""")

    sec("Fou ♗", rf"""
<p>Même propagation sur les diagonales $\mathcal{{D}}_\times = \{{(\pm 1, \pm 1)\}}$ :</p>
$$M_{{\text{{brut}}}} = \bigvee_{{d \in \mathcal{{D}}_\times}} A_d$$
<p>Exemple calculé : fou d4, pion ennemi b6 (prenable, rayon coupé), pion ami f2 (exclu) :</p>
$$B = {lat(b_bishop, sym=True)} \;\Rightarrow\; M = {lat(m_bishop)}$$
""")

    sec("Dame ♕", rf"""
<p>Union tour + fou : propagation sur les 8 directions $\mathcal{{D}}_8 = \mathcal{{D}}_+ \cup \mathcal{{D}}_\times$ :</p>
$$M_{{\text{{brut}}}} = \bigvee_{{d \in \mathcal{{D}}_8}} A_d$$
<p>Exemple calculé, dame en d4 sur plateau vide (14 + 13 = 27 cases) :</p>
$$B = {lat(b_queen, sym=True)} \;\Rightarrow\; M = {lat(m_queen)}$$
""")

    sec("Pion ♙", rf"""
<p>La seule pièce asymétrique : la direction dépend de la couleur, $\phi = -\sigma$
(les blancs montent vers $i$ décroissant), et <b>l'avance et la prise sont des
opérations différentes</b> :</p>
$$M_1 = \mathrm{{shift}}(P, \phi, 0) \wedge E \qquad\text{{(avance simple : case vide requise)}}$$
$$M_2 = \mathrm{{shift}}(M_1, \phi, 0) \wedge E \wedge \mathbb{{1}}[r = r_0]
\qquad\text{{(double pas depuis la rangée initiale, les deux cases vides)}}$$
$$M_\times = \big(\mathrm{{shift}}(P, \phi, -1) \vee \mathrm{{shift}}(P, \phi, +1)\big)
\wedge (A \vee P_{{ep}}) \qquad\text{{(prises diagonales + en passant)}}$$
$$M = M_1 \vee M_2 \vee M_\times$$
<p>Exemples calculés — avance e2 ; prises e4 vs d5/f5 (e5 occupée : pas d'avance) ;
en passant d5×e6 ($P_{{ep}}$ = case e6) :</p>
$$B = {lat(b_pawn, sym=True)} \Rightarrow M = {lat(m_pawn)}$$
$$B = {lat(b_pawnc, sym=True)} \Rightarrow M = {lat(m_pawnc)}$$
$$B = {lat(b_ep, sym=True)} \Rightarrow M = {lat(m_ep)}$$
<p>Matrices de coup spéciales du pion (plateau aplati, $b' = T\,b$) :</p>
$$T_{{\text{{promotion}}}} = I - E_{{ff}} - E_{{tt}} + k\,E_{{tf}} \quad (k = 5 \text{{ dame}},\, 4,\, 3,\, 2 ;
\text{{ le pion vaut }} \pm 1 \Rightarrow k\cdot(\pm 1) \text{{ signe la pièce promue}})$$
$$T_{{\text{{en passant}}}} = I - E_{{ff}} - E_{{tt}} + E_{{tf}} - E_{{cc}}
\quad (c = \text{{case du pion capturé, sur la rangée de départ}})$$
""")

    sec("Roque (roi + tour, une seule matrice)", r"""
<p>Le roque déplace deux pièces mais reste <b>une</b> matrice — la somme des deux
opérateurs élémentaires dans le même $T$ (petit roque blanc : e1→g1, h1→f1) :</p>
$$T_{\text{O-O}} = I - E_{e1,e1} - E_{g1,g1} + E_{g1,e1} - E_{h1,h1} - E_{f1,f1} + E_{f1,h1}$$
<p>Conditions (matricielles aussi) : cases traversées vides ($\wedge\, E$) et non
attaquées — le masque d'attaque adverse est l'union des masques bruts ci-dessus,
$\mathrm{Att}(\text{adv}) = \bigvee_{\text{pièces adverses}} M_{\text{brut}}$, et l'échec
lui-même est la réduction $\;\mathrm{roi} \wedge \mathrm{Att} \neq 0$.</p>
""")

    nav = " · ".join(f'<a href="#s{i}">{t}</a>' for i, (t, _) in enumerate(S))
    body = "\n".join(f'<section id="s{i}"><h2>{t}</h2>{b}</section>' for i, (t, b) in enumerate(S))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>♟ Matrices par pièce</title>
<script>MathJax = {{tex: {{inlineMath: [['$','$']]}}, chtml: {{scale: 0.92}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
body {{ font-family: system-ui, sans-serif; background:#14151a; color:#e8e8ea; margin:0; }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 22px; }} h2 {{ color:#8ab4f8; border-bottom:1px solid #333; padding-bottom:6px; margin-top:40px; }}
nav {{ position: sticky; top:0; background:#1d1f26; padding:10px 24px; font-size:14px; z-index:2; }}
nav a {{ color:#8ab4f8; text-decoration:none; }} nav a:hover {{ text-decoration:underline; }}
mjx-container[display="true"] {{ overflow-x:auto; overflow-y:hidden; max-width:100%; }}
code {{ background:#2a2c34; padding:1px 5px; border-radius:4px; }}
p {{ line-height: 1.55; }}
.back {{ float:right; }}
</style></head><body>
<nav>{nav} <a class="back" href="/">← retour au jeu</a></nav>
<main>
<h1>♟ Opérations matricielles de chaque pièce</h1>
<p>Toutes les matrices d'exemple de cette page sont <b>calculées par le moteur</b>
(<code>src/matrix_chess/engine.py</code>) au chargement — la page est générée depuis le code,
pas recopiée. $\\bar{{P}}$ = pièce noire.</p>
{body}
</main></body></html>"""


PAGE = build()
