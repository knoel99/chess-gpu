"""Génération de coups, perft et recherche entièrement sur GPU (PyTorch), par lots de positions.

Données (un lot = N positions, même camp au trait) :
- B  : (N, 65) int8, plateaux au format de reduced.py (64 cases + sentinelle NONE = 64 toujours nulle) ;
- cr : (N, 4) bool, droits de roque K, Q, k, q ;
- ep : (N,) int64, case en passant ou NONE.

Coups pseudo-légaux = 70 plans booléens (N, 70, 8, 8) indexés par la case de départ, tous obtenus
par shift + opérations élémentaires (comme engine.py) :
- 0..55  : 8 directions (KING_DIRS) × 7 distances, pour fou/tour/dame (et le roi à distance 1) ;
- 56..63 : cavalier ; 64..67 : pion (poussée simple, double, prise à gauche, à droite) ;
- 68, 69 : petit et grand roque.
Le plan donne le déplacement, donc la case d'arrivée : nonzero() les décode en coups (M, 6) au format
reduced.pack, appliqués par reduced_gpu.apply_moves ; in_check écarte les illégaux, en un lot.

Recherche : negamax pleine largeur, niveau par niveau (aucun élagage alpha-bêta, tout l'arbre est
développé en parallèle), puis quiescence (prises seulement), évaluation matériel + tables de cases.

Tests  : python -m unittest src.matrix_chess.test_gpu_engine -v
Débit  : python -m src.matrix_chess.gpu_engine [profondeur_perft]
"""

import time

import torch

from . import gpu_kernels
from .engine import BISHOP, DIAG_DIRS, KING, KING_DIRS, KNIGHT, KNIGHT_DIRS, PAWN, QUEEN, ROOK, WHITE
from .reduced_gpu import apply_moves, attacked_by, shift

NONE = 64
N_PLANES = 70
PROMOS = (QUEEN, ROOK, BISHOP, KNIGHT)
MATE = 30000.0


def plane_offsets(color):
    """(dr, dc) de chaque plan pour le camp `color` (seuls les plans de pion en dépendent)."""
    fwd = -color
    off = [(dr * k, dc * k) for dr, dc in KING_DIRS for k in range(1, 8)]
    off += KNIGHT_DIRS
    off += [(fwd, 0), (2 * fwd, 0), (fwd, -1), (fwd, 1), (0, 2), (0, -2)]
    return off


def _rays():
    """Cases vues depuis chaque case : RAYS (64, 8, 7) par direction de KING_DIRS (de la plus proche à la
    plus lointaine), KN (64, 8) sauts de cavalier, KG (64, 8) voisins ; NONE (toujours vide) hors plateau."""
    def at(r, c):
        return 8 * r + c if 0 <= r < 8 and 0 <= c < 8 else NONE
    rays = [[[at(s // 8 + dr * k, s % 8 + dc * k) for k in range(1, 8)] for dr, dc in KING_DIRS] for s in range(64)]
    kn = [[at(s // 8 + dr, s % 8 + dc) for dr, dc in KNIGHT_DIRS] for s in range(64)]
    kg = [[at(s // 8 + dr, s % 8 + dc) for dr, dc in KING_DIRS] for s in range(64)]
    # pions adverses qui attaqueraient un roi de couleur c : une rangée devant lui (sens de ses pions)
    pw = {c: [[at(s // 8 - c, s % 8 + dc) for dc in (-1, 1)] for s in range(64)] for c in (1, -1)}
    return rays, kn, kg, pw


_cache = {}


def _consts(dev):
    """Tables par périphérique : décalage d'index par plan et par camp, cases qui font perdre un roque."""
    if dev not in _cache:
        offs = {c: torch.tensor([8 * dr + dc for dr, dc in plane_offsets(c)], device=dev) for c in (1, -1)}
        lose = torch.zeros(65, 4, dtype=torch.bool)
        for sq_, rights in [(60, (0, 1)), (63, (0,)), (56, (1,)), (4, (2, 3)), (7, (2,)), (0, (3,))]:
            lose[sq_, list(rights)] = True
        rays, kn, kg, pw = _rays()
        geo = dict(rays=torch.tensor(rays, device=dev).view(64, 56), kn=torch.tensor(kn, device=dev),
                   kg=torch.tensor(kg, device=dev), pw={c: torch.tensor(v, device=dev) for c, v in pw.items()})
        rank = torch.tensor([0, 1, 3, 3, 5, 9, 0], device=dev)  # roi : ne prend que des pièces non défendues
        _cache[dev] = offs, lose.to(dev), torch.tensor(PROMOS, device=dev), geo, rank
    return _cache[dev]


def king_square(B, color):
    """Case (N,) du roi de `color`."""
    return (B[:, :64] == color * KING).to(torch.int8).argmax(1)


def in_check(B, color, ksq=None):
    """Booléen (N,) : le roi de `color` est-il attaqué ? Regard depuis le roi seulement : 74 cases lues
    par plateau (un gather), au lieu des masques d'attaque de tout le camp adverse."""
    geo = _consts(B.device)[3]
    ksq = king_square(B, color) if ksq is None else ksq
    n = len(B)
    v = B.gather(1, geo["rays"][ksq]).view(n, 8, 7) * color  # pièces adverses < 0
    first = torch.zeros_like(v[:, :, 0])
    for k in range(7):  # première pièce rencontrée par direction
        first = torch.where(first == 0, v[:, :, k], first)
    diag, ortho = first[:, :4], first[:, 4:]
    hit = ((diag == -BISHOP) | (diag == -QUEEN)).any(1) | ((ortho == -ROOK) | (ortho == -QUEEN)).any(1)
    hit |= (B.gather(1, geo["kn"][ksq]) * color == -KNIGHT).any(1)
    hit |= (B.gather(1, geo["kg"][ksq]) * color == -KING).any(1)
    return hit | (B.gather(1, geo["pw"][color][ksq]) * color == -PAWN).any(1)


def pseudo_planes(B, cr, ep, color, captures_only=False):
    """Masque (N, 70, 64) des coups pseudo-légaux, indexé par (plan, case de départ)."""
    n = len(B)
    b = B[:, :64].view(n, 8, 8)
    s = b * color  # pièces du trait > 0
    own, empty = s > 0, b == 0
    epm = torch.zeros(n, 65, dtype=torch.bool, device=B.device)
    epm[torch.arange(n, device=B.device), ep] = True
    epm = epm[:, :64].view(n, 8, 8)
    tgt = s < 0 if captures_only else ~own
    planes = []
    ortho, diag, king = (s == ROOK) | (s == QUEEN), (s == BISHOP) | (s == QUEEN), s == KING
    for dr, dc in KING_DIRS:
        clear = diag if dr and dc else ortho  # glisseurs dont le chemin est encore libre
        for k in range(1, 8):
            t = shift(tgt, -k * dr, -k * dc)  # t[s] = tgt[s + k·d], faux hors plateau
            planes.append((clear | king) & t if k == 1 else clear & t)
            clear = clear & shift(empty, -k * dr, -k * dc)
    knight = s == KNIGHT
    planes += [knight & shift(tgt, -dr, -dc) for dr, dc in KNIGHT_DIRS]
    fwd = -color
    pawn = s == PAWN
    push1 = pawn & shift(empty, -fwd, 0)
    start = torch.zeros(8, 8, dtype=torch.bool, device=B.device)
    start[6 if color == WHITE else 1] = True
    if captures_only:  # seules les poussées qui promeuvent
        last = torch.zeros_like(start)
        last[1 if color == WHITE else 6] = True
        planes += [push1 & last, torch.zeros_like(push1)]
    else:
        planes += [push1, push1 & start & shift(empty, -2 * fwd, 0)]
    capt = (s < 0) | epm
    planes += [pawn & shift(capt, -fwd, 1), pawn & shift(capt, -fwd, -1)]
    zero = torch.zeros_like(push1)
    if captures_only:
        planes += [zero, zero]
    else:
        row, i = (7, 0) if color == WHITE else (0, 2)
        att = attacked_by(b, -color)[:, row]
        r = b[:, row] * color
        base = (r[:, 4] == KING) & ~att[:, 4]
        ks = base & cr[:, i] & (r[:, 7] == ROOK) & (r[:, 5] == 0) & (r[:, 6] == 0) & ~att[:, 5] & ~att[:, 6]
        qs = base & cr[:, i + 1] & (r[:, 0] == ROOK) & (r[:, 1] == 0) & (r[:, 2] == 0) & (r[:, 3] == 0) \
            & ~att[:, 2] & ~att[:, 3]
        pk, pq = zero.clone(), zero.clone()
        pk[:, row, 4], pq[:, row, 4] = ks, qs
        planes += [pk, pq]
    return torch.stack(planes, 1).view(n, N_PLANES, 64)


def decode(mask, B, ep, color):
    """Masque (N, 70, 64) -> (parent (M,), coups (M, 6) f, t, k, ec, f2, t2, poussée double (M,))."""
    offs, _, promos = _consts(B.device)[:3]
    n, p, f = mask.nonzero(as_tuple=True)
    t = f + offs[color][p]
    promo = (p >= 64) & (p < 68) & (t // 8 == (0 if color == WHITE else 7))
    if promo.any():  # une ligne par pièce de promotion
        reps = 1 + 3 * promo.long()
        idx = torch.repeat_interleave(torch.arange(len(n), device=B.device), reps)
        j = torch.arange(len(idx), device=B.device) - (reps.cumsum(0) - reps)[idx]
        n, p, f, t, promo = n[idx], p[idx], f[idx], t[idx], promo[idx]
        k = torch.where(promo, promos[j.clamp(max=3)], 1)
    else:
        k = torch.ones_like(f)
    return n, _pack(p, f, t, k, ep[n], color), p == 65


def _pack(p, f, t, k, ep, color):
    """Coups (M, 6) : case vidée en passant, déplacement de la tour du roque."""
    none = torch.full_like(f, NONE)
    ec = torch.where((p >= 66) & (p < 68) & (t == ep), (f // 8) * 8 + t % 8, none)
    row = 56 if color == WHITE else 0
    f2 = torch.where(p == 68, row + 7, torch.where(p == 69, row, none))
    t2 = torch.where(p == 68, row + 5, torch.where(p == 69, row + 3, none))
    return torch.stack([f, t, k, ec, f2, t2], 1)


def _children(B, cr, kp, m, double, color):
    """Plateaux enfants, masque de légalité et droits des enfants, tous de longueur M (non filtrés)."""
    lose = _consts(B.device)[1]
    child = apply_moves(B, m)
    ok = ~in_check(child, color, torch.where(m[:, 0] == kp, m[:, 1], kp))
    ccr = cr & ~lose[m[:, 0]] & ~lose[m[:, 1]]
    cep = torch.where(double, (m[:, 0] + m[:, 1]) // 2, torch.full_like(m[:, 0], NONE))
    return child, ok, ccr, cep


def expand(B, cr, ep, color, captures_only=False, count_only=False):
    """Enfants légaux du lot : (parent, coups, B', cr', ep'), ou leur nombre si count_only.
    Sur cuda avec CuPy : noyau fusionné de gpu_kernels ; sinon expand_torch (référence, même résultat)."""
    fused = gpu_kernels.expand if gpu_kernels.available(B.device) else expand_torch
    return fused(B, cr, ep, color, captures_only, count_only)


def expand_torch(B, cr, ep, color, captures_only=False, count_only=False):
    n, m, double = decode(pseudo_planes(B, cr, ep, color, captures_only), B, ep, color)
    if captures_only:
        # shortcut: quiescence sans alpha-bêta = arbre des prises complet, qui explose ; on ne garde que
        # les prises d'une pièce au moins aussi chère (MVV/LVA), les promotions et les prises du roi.
        # À remplacer par un SEE si la quiescence rate des gains de pièce.
        rank = _consts(B.device)[4]
        bn = B[n]
        victim = rank[bn.gather(1, m[:, 1:2]).abs().long().squeeze(1)]
        attacker = rank[bn.gather(1, m[:, :1]).abs().long().squeeze(1)]
        keep = (victim >= attacker) | (m[:, 2] != 1) | (m[:, 3] != NONE)
        n, m, double = n[keep], m[keep], double[keep]
    child, ok, ccr, cep = _children(B[n], cr[n], king_square(B, color)[n], m, double, color)
    if count_only:
        return int(ok.sum())
    return n[ok], m[ok], child[ok], ccr[ok], cep[ok]


def enable_compile():
    """Fusionne les opérations élément par élément en noyaux GPU (torch.compile, formes dynamiques)."""
    global pseudo_planes, in_check, _pack, _children
    pseudo_planes, in_check, _pack, _children = (torch.compile(f, dynamic=True)
                                                 for f in (pseudo_planes, in_check, _pack, _children))


def chunks(total, size):
    return ((i, min(i + size, total)) for i in range(0, total, size))


def expand_all(B, cr, ep, color, chunk, captures_only=False):
    """expand par morceaux de `chunk` parents (borne la mémoire), résultats concaténés."""
    parts = []
    for a, z in chunks(len(B), chunk):
        n, m, child, ccr, cep = expand(B[a:z], cr[a:z], ep[a:z], color, captures_only)
        parts.append((n + a, m, child, ccr, cep))
    return [torch.cat(x) for x in zip(*parts)]


def perft(B, cr, ep, color, depth, chunk=200_000):
    """Nombre de feuilles à `depth` demi-coups (comptage en masse au dernier niveau)."""
    total = 0
    for a, z in chunks(len(B), chunk):
        if depth == 1:
            total += expand(B[a:z], cr[a:z], ep[a:z], color, count_only=True)
        else:
            _, _, child, ccr, cep = expand(B[a:z], cr[a:z], ep[a:z], color)
            total += perft(child, ccr, cep, -color, depth - 1, chunk)
    return total


# --- interop avec engine.Game ---

def from_game(g, dev):
    """Game -> (B (1, 65), cr (1, 4), ep (1,)) sur `dev`."""
    B = torch.zeros(1, 65, dtype=torch.int8)
    B[0, :64] = torch.from_numpy(g.board.reshape(64).copy())
    cr = torch.tensor([[c in g.castling for c in "KQkq"]])
    ep = torch.tensor([NONE if g.ep_square is None else 8 * g.ep_square[0] + g.ep_square[1]])
    return B.to(dev), cr.to(dev), ep.to(dev)


def to_move(m):
    """Ligne (6,) -> coup du moteur (f, t, promo, extra_clear, also)."""
    f, t, k, ec, f2, t2 = (int(x) for x in m)
    return (f, t, None if k == 1 else k, None if ec == NONE else ec, None if f2 == NONE else (f2, t2))


# --- évaluation : matériel + tables de cases (Simplified Evaluation Function, T. Michniewski) ---
# tables du point de vue des blancs, ligne 0 = rangée 8 comme le plateau

PST = {
    PAWN: [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10,
           5, 5, 10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5,
           5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40,
             -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30,
             -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30,
             -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
             -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10,
             -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10,
             -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    ROOK: [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0],
    QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
    KING: [-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
           -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10,
           20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20],
}
VALUE = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}


def _eval_table(dev):
    """E (13, 65) : E[piece + 6, case] = valeur du point de vue des blancs ; sentinelle nulle."""
    E = torch.zeros(13, 65)
    for p, tab in PST.items():
        w = torch.tensor(tab, dtype=torch.float32) + VALUE[p]
        E[6 + p, :64] = w
        E[6 - p, :64] = -w.view(8, 8).flip(0).reshape(64)  # noirs : rangées en miroir
    return E.to(dev)


def evaluate(B, color, E, chunk=8_000_000):
    """Évaluation (N,) du point de vue du camp `color` (negamax), par morceaux (index int64 = 520 o/plateau)."""
    sq_ = torch.arange(65, device=B.device)
    return torch.cat([color * E[B[a:z].long() + 6, sq_].sum(1) for a, z in chunks(len(B), chunk)] or [E[:0, 0]])


# --- recherche : negamax pleine largeur + quiescence, par lots récursifs (mémoire bornée) ---

class Timeout(Exception):
    """Pendule dépassée au milieu d'une itération. `partial` : scores racine acquis avant
    l'interruption (-inf pour les coups non cherchés), ou None si le premier coup n'a pas fini."""

    partial = None


def position_keys(B, cr, ep, color):
    """Clés de répétition (une par plateau) : pièces, droits de roque, trait `color`.
    La case en passant est omise : une position où elle compte suit une poussée de deux cases et
    n'a donc jamais pu être vue avant."""
    rows = torch.cat([B, cr.to(torch.int8)], 1).cpu().numpy()
    return [r.tobytes() + (b"w" if color == WHITE else b"b") for r in rows]


class Searcher:
    """Choisit un coup par approfondissement itératif dans un budget de temps (secondes).

    Chaque appel récursif ne garde en mémoire qu'un lot d'au plus `chunk` parents et leurs enfants
    par niveau (comme perft) : seules les valeurs remontent, jamais l'arbre. La profondeur n'est
    donc bornée que par le temps, et une itération trop longue est interrompue à la pendule.
    """

    WAVES = {"ybw": (0, 1), "hybride": (0, 1, 4), "1-2-4": (0, 1, 3, 7, 15, 31), "adaptatif": None}
    # vagues adaptatives : (taille estimée du sous-arbre du lot en dessous de laquelle…, découpe) ; au-delà, 1-2-4
    SEUILS = ((3e4, (0,)), (3e7, (0, 1)), (3e9, (0, 1, 4)))  # calibrés sur RTX PRO 6000, optimum plat
    TRIS = ("statique", "mvv", "historique")

    def __init__(self, dev, chunk=None, qdepth=4, prune=False, waves="ybw", tri="mvv"):
        """waves : découpe des fils par rang en vagues successives (aîné seul, puis…), voir WAVES ;
        "adaptatif" choisit la découpe lot par lot (SEUILS). tri : ordre des fils, voir _key."""
        assert tri in self.TRIS, tri
        self.dev, self.prune, self.tri = dev, prune, tri
        self.waves = (*(self.WAVES[waves] or self.WAVES["1-2-4"]), 1 << 30)
        self.seuils = self.SEUILS if waves == "adaptatif" else ()
        self.w_hist, self.journal = 400.0, True  # poids de l'historique dans la clé ; compteurs de coupe actifs
        self.chunk = chunk or (400_000 if dev == "cuda" else 20_000)
        self.qdepth = qdepth
        self.E = _eval_table(dev)
        self.deadline, self.nodes = float("inf"), 0
        self.reset()
        if dev == "cuda":
            enable_compile()
            self.warmup()

    def reset(self):
        """Tables de tri et compteurs remis à zéro (une recherche = un résultat reproductible)."""
        self.hist = torch.zeros(2, 4096, device=self.dev)   # [camp, 64·départ + arrivée] : coups calmes qui coupent
        self.killer = torch.zeros(128, 4096, device=self.dev)  # idem par demi-coup depuis la racine
        self.cuts = torch.zeros(128, 3, device=self.dev)    # par demi-coup : parents développés, coupés, coupés par l'aîné

    def stats(self):
        """Grandeurs de convergence de la dernière recherche : taux de coupe en première vague
        (parents coupés par l'aîné / parents coupés), global et par demi-coup."""
        c = self.cuts.cpu()
        used = c[:, 0] > 0
        tot = c.sum(0)
        return dict(parents=int(tot[0]), cut=int(tot[1]), cut_first=int(tot[2]),
                    first_rate=float(tot[2] / tot[1].clamp(min=1)),
                    first_rate_by_ply=[round(float(x), 4) for x in (c[used, 2] / c[used, 1].clamp(min=1))])

    def _key(self, B, par, m, child, color, ply):
        """Clé de tri (N,) int64 < 2^32 des fils (petit = cherché d'abord), code 64·f + t et drapeau « calme ».
        "statique" : évaluation statique de l'enfant. "mvv" : prises et promotions d'abord, par victime
        décroissante puis attaquant croissant. "historique" : en plus, coups killer du demi-coup, puis
        coups calmes avancés selon la table d'historique."""
        ev = (evaluate(child, -color, self.E) + 65536).long().clamp(0, (1 << 17) - 1)  # petit pour l'adversaire = bon pour nous
        if self.tri == "statique":
            return ev, None, None
        r = _consts(B.device)[4][B[par.unsqueeze(1), m[:, :2]].abs().long()]  # rangs (attaquant, victime)
        vr = r[:, 1] + 8 * (m[:, 2] != 1) + (m[:, 3] != NONE)  # promotion, prise en passant
        key = torch.where(vr == 0, (2 << 30) + ev, ((400 - 16 * vr + r[:, 0]) << 17) + ev)
        if self.tri == "mvv":
            return key, None, None
        code, quiet = m[:, 0] * 64 + m[:, 1], vr == 0
        h = self.hist[(1 - color) // 2]
        adv = (self.w_hist * h[code] / h.max().clamp(min=1)).long()
        k = self.killer[ply]
        kil = (k[code] > 0) & (k[code] >= k.topk(2).values[1])
        return torch.where(quiet, torch.where(kil, 1 << 30, 2 << 30) + (ev - adv).clamp(min=0), key), code, quiet

    def _waves(self, n_par, n_child, remaining):
        """Découpe en vagues du lot : fixe, ou choisie d'après la taille estimée du sous-arbre
        (fils × branchement^(remaining − 1)) : petit lot = tout d'un coup, grand = échelonné fin."""
        est = n_child * (n_child / n_par) ** (remaining - 1)
        for s, w in self.seuils:
            if est < s:
                return (*w, 1 << 30)
        return self.waves

    def warmup(self):
        """Compile les noyaux pour les deux camps, avec et sans quiescence, hors pendule."""
        from .engine import Game
        g = Game()
        # troisième passe : in_check est recompilé une fois de plus pour le premier camp (garde sur _cache)
        for mv in ((52, 36, None, None, None), (12, 28, None, None, None), (62, 45, None, None, None)):
            self.search(*from_game(g, self.dev), g.turn, budget=0.0, min_depth=3, max_depth=3)
            g.push(mv)

    def value(self, B, cr, ep, color, ply, remaining, qleft, alpha=None, beta=None):
        """Valeurs negamax (N,) des positions, du point de vue du trait `color`.

        Avec self.prune : alpha-bêta par lots (fenêtres alpha, beta (N,), fail-soft). Les fils de chaque
        parent, triés (_key), sont cherchés par vagues de rangs (_waves, fixes ou adaptatives) : l'aîné
        seul, puis les suivants avec la fenêtre resserrée, seulement si le parent n'est pas déjà coupé
        (Young Brothers Wait pour (0, 1) ; plus de vagues = moins de nœuds mais des lots plus petits). En quiescence, le
        stand pat coupe avant toute prise. La valeur est exacte si alpha < v < beta, sinon une borne du
        bon côté ; sans self.prune, c'est le negamax pleine largeur exact.
        """
        inf, dev = float("inf"), B.device
        alpha = torch.full((len(B),), -inf, device=dev) if alpha is None else alpha
        beta = torch.full((len(B),), inf, device=dev) if beta is None else beta
        quiesce = remaining == 0
        if quiesce:
            stand = evaluate(B, color, self.E)
            if qleft == 0:
                return stand
            best = stand.clone()  # stand pat : le trait peut ne pas prendre
        else:
            best = torch.full((len(B),), -inf, device=dev)
        live = best < beta
        for a, z in chunks(len(B), self.chunk):
            if time.perf_counter() > self.deadline:
                raise Timeout
            idx = live[a:z].nonzero().squeeze(1) + a if self.prune else torch.arange(a, z, device=dev)
            if not len(idx):
                continue
            n, m, child, ccr, cep = expand(B[idx], cr[idx], ep[idx], color, captures_only=quiesce)
            self.nodes += len(n)
            if not len(n):
                continue
            par = idx[n]
            waves = self._waves(len(idx), len(n), remaining) if self.prune and not quiesce else (0, 1 << 30)
            if self.prune and not quiesce:  # enfants groupés par parent, meilleur d'abord
                key, code, quiet = self._key(B, par, m, child, color, ply)
                o = ((par << 32) | key).argsort(stable=True)
                par, child, ccr, cep = par[o], child[o], ccr[o], cep[o]
                first = torch.ones_like(par, dtype=torch.bool)
                first[1:] = par[1:] != par[:-1]
                pos = torch.arange(len(par), device=dev)
                rank = pos - torch.cummax(torch.where(first, pos, 0), 0).values  # rang du coup chez son parent
                stages = [(rank >= lo) & (rank < hi) for lo, hi in zip(waves, waves[1:])]
            else:
                stages = [None]
            for k, sel in enumerate(stages):
                if sel is not None and k >= 1:
                    sel = sel & (best[par] < beta[par])  # cadets des parents non coupés
                p, c, cc, ce = (par, child, ccr, cep) if sel is None else (par[sel], child[sel], ccr[sel], cep[sel])
                if not len(p):
                    continue
                v = self.value(c, cc, ce, -color, ply + 1, max(remaining - 1, 0), qleft - quiesce,
                               -beta[p], -torch.maximum(alpha[p], best[p]))
                best = best.scatter_reduce(0, p, -v, "amax")
                if sel is None:
                    continue
                if k == 0 and self.journal:
                    self.cuts[ply, 2] += ((-v >= beta[p]) & first[sel]).sum()
                if code is not None:  # les coups calmes qui coupent montent dans les tables
                    qc = code[o][sel][(-v >= beta[p]) & quiet[o][sel]]
                    self.hist[(1 - color) // 2].index_add_(0, qc, torch.full(qc.shape, float(remaining ** 2), device=dev))
                    self.killer[ply].index_add_(0, qc, torch.ones(qc.shape, device=dev))
            if self.prune and not quiesce and self.journal:
                self.cuts[ply, 0] += len(idx)
                self.cuts[ply, 1] += (best[idx] >= beta[idx]).sum()
        if quiesce:
            return best
        chk = torch.cat([in_check(B[a:z], color) for a, z in chunks(len(B), 8_000_000)])
        return torch.where(torch.isinf(best), torch.where(chk, -MATE + ply, 0.0), best)  # mat ou pat

    def root_scores(self, child, ccr, cep, color, depth, prev):
        """Scores des coups racine, cherchés par groupes de 1, 1, 2, 4, 8… coups dans l'ordre de
        `prev` (scores de l'itération précédente), le meilleur d'abord et seul.

        Avec self.prune : le premier en fenêtre pleine, les groupes suivants avec la fenêtre
        (-inf, -meilleur score acquis). Un coup qui dépasse ce meilleur a un score exact ; sinon son
        score n'est qu'une borne haute, peut-être égale au meilleur : on la place strictement en
        dessous pour que l'argmax ne choisisse jamais un coup réfuté (les scores sont entiers).

        Interrompue par la pendule, lève Timeout avec `partial` = scores des groupes terminés. Leur
        argmax est sûr dès que le premier coup a fini : c'est soit lui (score exact à `depth`), soit
        un coup dont le score exact à `depth` dépasse le sien."""
        inf, dev = float("inf"), child.device
        o = torch.arange(len(child), device=dev) if prev is None else prev.argsort(descending=True, stable=True)
        s = torch.full((len(child),), -inf, device=dev)
        a, z = 0, 1
        try:
            while a < len(o):
                g = o[a:z]
                if self.prune and a:
                    top = s.max()
                    r = -self.value(child[g], ccr[g], cep[g], -color, 1, depth - 1, self.qdepth,
                                    torch.full((len(g),), -inf, device=dev), -top.expand(len(g)))
                    s[g] = torch.where(r > top, r, torch.minimum(r, top) - 0.5)
                else:
                    s[g] = -self.value(child[g], ccr[g], cep[g], -color, 1, depth - 1, self.qdepth)
                a, z = z, 2 * z
        except Timeout as e:
            e.partial = s if a else None
            raise
        return s

    def search(self, B, cr, ep, color, budget, hard=None, min_depth=1, max_depth=64, history=()):
        """Approfondit tant que l'itération suivante, prédite par le facteur de branchement effectif
        mesuré (rapport des durées de deux itérations de même parité), finit avant `hard` (défaut 1,5 × budget). Une itération qui dépasse quand même `hard` est
        interrompue : on garde son résultat partiel s'il est sûr (voir root_scores), sinon le coup de
        la dernière profondeur complète.

        history : clés (position_keys) des positions déjà vues dans la partie. Un coup racine qui
        ramène à l'une d'elles vaut 0 (nulle) : évité quand on mène, choisi quand on est perdu.
        shortcut: répétitions vues à la racine seulement ; l'adversaire peut encore répéter par un
        autre chemin au deuxième demi-coup. À étendre dans value() si des parties le montrent."""
        t0 = time.perf_counter()
        hard = 1.5 * budget if hard is None else hard
        n, m, child, ccr, cep = expand(B, cr, ep, color)
        if len(m) <= 1:
            return (to_move(m[0]) if len(m) else None), dict(depth=0, nodes=len(m), nps=0.0, score=0.0, time=0.0)
        seen = torch.tensor([k in history for k in position_keys(child, ccr, cep, -color)] if history
                            else [False] * len(m), device=B.device)
        new = (~seen).nonzero().squeeze(1)  # seuls les coups vers une position nouvelle sont cherchés
        child, ccr, cep = child[new], ccr[new], cep[new]
        self.nodes, iters = len(m), []  # iters : (nœuds, durée) de chaque itération complète
        self.reset()
        prev = -evaluate(child, -color, self.E)  # ordre des coups racine : évaluation statique, puis itération précédente
        best, info = to_move(m[0]), dict(depth=0, nodes=len(m), nps=0.0, score=0.0, time=0.0)  # si tous les coups répètent

        def result(score, depth, **extra):
            full = torch.zeros(len(m), device=B.device)
            full[new] = score
            i, elapsed = int(full.argmax()), time.perf_counter() - t0
            return to_move(m[i]), dict(depth=depth, nodes=self.nodes, nps=self.nodes / max(elapsed, 1e-9),
                                       score=float(full[i]), time=elapsed, **extra)

        for depth in range(1, max_depth + 1 if len(new) else 0):
            if depth > min_depth and iters:
                # alpha-bêta alterne plis chers et bon marché : le facteur de branchement effectif se lit
                # sur les durées à deux plis d'écart (meilleur prédicteur mesuré, encore faux d'un facteur 3 en
                # moyenne : c'est l'interruption à `hard` et le résultat partiel qui bornent le temps)
                d = [x[1] for x in iters]
                ebf = d[-2] / d[-3] if len(d) > 2 else d[-1] / d[-2] if len(d) == 2 else 4.0
                if time.perf_counter() - t0 + d[-1] * min(max(ebf, 1.5), 8.0) > hard:  # bornes : durées courtes bruitées
                    break
            t_iter, n_iter = time.perf_counter(), self.nodes
            self.deadline = t0 + hard if depth > min_depth else float("inf")
            try:
                prev = self.root_scores(child, ccr, cep, color, depth, prev)
            except Timeout as e:
                if e.partial is not None:
                    best, info = result(e.partial, depth - 1, partial=depth)
                info["aborted"] = depth
                break
            iters.append((self.nodes - n_iter, time.perf_counter() - t_iter))
            best, info = result(prev, depth)
            if depth >= min_depth and (abs(info["score"]) > MATE - 100 or info["time"] > budget):
                break
        self.deadline = float("inf")
        elapsed = time.perf_counter() - t0  # travail d'une itération interrompue compris
        info.update(nodes=self.nodes, time=elapsed, nps=self.nodes / max(elapsed, 1e-9),
                    iters=[(k, round(d, 4)) for k, d in iters])
        return best, info


if __name__ == "__main__":
    import sys

    from .engine import Game

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else (400_000 if dev == "cuda" else 20_000)
    if dev == "cuda":
        enable_compile()
    B, cr, ep = from_game(Game(), dev)
    for d in range(1, depth + 1):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        nodes = perft(B, cr, ep, WHITE, d, chunk)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{dev} perft({d}) = {nodes} en {dt:.3f} s -> {nodes / dt:.3e} nœuds/s")
