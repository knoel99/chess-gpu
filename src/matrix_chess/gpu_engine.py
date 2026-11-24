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
        _cache[dev] = offs, lose.to(dev), torch.tensor(PROMOS, device=dev), geo
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
    occ = v != 0
    first = (v * (occ & (occ.cumsum(2) == 1))).sum(2)  # première pièce rencontrée par direction
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
    offs, _, promos, _ = _consts(B.device)
    n, p, f = mask.nonzero(as_tuple=True)
    t = f + offs[color][p]
    pawn = (p >= 64) & (p < 68)
    promo = pawn & (t // 8 == (0 if color == WHITE else 7))
    if promo.any():  # une ligne par pièce de promotion
        reps = 1 + 3 * promo.long()
        idx = torch.repeat_interleave(torch.arange(len(n), device=B.device), reps)
        j = torch.arange(len(idx), device=B.device) - (reps.cumsum(0) - reps)[idx]
        n, p, f, t, promo = n[idx], p[idx], f[idx], t[idx], promo[idx]
        k = torch.where(promo, promos[j.clamp(max=3)], 1)
    else:
        k = torch.ones_like(f)
    none = torch.full_like(f, NONE)
    ec = torch.where((p >= 66) & (p < 68) & (t == ep[n]), (f // 8) * 8 + t % 8, none)
    row = 56 if color == WHITE else 0
    f2 = torch.where(p == 68, row + 7, torch.where(p == 69, row, none))
    t2 = torch.where(p == 68, row + 5, torch.where(p == 69, row + 3, none))
    return n, torch.stack([f, t, k, ec, f2, t2], 1), p == 65


def expand(B, cr, ep, color, captures_only=False):
    """Enfants légaux du lot : (parent, coups, B', cr', ep')."""
    lose = _consts(B.device)[1]
    n, m, double = decode(pseudo_planes(B, cr, ep, color, captures_only), B, ep, color)
    child = apply_moves(B[n], m)
    kp = king_square(B, color)[n]
    ok = ~in_check(child, color, torch.where(m[:, 0] == kp, m[:, 1], kp))
    n, m, child, double = n[ok], m[ok], child[ok], double[ok]
    ccr = cr[n] & ~lose[m[:, 0]] & ~lose[m[:, 1]]
    cep = torch.where(double, (m[:, 0] + m[:, 1]) // 2, torch.full_like(n, NONE))
    return n, m, child, ccr, cep


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
        n, _, child, ccr, cep = expand(B[a:z], cr[a:z], ep[a:z], color)
        total += len(n) if depth == 1 else perft(child, ccr, cep, -color, depth - 1, chunk)
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


def evaluate(B, color, E):
    """Évaluation (N,) du point de vue du camp `color` (negamax)."""
    return color * E[B.long() + 6, torch.arange(65, device=B.device)].sum(1)


# --- recherche : negamax pleine largeur + quiescence, arbre développé niveau par niveau ---

def root_values(levels):
    """Remonte les valeurs du dernier niveau jusqu'aux enfants de la racine (levels[1]).

    Chaque niveau porte `parent` (indice dans le niveau précédent) et `stand`, sa valeur s'il n'a
    aucun enfant (-MATE + ply ou 0 en pleine largeur, évaluation statique en quiescence). En
    quiescence le trait peut aussi s'arrêter (stand pat) : v = max(stand, max des -v enfants).
    """
    v = levels[-1]["stand"]
    for parent, child in zip(reversed(levels[1:-1]), reversed(levels[2:])):
        best = torch.full_like(parent["stand"], -float("inf")).scatter_reduce(0, child["parent"], -v, "amax")
        v = torch.maximum(best, parent["stand"]) if parent["quiesce"] else \
            torch.where(torch.isinf(best), parent["stand"], best)
    return v


class Searcher:
    """Choisit un coup par approfondissement itératif dans un budget de temps (secondes)."""

    def __init__(self, dev, chunk=None, qdepth=4, max_nodes=None):
        self.dev = dev
        self.chunk = chunk or (400_000 if dev == "cuda" else 20_000)
        self.qdepth = qdepth
        self.max_nodes = max_nodes or (150_000_000 if dev == "cuda" else 3_000_000)
        self.E = _eval_table(dev)

    def search(self, B, cr, ep, color, budget, min_depth=1, max_depth=64):
        t0 = time.perf_counter()
        sync = torch.cuda.synchronize if self.dev == "cuda" else (lambda: None)

        def side(ply):
            return color if ply % 2 == 0 else -color

        def full_level(lv, ply):  # sans coup légal : mat (le plus tard possible) ou pat
            chk = in_check(lv["B"], side(ply))
            return dict(lv, stand=torch.where(chk, -MATE + ply, 0.0).float(), quiesce=False)

        full = [full_level(dict(B=B, cr=cr, ep=ep, parent=None), 0)]
        best, info = None, {}
        for depth in range(1, max_depth + 1):
            nodes = full[-1]
            n, m, child, ccr, cep = expand_all(nodes["B"], nodes["cr"], nodes["ep"], side(depth - 1), self.chunk)
            if depth == 1:
                root_moves = m
                if len(m) <= 1:
                    return (to_move(m[0]) if len(m) else None), dict(depth=0, nodes=len(m), nps=0.0, score=0.0)
            q = dict(B=child, cr=ccr, ep=cep, parent=n)
            total = sum(len(lv["B"]) for lv in full) + len(n)
            levels = full + [dict(q, stand=evaluate(child, side(depth), self.E), quiesce=True)]
            for qi in range(self.qdepth):  # quiescence : prises seulement, depuis les feuilles
                if not len(q["B"]) or total > self.max_nodes:
                    break
                c = side(depth + qi)
                qn, _, qb, qcr, qep = expand_all(q["B"], q["cr"], q["ep"], c, self.chunk, captures_only=True)
                q, total = dict(B=qb, cr=qcr, ep=qep, parent=qn), total + len(qn)
                levels.append(dict(q, stand=evaluate(qb, -c, self.E), quiesce=True))
            score = -root_values(levels)
            sync()
            elapsed = time.perf_counter() - t0
            i = int(score.argmax())
            best = to_move(root_moves[i])
            info = dict(depth=depth, nodes=total, nps=total / max(elapsed, 1e-9), score=float(score[i]),
                        time=elapsed)
            full.append(full_level(dict(B=child, cr=ccr, ep=cep, parent=n), depth))
            growth = len(child) / max(len(nodes["B"]), 1)
            if depth >= min_depth and (abs(info["score"]) > MATE - 100 or elapsed * (1 + growth) > budget
                                       or len(child) * growth > self.max_nodes):
                break
        return best, info


if __name__ == "__main__":
    import sys

    from .engine import Game

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else (400_000 if dev == "cuda" else 20_000)
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
