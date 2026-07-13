"""Imposition générale d'une condition finale : rembobinage depuis un prédicat.

`impose(position, prédicat, max_plies)` fonctionne depuis N'IMPORTE quelle
position et pour N'IMPORTE quel prédicat sur l'état final (mat étouffé, mat
par pion, mat quelconque, …). Principe :

1. SYNTHÈSE DES BUTS : on ne connaît pas le plateau final — on l'invente.
   On énumère les éditions minimales du plateau courant autour du roi cible
   (≤ max_changes pièces déplacées dans la zone du roi, ou sacrifiées),
   filtrées par des conditions nécessaires matricielles (échec + aucune case
   de fuite libre) puis par le prédicat lui-même. Matériel conservé : aucune
   pièce n'apparaît, seules des pièces existantes bougent ou disparaissent.
2. RELIAISON RÉTROGRADE : chaque but candidat (trié par nombre de changements
   puis distance de déplacement) est relié à la position courante par
   rencontre au milieu (`connect`) — la préimage b_prev = R·b + c·e_t fait
   le travail. Le premier but relié donne la ligne de coups.

Le prédicat est une boîte noire Game -> bool : le mécanisme est indépendant
du motif de mat recherché.
"""

import itertools

import numpy as np

from .engine import (KING, KING_DIRS, Game, attacked_by, in_check,
                     piece_moves, step_attacks)


def maybe_mate(board, mated):
    """Condition nécessaire (pas suffisante) de mat, en pur calcul de masques :
    roi en échec et aucune case voisine à la fois libre et non attaquée."""
    if not in_check(board, mated):
        return False
    king = board == mated * KING
    neigh = step_attacks(king, KING_DIRS)
    free = neigh & ~((board * mated) > 0)
    return bool((~free | attacked_by(board, -mated)).all())


def synthesize_goals(game, predicate, max_plies=3, max_changes=3, radius=2):
    """Plateaux finaux satisfaisant `predicate`, par édition minimale.

    Une édition = (pièce, destination) avec destination ∈ zone du roi cible
    (rayon `radius`) ou None (pièce capturée en route). Bornes de faisabilité :
    le camp qui mate joue ceil(p/2) coups, l'adversaire floor(p/2).
    Rend les buts triés par (nb d'éditions, déplacement total).
    """
    mover, mated = game.turn, -game.turn
    board = game.board
    kr, kc = np.argwhere(board == mated * KING)[0]
    box = [(r, c) for r in range(max(0, kr - radius), min(8, kr + radius + 1))
           for c in range(max(0, kc - radius), min(8, kc + radius + 1))]
    pieces = [(int(r), int(c), int(board[r, c]))
              for r, c in zip(*np.nonzero(board))]
    budget_mover = (max_plies + 1) // 2
    budget_mated = max_plies // 2

    def targets(piece):
        r0, c0, v = piece
        out = [(r, c) for r, c in box
               if abs(v) != 1 or 0 < r < 7]          # pas de pion en rangée 0/7
        if np.sign(v) == mated and abs(v) != KING:
            # coup d'attente du camp maté : n'importe quelle case atteignable
            # en un coup (approximation sur le plateau courant)
            out += [(int(r), int(c)) for r, c
                    in zip(*np.nonzero(piece_moves(board, r0, c0) & (board == 0)))]
        if abs(v) != KING:
            out.append(None)                          # sacrifiée / capturée
        return list(dict.fromkeys(out))

    goals = []
    for m in range(1, max_changes + 1):
        for combo in itertools.combinations(pieces, m):
            for dests in itertools.product(*(targets(p) for p in combo)):
                placed = [d for d in dests if d is not None]
                if len(set(placed)) != len(placed):
                    continue
                # bornes de coups par camp
                mv_moved = sum(1 for p, d in zip(combo, dests)
                               if d is not None and np.sign(p[2]) == mover)
                mt_moved = sum(1 for p, d in zip(combo, dests)
                               if d is not None and np.sign(p[2]) == mated)
                mv_del = sum(1 for p, d in zip(combo, dests)
                             if d is None and np.sign(p[2]) == mover)
                mt_del = sum(1 for p, d in zip(combo, dests)
                             if d is None and np.sign(p[2]) == mated)
                if mv_moved > budget_mover or mt_moved + mv_del > budget_mated \
                        or mt_del > budget_mover:
                    continue
                b = board.copy()
                for (r, c, _v) in combo:
                    b[r, c] = 0
                ok = True
                dist = 0
                for (r, c, v), d in zip(combo, dests):
                    if d is None:
                        continue
                    if b[d] != 0 or (r, c) == d:
                        ok = False
                        break
                    b[d] = v
                    dist += max(abs(d[0] - r), abs(d[1] - c))
                if not ok or not maybe_mate(b, mated):
                    continue
                g = Game(b, turn=mated, castling="")
                if predicate(g):
                    goals.append((m, dist, b))
    goals.sort(key=lambda x: (x[0], x[1]))
    return [b for _, _, b in goals]


def impose(game, predicate, max_plies=3, max_changes=3, goal_cap=200, radius=2):
    """Cherche une suite de coups depuis `game` menant à un état final
    satisfaisant `predicate`. Retourne (ucis, plateau_final) ou (None, None).

    Rencontre au milieu asymétrique : l'avant (coups légaux) est développé UNE
    fois par profondeur et partagé entre tous les buts ; chaque but synthétisé
    ne coûte qu'un niveau de préimages. Profondeurs impaires croissantes
    (le camp au trait donne le mat).
    # ponytail: arrière limité à 1 niveau — au-delà de p=5 l'avant explose,
    # passer à un arrière multi-niveaux partagé si besoin de profondeur.
    """
    from .engine import to_uci
    from .retro import predecessors
    mover = game.turn
    fwd = [{game.board.tobytes(): (game, [])}]
    for p in range(1, max_plies + 1, 2):
        while len(fwd) - 1 < p - 1:                    # étendre l'avant à p−1
            nxt = {}
            for g, path in fwd[-1].values():
                for m in g.legal_moves():
                    h = Game(g.board.copy(), g.turn, g.castling, g.ep_square)
                    h.push(m)
                    nxt.setdefault(h.board.tobytes(), (h, path + [to_uci(m)]))
            fwd.append(nxt)
        level = fwd[p - 1]                             # trait = mover (p impair)
        for b in synthesize_goals(game, predicate, max_plies=p,
                                  max_changes=max_changes, radius=radius)[:goal_cap]:
            for prev, mv in predecessors(b, mover):
                hit = level.get(prev.tobytes())
                if hit is None:
                    continue
                path = hit[1] + [to_uci(mv)]
                g = Game(game.board.copy(), game.turn, game.castling, game.ep_square)
                for u in path:                          # re-vérification complète
                    g.push(next(m for m in g.legal_moves() if to_uci(m) == u))
                if predicate(g):
                    return path, g.board
    return None, None
