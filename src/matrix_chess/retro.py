"""Analyse rétrograde matricielle : remonter l'arbre des coups à l'envers.

Sans cimetière, l'inverse d'un coup n'est pas une fonction mais une RELATION :
    b_prev = R · b + c · e_t,   c ∈ {0, pièces restaurables}
R = I − E_tt − E_ff + (1/k)·E_ft défait le déplacement ; le paramètre libre c
est l'information détruite par la prise, qui réapparaît en arrière sous forme
de branches. Remonter une partie = explorer cet arbre de préimages, en ne
gardant que les prédécesseurs dont le coup avant est légal.

Recherche : rencontre au milieu — préimages depuis l'état final (arrière) ∩
coups légaux depuis l'état de départ (avant), sur `plies` demi-coups.
"""

import numpy as np

from .engine import (BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, Game,
                     KING_DIRS, KNIGHT_DIRS, DIAG_DIRS, ORTHO_DIRS,
                     attacked_by, in_check, piece_moves, ray_attacks, sq,
                     step_attacks, to_uci)


# ---------------------------------------------------------------------------
# Prédicats de condition finale (tout en masques matriciels)

def checker_kinds(board, color):
    """Types des pièces adverses donnant échec au roi de `color`."""
    king = np.argwhere(board == color * KING)
    if not len(king):
        return []
    kr, kc = king[0]
    kinds = []
    for r, c in zip(*np.nonzero(board * color < 0)):
        kind = abs(board[r, c])
        pos = np.zeros((8, 8), dtype=bool)
        pos[r, c] = True
        if kind == PAWN:
            atk = step_attacks(pos, [(color, -1), (color, 1)])  # avance du camp -color
        elif kind == KNIGHT:
            atk = step_attacks(pos, KNIGHT_DIRS)
        elif kind == KING:
            atk = step_attacks(pos, KING_DIRS)
        else:
            dirs = {BISHOP: DIAG_DIRS, ROOK: ORTHO_DIRS, QUEEN: KING_DIRS}[kind]
            atk = ray_attacks(pos, board == 0, dirs)
        if atk[kr, kc]:
            kinds.append(kind)
    return kinds


def is_pawn_mate(game):
    """Mat donné par un pion."""
    return game.status() == "checkmate" and PAWN in checker_kinds(game.board, game.turn)


def is_smothered_mate(game):
    """Mat étouffé : échec de cavalier seul, toutes les cases voisines du roi
    occupées par ses propres pièces."""
    if game.status() != "checkmate":
        return False
    if checker_kinds(game.board, game.turn) != [KNIGHT]:
        return False
    king = np.zeros((8, 8), dtype=bool)
    kr, kc = np.argwhere(game.board == game.turn * KING)[0]
    king[kr, kc] = True
    neigh = step_attacks(king, KING_DIRS)
    return bool((((game.board * game.turn) > 0) | ~neigh).all())


# ---------------------------------------------------------------------------
# Génération des prédécesseurs (dé-coups)

def _restores(board, S, r):
    """Pièces de couleur S restaurables sur une case de la rangée r."""
    out = [S * k for k in (KNIGHT, BISHOP, ROOK, QUEEN)]
    if 0 < r < 7 and int((board == S * PAWN).sum()) < 8:
        out.append(S * PAWN)
    return out


def predecessors(board, mover):
    """Tous les (plateau_précédent, coup) tels que `mover` jouant `coup`
    depuis plateau_précédent produit exactement `board`.

    Dé-coup = R·b + c·e_t : le déplacement est défait, c parcourt {0} ∪ pièces
    restaurables (branches de l'information perdue). Filtres de légalité :
    - `mover` ne peut pas être en échec après son coup (test global sur board) ;
    - dans la position précédente, l'adversaire ne doit pas être en échec.

    Contrat : cohérence à UN demi-coup (préimage + coup ⇒ exactement `board`,
    légalement). La légalité profonde (atteignabilité depuis b0) n'est pas
    décidée ici : de rares préimages (~0,1 %) sont des impasses rétrogrades
    (ex. double échec improductible) — leur propre ensemble de préimages est
    vide, donc les chaînes construites par `connect` ne les traversent pas.
    """
    S = -mover
    if in_check(board, mover):
        return []
    out = []
    empty = board == 0
    last = 0 if mover == 1 else 7          # rangée de promotion du mover
    back = mover                           # +1 pour blancs : l'origine est "derrière"

    def emit(prev, move):
        if not in_check(prev, S):
            out.append((prev, move))

    for r, c in zip(*np.nonzero(board * mover > 0)):
        kind = abs(board[r, c])
        piece = board[r, c]
        t = sq(r, c)
        if kind != PAWN:
            # dé-coup normal : origines vides atteignables (géométrie symétrique)
            for fr, fc in zip(*np.nonzero(piece_moves(board, r, c) & empty)):
                f = sq(int(fr), int(fc))
                for restored in [0] + _restores(board, S, r):
                    prev = board.copy()
                    prev[fr, fc] = piece
                    prev[r, c] = restored
                    emit(prev, (f, t, None, None, None))
            # dé-promotion : la pièce était un pion une rangée plus bas
            if r == last and kind in (KNIGHT, BISHOP, ROOK, QUEEN):
                fr = r + back
                for dc in (-1, 0, 1):
                    fc = c + dc
                    if not (0 <= fc < 8) or not empty[fr, fc]:
                        continue
                    rest = _restores(board, S, r) if dc else [0]
                    rest = [x for x in rest if abs(x) != PAWN]  # pas de pion en rangée 0/7
                    for restored in rest:
                        prev = board.copy()
                        prev[fr, fc] = mover * PAWN
                        prev[r, c] = restored
                        emit(prev, (sq(fr, fc), t, kind, None, None))
            # dé-roque
            if kind == KING:
                row = 7 if mover == 1 else 0
                right = "K" if mover == 1 else "k"
                if (r, c) == (row, 6) and board[row, 5] == mover * ROOK \
                        and empty[row, 4] and empty[row, 7]:
                    prev = board.copy()
                    prev[row, 6] = prev[row, 5] = 0
                    prev[row, 4], prev[row, 7] = mover * KING, mover * ROOK
                    g = Game(prev, turn=mover, castling=right)
                    mv = (sq(row, 4), sq(row, 6), None, None, (sq(row, 7), sq(row, 5)))
                    if mv in g.legal_moves():
                        out.append((prev, mv))
                right = "Q" if mover == 1 else "q"
                if (r, c) == (row, 2) and board[row, 3] == mover * ROOK \
                        and empty[row, 4] and empty[row, 0] and empty[row, 1]:
                    prev = board.copy()
                    prev[row, 2] = prev[row, 3] = 0
                    prev[row, 4], prev[row, 0] = mover * KING, mover * ROOK
                    g = Game(prev, turn=mover, castling=right)
                    mv = (sq(row, 4), sq(row, 2), None, None, (sq(row, 0), sq(row, 3)))
                    if mv in g.legal_moves():
                        out.append((prev, mv))
        else:
            fr = r + back
            if not (1 <= fr <= 6):
                continue
            # avance simple (jamais de prise en avançant)
            if empty[fr, c]:
                prev = board.copy()
                prev[fr, c], prev[r, c] = piece, 0
                emit(prev, (sq(fr, c), t, None, None, None))
            # double pas
            start = 6 if mover == 1 else 1
            fr2 = r + 2 * back
            if fr2 == start and empty[fr, c] and empty[fr2, c]:
                prev = board.copy()
                prev[fr2, c], prev[r, c] = piece, 0
                emit(prev, (sq(fr2, c), t, None, None, None))
            # prises diagonales (restauration obligatoire)
            for dc in (-1, 1):
                fc = c + dc
                if not (0 <= fc < 8) or not empty[fr, fc]:
                    continue
                for restored in _restores(board, S, r):
                    prev = board.copy()
                    prev[fr, fc] = piece
                    prev[r, c] = restored
                    emit(prev, (sq(fr, fc), t, None, None, None))
                # dé-en-passant : le pion capturé était à côté de l'origine,
                # et venait de doubler depuis (r − back, c) — qui doit être vide
                ep_row = 2 if mover == 1 else 5
                if r == ep_row and empty[r + back, c] and empty[r - back, c] \
                        and int((board == S * PAWN).sum()) < 8:
                    prev = board.copy()
                    prev[fr, fc] = piece
                    prev[r, c] = 0
                    prev[r + back, c] = S * PAWN
                    if not in_check(prev, S):
                        g = Game(prev, turn=mover, castling="", ep_square=(r, c))
                        mv = (sq(fr, fc), t, None, sq(r + back, c), None)
                        if mv in g.legal_moves():
                            out.append((prev, mv))
    return out


# ---------------------------------------------------------------------------
# Rencontre au milieu : relier un état de départ à un état final en p demi-coups

def _key(board, turn):
    return board.tobytes() + (b"w" if turn == 1 else b"b")


def connect(start, final_board, final_turn, plies, node_cap=400_000):
    """Chemin de `start` (Game) vers `final_board` en exactement `plies` coups.

    Avant : BFS sur les coups légaux depuis start (plies // 2 niveaux).
    Arrière : BFS sur les préimages depuis l'état final (le reste).
    Retourne (ucis, n_histoires) — n_histoires = nombre de nœuds de rencontre,
    i.e. combien de passés distincts mènent à ce futur — ou (None, 0).
    """
    fa = plies // 2
    fwd = [{_key(start.board, start.turn): (start, [])}]
    for _ in range(fa):
        nxt = {}
        for g, path in fwd[-1].values():
            for m in g.legal_moves():
                h = Game(g.board.copy(), g.turn, g.castling, g.ep_square)
                h.push(m)
                nxt.setdefault(_key(h.board, h.turn), (h, path + [to_uci(m)]))
        fwd.append(nxt)

    bwd = {_key(np.asarray(final_board), final_turn):
           (np.asarray(final_board), final_turn, [])}
    for depth in range(plies - fa):
        # rencontre : niveau arrière `depth` ↔ niveau avant `plies - depth`
        i = plies - depth
        if i <= fa:
            hits = set(fwd[i]) & set(bwd)
            if hits:
                k = sorted(hits)[0]
                return fwd[i][k][1] + bwd[k][2], len(hits)
        nxt = {}
        for b, turn, path in bwd.values():
            for prev, mv in predecessors(b, -turn):
                nxt.setdefault(_key(prev, -turn), (prev, -turn, [to_uci(mv)] + path))
            if len(nxt) > node_cap:
                break
        bwd = nxt
    hits = set(fwd[fa]) & set(bwd)
    if hits:
        k = sorted(hits)[0]
        return fwd[fa][k][1] + bwd[k][2], len(hits)
    return None, 0
