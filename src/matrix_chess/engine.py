"""Moteur d'échecs 100% matriciel (numpy).

Conventions (docs/king_matrix.md) :
- plateau B ∈ Z^{8×8} : +1..+6 = pion/cavalier/fou/tour/dame/roi blancs, négatif = noirs, 0 = vide
- ligne i=0 = rangée 8 (noirs), i=7 = rangée 1 (blancs) ; colonne j=0 = 'a'
- tout masque d'attaque/mouvement = shifts + opérations élémentaires (batch-aware : (..., 8, 8))
- tout coup = matrice T ∈ Z^{64×64} agissant sur le plateau aplati : b' = T @ b
"""

import numpy as np

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, -1

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KING_DIRS = DIAG_DIRS + ORTHO_DIRS
KNIGHT_DIRS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def shift(m, dr, dc):
    """Décale les deux derniers axes de (dr, dc), remplissage à zéro (bords clippés)."""
    out = np.zeros_like(m)
    rs_src = slice(max(0, -dr), 8 - max(0, dr))
    cs_src = slice(max(0, -dc), 8 - max(0, dc))
    rs_dst = slice(max(0, dr), 8 - max(0, -dr))
    cs_dst = slice(max(0, dc), 8 - max(0, -dc))
    out[..., rs_dst, cs_dst] = m[..., rs_src, cs_src]
    return out


def ray_attacks(pos, empty, dirs):
    """Propagation glissante avec blocage : union des rayons depuis les 1 de `pos`.

    Inclut la première case occupée rencontrée (cible de prise ou pièce amie,
    filtrée plus tard). Pur matriciel : 7 shifts max par direction.
    """
    acc = np.zeros_like(pos)
    for dr, dc in dirs:
        front = shift(pos, dr, dc)
        for _ in range(7):
            acc = acc | front
            front = shift(front & empty, dr, dc)
    return acc


def step_attacks(pos, dirs):
    """Union des shifts unitaires (roi, cavalier, diagonales de pion)."""
    acc = np.zeros_like(pos)
    for dr, dc in dirs:
        acc = acc | shift(pos, dr, dc)
    return acc


def attacked_by(board, color):
    """Masque booléen (..., 8, 8) des cases attaquées par `color`. Batch-aware."""
    empty = board == 0
    pawns = board == color * PAWN
    fwd = -color  # blancs (+1) montent vers i décroissant
    acc = step_attacks(pawns, [(fwd, -1), (fwd, 1)])
    acc = acc | step_attacks(board == color * KNIGHT, KNIGHT_DIRS)
    acc = acc | step_attacks(board == color * KING, KING_DIRS)
    acc = acc | ray_attacks((board == color * BISHOP) | (board == color * QUEEN), empty, DIAG_DIRS)
    acc = acc | ray_attacks((board == color * ROOK) | (board == color * QUEEN), empty, ORTHO_DIRS)
    return acc


def in_check(board, color):
    """Booléen (batch-aware) : le roi de `color` est-il attaqué ?"""
    king = board == color * KING
    return (king & attacked_by(board, -color)).any(axis=(-2, -1))


def piece_moves(board, r, c, ep_square=None):
    """Masque 8×8 des destinations pseudo-légales de la pièce en (r, c) (sans roque)."""
    piece = board[r, c]
    color = 1 if piece > 0 else -1
    pos = np.zeros((8, 8), dtype=bool)
    pos[r, c] = True
    empty = board == 0
    enemy = (board * color) < 0
    kind = abs(piece)

    if kind == PAWN:
        fwd = -color
        one = shift(pos, fwd, 0) & empty
        start_rank = 6 if color == WHITE else 1
        two = shift(one, fwd, 0) & empty if r == start_rank else np.zeros_like(pos)
        targets = enemy.copy()
        if ep_square is not None:
            targets[ep_square] = True
        caps = step_attacks(pos, [(fwd, -1), (fwd, 1)]) & targets
        return one | two | caps
    if kind == KNIGHT:
        return step_attacks(pos, KNIGHT_DIRS) & (empty | enemy)
    if kind == KING:
        return step_attacks(pos, KING_DIRS) & (empty | enemy)
    dirs = {BISHOP: DIAG_DIRS, ROOK: ORTHO_DIRS, QUEEN: KING_DIRS}[kind]
    return ray_attacks(pos, empty, dirs) & (empty | enemy)


# ---------------------------------------------------------------------------
# Coups = matrices 64×64 : b' = T @ b (plateau aplati, ordre C : idx = 8r + c)


def sq(r, c):
    return 8 * r + c


def move_matrix(f, t, promo=None, extra_clear=None, also=None):
    """T = I − E_ff − E_tt + k·E_tf : vide f, écrase t, y dépose k× la pièce de f.

    promo : coefficient k (ex. QUEEN pour promotion dame, la valeur du pion étant ±1).
    extra_clear : case supplémentaire vidée (prise en passant).
    also : second couple (f2, t2) appliqué dans la même matrice (roque : la tour).
    """
    T = np.eye(64, dtype=np.int64)
    T[f, f] = 0
    T[t, t] = 0
    T[t, f] = promo if promo is not None else 1
    if extra_clear is not None:
        T[extra_clear, extra_clear] = 0
    if also is not None:
        f2, t2 = also
        T[f2, f2] = 0
        T[t2, t2] = 0
        T[t2, f2] = 1
    return T


def reverse_matrix(T):
    """Matrice de retour R : R @ (T @ b) = b sur les cases non écrasées par T.

    T étant singulière dès qu'il y a prise (l'information de la pièce capturée
    est détruite), l'inversion exacte est affine : b = R @ b' + correction,
    où la correction restaure les pièces écrasées (voir invert_game).
    """
    R = np.eye(64, dtype=np.float64)
    D = T - np.eye(64, dtype=np.int64)
    for t_idx, f_idx in zip(*np.nonzero(D)):
        if t_idx != f_idx:  # entrée +k en (t, f) : la pièce est allée f → t
            k = D[t_idx, f_idx]
            R[t_idx, t_idx] = 0
            R[f_idx, f_idx] = 0
            R[f_idx, t_idx] = 1.0 / k
    return R


# ---------------------------------------------------------------------------
# État de partie : plateau matriciel + droits (roque, en passant, trait)


class Game:
    """Partie d'échecs dont toute la mécanique de plateau est matricielle.

    Les droits (roque, en passant) sont des scalaires d'appoint : ils
    paramètrent quelles matrices T sont proposées, mais chaque application
    de coup reste b' = T @ b.
    """

    START = np.array([
        [-4, -2, -3, -5, -6, -3, -2, -4],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [4, 2, 3, 5, 6, 3, 2, 4],
    ], dtype=np.int64)

    def __init__(self, board=None, turn=WHITE, castling="KQkq", ep_square=None):
        self.board = Game.START.copy() if board is None else np.asarray(board, dtype=np.int64)
        self.turn = turn
        self.castling = castling  # sous-ensemble de "KQkq"
        self.ep_square = ep_square  # (r, c) ou None

    # --- génération ---

    def pseudo_legal(self):
        """Liste de coups (f, t, promo, extra_clear, also) pour le trait."""
        moves = []
        color = self.turn
        for r, c in zip(*np.nonzero(self.board * color > 0)):
            mask = piece_moves(self.board, r, c, self.ep_square)
            piece = abs(self.board[r, c])
            for tr, tc in zip(*np.nonzero(mask)):
                f, t = sq(r, c), sq(tr, tc)
                ec = None
                if piece == PAWN and self.ep_square == (tr, tc):
                    ec = sq(r, tc)  # pion capturé en passant : même ligne que f
                last = 0 if color == WHITE else 7
                if piece == PAWN and tr == last:
                    for k in (QUEEN, ROOK, BISHOP, KNIGHT):
                        moves.append((f, t, k, ec, None))  # pion = ±1 ⇒ k·(±1) = pièce promue signée
                else:
                    moves.append((f, t, None, ec, None))
        moves.extend(self._castle_moves())
        return moves

    def _castle_moves(self):
        color = self.turn
        row = 7 if color == WHITE else 0
        rights = ("K", "Q") if color == WHITE else ("k", "q")
        out = []
        if self.board[row, 4] != color * KING:
            return out
        attacked = attacked_by(self.board, -color)
        if attacked[row, 4]:
            return out
        # petit roque : cases f, g vides et non attaquées (e déjà vérifié)
        if rights[0] in self.castling and self.board[row, 7] == color * ROOK \
                and not self.board[row, 5] and not self.board[row, 6] \
                and not attacked[row, 5] and not attacked[row, 6]:
            out.append((sq(row, 4), sq(row, 6), None, None, (sq(row, 7), sq(row, 5))))
        # grand roque : b, c, d vides ; c, d non attaquées
        if rights[1] in self.castling and self.board[row, 0] == color * ROOK \
                and not self.board[row, 1] and not self.board[row, 2] and not self.board[row, 3] \
                and not attacked[row, 2] and not attacked[row, 3]:
            out.append((sq(row, 4), sq(row, 2), None, None, (sq(row, 0), sq(row, 3))))
        return out

    def legal_moves(self):
        """Filtre matriciel parallèle : tous les plateaux candidats sont empilés
        en un tenseur (N, 8, 8) et l'échec est testé en un seul calcul batch."""
        pseudo = self.pseudo_legal()
        if not pseudo:
            return []
        flat = self.board.reshape(64)
        cand = np.stack([(move_matrix(*m[:2], promo=m[2], extra_clear=m[3], also=m[4]) @ flat)
                         .reshape(8, 8) for m in pseudo])
        ok = ~in_check(cand, self.turn)
        return [m for m, keep in zip(pseudo, ok) if keep]

    # --- application ---

    def push(self, move):
        """Applique b' = T @ b et met à jour les droits. Retourne T."""
        f, t, promo, ec, also = move
        T = move_matrix(f, t, promo=promo, extra_clear=ec, also=also)
        flat = self.board.reshape(64)
        piece, target = flat[f], flat[t]
        self.board = (T @ flat).reshape(8, 8)
        fr, fc, tr, tc = f // 8, f % 8, t // 8, t % 8
        # droit en passant
        self.ep_square = ((fr + tr) // 2, fc) if abs(piece) == PAWN and abs(tr - fr) == 2 else None
        # droits de roque perdus si roi/tour bouge ou tour capturée
        lost = ""
        for square, right in [((7, 4), "KQ"), ((0, 4), "kq"), ((7, 7), "K"),
                              ((7, 0), "Q"), ((0, 7), "k"), ((0, 0), "q")]:
            if (fr, fc) == square or (tr, tc) == square:
                lost += right
        self.castling = "".join(ch for ch in self.castling if ch not in lost)
        self.turn = -self.turn
        return T

    # --- verdicts ---

    def status(self):
        """'ongoing' | 'checkmate' | 'stalemate' (pour le trait)."""
        if self.legal_moves():
            return "ongoing"
        return "checkmate" if in_check(self.board, self.turn) else "stalemate"


# ---------------------------------------------------------------------------
# Inversion : reconstruire b0 depuis b_n, les matrices T et la pile des prises


def invert_game(final_board, matrices, captures):
    """Rejoue la partie à l'envers : b_{k-1} = R_k @ b_k + correction_k.

    `captures` : liste (une entrée par coup) de [(idx_case, valeur_écrasée), ...]
    — l'information détruite par T (prises, cases vidées) qu'aucune matrice
    inverse ne peut restituer, T étant singulière.
    """
    b = final_board.reshape(64).astype(np.float64)
    for T, caps in zip(reversed(matrices), reversed(captures)):
        b = reverse_matrix(T) @ b
        for idx, val in caps:
            b[idx] = val
    return np.rint(b).astype(np.int64).reshape(8, 8)


def transition_matrix(b1, b2, game1):
    """Vérifie par calcul vectoriel rapide si b1 → b2 est un coup légal.

    Diff Δ = b2 − b1 : un coup touche ≤ 4 cases. On ne compare Δ qu'aux
    coups légaux dont le support correspond — O(64) + une poignée de produits.
    Retourne la matrice T du coup, ou None si la transition est illégale.
    """
    d = (np.asarray(b2) - np.asarray(b1)).reshape(64)
    changed = set(np.nonzero(d)[0])
    if not 2 <= len(changed) <= 4:
        return None
    flat = np.asarray(b1).reshape(64)
    for m in game1.legal_moves():
        T = move_matrix(*m[:2], promo=m[2], extra_clear=m[3], also=m[4])
        if np.array_equal(T @ flat, flat + d):
            return T
    return None


# --- interop notation (uci ↔ interne), pour les tests contre python-chess ---

def to_uci(move):
    f, t, promo, _, _ = move
    s = "abcdefgh"[f % 8] + str(8 - f // 8) + "abcdefgh"[t % 8] + str(8 - t // 8)
    if promo is not None:
        s += {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}[abs(promo)]
    return s
