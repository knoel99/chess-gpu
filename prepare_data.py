#!/usr/bin/env python3
"""
Convertit des fichiers PGN Chess.com en données d'entraînement (NumPy .npz).

Chaque position d'une partie génère un exemple :
  - X : vecteur one-hot de 832 features (64 cases × 13 états)
  - y : index du coup joué dans le dictionnaire de 1968 tokens

Usage:
    python prepare_data.py data/magnus_2025_01.pgn data/training_data.npz
"""

import sys
import json
import numpy as np
import chess
import chess.pgn
import io


# ---------------------------------------------------------------------------
# 1. Construction du dictionnaire de coups (tokens)
# ---------------------------------------------------------------------------

def build_move_dict():
    """Énumère tous les coups possibles sur un échiquier 8×8.
    Retourne (move_to_idx, idx_to_move)."""

    moves = []

    for from_sq in range(64):
        from_r, from_c = divmod(from_sq, 8)

        # Glissements (Tour, Fou, Dame) + Roi
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),      # lignes/colonnes
            (1, 1), (1, -1), (-1, 1), (-1, -1),     # diagonales
        ]
        for dr, dc in directions:
            for dist in range(1, 8):
                to_r, to_c = from_r + dr * dist, from_c + dc * dist
                if 0 <= to_r < 8 and 0 <= to_c < 8:
                    moves.append((from_sq, to_r * 8 + to_c, None))

        # Cavalier
        for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
            to_r, to_c = from_r + dr, from_c + dc
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                moves.append((from_sq, to_r * 8 + to_c, None))

        # Pion blanc (avance vers rang 0) — promotions
        if from_r == 1:  # rang 7 en notation chess (avant-dernière rangée blanche vue du haut)
            for dc in [-1, 0, 1]:
                to_c = from_c + dc
                if 0 <= to_c < 8:
                    to_sq = 0 * 8 + to_c
                    if dc == 0:
                        # avance droite
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append((from_sq, to_sq, promo))
                    else:
                        # capture diagonale
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append((from_sq, to_sq, promo))

        # Pion noir (avance vers rang 7) — promotions
        if from_r == 6:
            for dc in [-1, 0, 1]:
                to_c = from_c + dc
                if 0 <= to_c < 8:
                    to_sq = 7 * 8 + to_c
                    if dc == 0:
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append((from_sq, to_sq, promo))
                    else:
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append((from_sq, to_sq, promo))

    # Dédupliquer et indexer
    unique_moves = list(dict.fromkeys(moves))
    move_to_idx = {m: i for i, m in enumerate(unique_moves)}
    idx_to_move = {i: m for m, i in move_to_idx.items()}

    return move_to_idx, idx_to_move


# ---------------------------------------------------------------------------
# 2. Encodage d'une position en vecteur one-hot (832,)
# ---------------------------------------------------------------------------

# Mapping pièce chess.Piece → index one-hot [0..12]
PIECE_TO_IDX = {
    None: 0,
    (chess.PAWN, chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5,
    (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11,
    (chess.KING, chess.BLACK): 12,
}


def board_to_vector(board):
    """Encode un chess.Board en vecteur one-hot (832,) uint8."""
    x = np.zeros(832, dtype=np.uint8)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            idx = 0
        else:
            idx = PIECE_TO_IDX[(piece.piece_type, piece.color)]
        x[sq * 13 + idx] = 1
    return x


def move_to_token(move, move_to_idx):
    """Convertit un chess.Move en index de token."""
    key = (move.from_square, move.to_square, move.promotion)
    return move_to_idx.get(key, None)


# ---------------------------------------------------------------------------
# 3. Parsing PGN et extraction des exemples
# ---------------------------------------------------------------------------

def parse_pgn_file(pgn_path, move_to_idx):
    """Parse un fichier PGN et retourne (X, y) pour l'entraînement."""
    X_list = []
    y_list = []
    n_games = 0
    n_skipped_moves = 0

    with open(pgn_path, "r") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Ignorer les variantes (Chess960, etc.)
            variant = game.headers.get("Variant", "Standard")
            if variant not in ("Standard", "standard", ""):
                continue

            board = game.board()
            n_games += 1

            for move in game.mainline_moves():
                # Encoder la position
                x = board_to_vector(board)

                # Encoder le coup
                token = move_to_token(move, move_to_idx)
                if token is not None:
                    X_list.append(x)
                    y_list.append(token)
                else:
                    n_skipped_moves += 1

                board.push(move)

    X = np.array(X_list, dtype=np.uint8)
    y = np.array(y_list, dtype=np.int32)

    return X, y, n_games, n_skipped_moves


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.pgn> <output.npz>")
        sys.exit(1)

    pgn_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Construction du dictionnaire de coups...")
    move_to_idx, idx_to_move = build_move_dict()
    print(f"  → {len(move_to_idx)} tokens")

    print(f"Parsing de {pgn_path}...")
    X, y, n_games, n_skipped = parse_pgn_file(pgn_path, move_to_idx)
    print(f"  → {n_games} parties, {len(y)} exemples, {n_skipped} coups ignorés")
    print(f"  → X shape: {X.shape}, y shape: {y.shape}")

    # Sauvegarder le dictionnaire de coups + données
    # Convertir le dict en listes pour le stockage
    move_tokens = np.array(
        [(f, t, p if p else 0) for (f, t, p) in [idx_to_move[i] for i in range(len(idx_to_move))]],
        dtype=np.int32
    )

    np.savez_compressed(out_path, X=X, y=y, move_tokens=move_tokens)
    print(f"  → Sauvegardé dans {out_path}")

    # Stats
    file_size = os.path.getsize(out_path)
    print(f"  → Taille : {file_size / (1024**2):.1f} Mo")


if __name__ == "__main__":
    import os
    main()
