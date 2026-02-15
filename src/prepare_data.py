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
import os

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
    """Encode un chess.Board en vecteur one-hot (832,) uint8.
    Version optimisée avec piece_map() au lieu de 64× piece_at()."""
    x = np.zeros(832, dtype=np.uint8)
    # Cases vides : catégorie 0
    for sq in range(64):
        x[sq * 13] = 1
    # Pièces présentes : écraser la catégorie 0
    for sq, piece in board.piece_map().items():
        idx = PIECE_TO_IDX[(piece.piece_type, piece.color)]
        x[sq * 13] = 0
        x[sq * 13 + idx] = 1
    return x


def move_to_token(move, move_to_idx):
    """Convertit un chess.Move en index de token."""
    key = (move.from_square, move.to_square, move.promotion)
    return move_to_idx.get(key, None)


# ---------------------------------------------------------------------------
# 3. Parsing PGN et extraction des exemples
# ---------------------------------------------------------------------------

def _parse_game(game_text, move_to_idx):
    """Parse une partie PGN (texte) et retourne (X_list, y_list, n_skipped)."""
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
    except Exception:
        return [], [], 0

    if game is None:
        return [], [], 0

    variant = game.headers.get("Variant", "Standard")
    if variant not in ("Standard", "standard", ""):
        return [], [], 0

    board = game.board()
    X_list = []
    y_list = []
    n_skipped = 0

    for move in game.mainline_moves():
        x = board_to_vector(board)
        token = move_to_token(move, move_to_idx)
        if token is not None:
            X_list.append(x)
            y_list.append(token)
        else:
            n_skipped += 1
        board.push(move)

    return X_list, y_list, n_skipped


def _split_pgn(pgn_path):
    """Découpe un fichier PGN en textes de parties individuelles."""
    import sys
    file_size = os.path.getsize(pgn_path)
    games = []
    current = []
    bytes_read = 0
    last_pct = -1
    with open(pgn_path, "r") as f:
        for line in f:
            bytes_read += len(line)
            if line.startswith("[Event ") and current:
                games.append("".join(current))
                current = []
            current.append(line)
            pct = bytes_read * 100 // file_size
            if pct >= last_pct + 5:
                last_pct = pct
                print(f"\r  Lecture du PGN... {pct}% ({len(games)} parties)", end="", flush=True)
    if current:
        games.append("".join(current))
    print(f"\r  Lecture du PGN... 100% ({len(games)} parties)", flush=True)
    return games


def _parse_batch(args):
    """Parse un batch de parties PGN. Fonction module-level pour ProcessPoolExecutor."""
    batch, move_to_idx = args
    bx, by, bs = [], [], 0
    for text in batch:
        xl, yl, sk = _parse_game(text, move_to_idx)
        bx.extend(xl)
        by.extend(yl)
        bs += sk
    return bx, by, len(batch), bs


def parse_pgn_file(pgn_path, move_to_idx):
    """Parse un fichier PGN et retourne (X, y) pour l'entraînement.
    Utilise le parallélisme CPU pour accélérer le parsing."""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t0 = _time.time()
    print(f"  Découpage du PGN...", end=" ", flush=True)
    game_texts = _split_pgn(pgn_path)
    n_total = len(game_texts)
    print(f"{n_total} parties trouvées ({_time.time()-t0:.1f}s)")

    n_workers = max(1, min(os.cpu_count() or 1, 8))
    print(f"  Parsing avec {n_workers} workers...", flush=True)

    X_list = []
    y_list = []
    n_games = 0
    n_skipped_moves = 0

    # Traitement par batch pour réduire l'overhead IPC
    batch_size = max(100, n_total // (n_workers * 4))

    batches = [game_texts[i:i+batch_size] for i in range(0, n_total, batch_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_parse_batch, (b, move_to_idx)): i for i, b in enumerate(batches)}
        done = 0
        for future in as_completed(futures):
            bx, by, ng, sk = future.result()
            X_list.extend(bx)
            y_list.extend(by)
            n_games += ng
            n_skipped_moves += sk
            done += 1
            elapsed = _time.time() - t0
            pct = done / len(batches) * 100
            print(f"  [{done}/{len(batches)}] {pct:.0f}% │ "
                  f"{n_games} parties │ {len(y_list):,} exemples │ "
                  f"{elapsed:.0f}s", flush=True)
    X = np.array(X_list, dtype=np.uint8)
    y = np.array(y_list, dtype=np.int32)

    return X, y, n_games, n_skipped_moves


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def run(pgn_path, out_path):
    """Convertit un PGN en données d'entraînement .npz. Retourne le chemin."""
    print(f"Construction du dictionnaire de coups...")
    move_to_idx, idx_to_move = build_move_dict()
    print(f"  → {len(move_to_idx)} tokens")

    print(f"Parsing de {pgn_path}...")
    X, y, n_games, n_skipped = parse_pgn_file(pgn_path, move_to_idx)
    print(f"  → {n_games} parties, {len(y)} exemples, {n_skipped} coups ignorés")
    print(f"  → X shape: {X.shape}, y shape: {y.shape}")

    move_tokens = np.array(
        [(f, t, p if p else 0) for (f, t, p) in [idx_to_move[i] for i in range(len(idx_to_move))]],
        dtype=np.int32
    )

    np.savez_compressed(out_path, X=X, y=y, move_tokens=move_tokens)
    print(f"  → Sauvegardé dans {out_path}")

    file_size = os.path.getsize(out_path)
    print(f"  → Taille : {file_size / (1024**2):.1f} Mo")
    return out_path


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.pgn> <output.npz>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    import os
    main()
