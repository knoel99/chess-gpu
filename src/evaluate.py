#!/usr/bin/env python3
"""
Fait jouer le modèle linéaire contre Stockfish sur des parties complètes.

Stockfish doit être installé :
    apt install stockfish          # Linux / Colab
    brew install stockfish         # macOS

Usage:
    python evaluate.py data/magnuscarlsen_2025_01_model.npz [--games 10] [--stockfish-elo 800]
"""

import sys
import os
import argparse
import numpy as np
import chess
import chess.engine
import chess.pgn

# Auto-détection GPU
try:
    import cupy as xp
    GPU = True
except (ImportError, Exception):
    import numpy as xp
    GPU = False


# ── Modèle ──────────────────────────────────────────────────────────────────

def board_to_vector(board):
    """Encode le plateau en vecteur one-hot (832 features)."""
    PIECE_MAP = {
        (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
        (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
        (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
        (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8,
        (chess.BISHOP, chess.BLACK): 9, (chess.ROOK, chess.BLACK): 10,
        (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
    }
    vec = np.zeros(832, dtype=np.float32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            idx = sq * 13  # case vide = catégorie 0
        else:
            cat = PIECE_MAP[(piece.piece_type, piece.color)]
            idx = sq * 13 + cat
        vec[idx] = 1.0
    return vec


def load_model(model_path):
    """Charge le modèle 2 couches (W1, b1, W2, b2, move_tokens)."""
    data = np.load(model_path)
    W1 = xp.array(data["W1"])
    b1 = xp.array(data["b1"])
    W2 = xp.array(data["W2"])
    b2 = xp.array(data["b2"])
    move_tokens = data["move_tokens"]
    return W1, b1, W2, b2, move_tokens


def build_token_to_move(move_tokens):
    """Construit le mapping index → chess.Move."""
    moves = {}
    for i, (f, t, p) in enumerate(move_tokens):
        promo = int(p) if int(p) != 0 else None
        moves[i] = chess.Move(int(f), int(t), promotion=promo)
    return moves


def get_top_moves(board, W1, b1, W2, b2, token_to_move, k=10):
    """Retourne les k meilleurs coups légaux avec leurs probabilités."""
    x = board_to_vector(board)
    # Utiliser le même type que W1 (numpy ou cupy)
    _xp = type(W1)
    if hasattr(W1, '__array_namespace__') or 'cupy' in str(type(W1)):
        import cupy as _xp
    else:
        _xp = np
    x_arr = _xp.array(x.reshape(1, -1))

    z1 = x_arr @ W1.T + b1
    a1 = _xp.maximum(z1, 0)
    logits = a1 @ W2.T + b2
    logits = logits - logits.max()
    probs = _xp.exp(logits)
    probs = probs / probs.sum()

    if hasattr(probs, 'get'):
        probs = probs.get()
    probs = probs.flatten()

    legal_moves = set(board.legal_moves)
    sorted_indices = np.argsort(probs)[::-1]

    top = []
    for idx in sorted_indices:
        move = token_to_move.get(int(idx))
        if move and move in legal_moves:
            top.append((move, float(probs[idx])))
            if len(top) >= k:
                break
    return top


# Valeur matérielle des pièces
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def evaluate_material(board):
    """Score matériel du point de vue des blancs."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score


def search(board, W1, b1, W2, b2, token_to_move, model_color,
           depth, deadline, top_k=5):
    """
    Recherche arborescente avec le réseau de neurones.
    Combine la probabilité du réseau et l'évaluation matérielle.
    Retourne (score, meilleur_coup).
    """
    import time as _time

    if _time.time() > deadline:
        return evaluate_position(board, model_color), None

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return (1000 if model_color == chess.WHITE else -1000), None
        elif result == "0-1":
            return (-1000 if model_color == chess.WHITE else 1000), None
        else:
            return 0, None

    if depth == 0:
        return evaluate_position(board, model_color), None

    candidates = get_top_moves(board, W1, b1, W2, b2, token_to_move, k=top_k)

    if not candidates:
        return evaluate_position(board, model_color), None

    is_model_turn = (board.turn == model_color)
    best_score = -99999 if is_model_turn else 99999
    best_move = candidates[0][0]

    for move, prob in candidates:
        if _time.time() > deadline:
            break

        board.push(move)
        child_score, _ = search(
            board, W1, b1, W2, b2, token_to_move, model_color,
            depth - 1, deadline, top_k=max(3, top_k - 1)
        )
        # Bonus pour les coups à haute probabilité du réseau
        child_score += prob * 2 if is_model_turn else -prob * 2
        board.pop()

        if is_model_turn:
            if child_score > best_score:
                best_score = child_score
                best_move = move
        else:
            if child_score < best_score:
                best_score = child_score
                best_move = move

    return best_score, best_move


def evaluate_position(board, model_color):
    """Évaluation heuristique : matériel + mobilité."""
    mat = evaluate_material(board)
    if model_color == chess.BLACK:
        mat = -mat
    # Bonus mobilité
    mobility = board.legal_moves.count() * 0.05
    if board.turn != model_color:
        mobility = -mobility
    return mat + mobility


def predict_move(board, W1, b1, W2, b2, token_to_move, think_time=0):
    """Prédit le meilleur coup. Si think_time > 0, utilise la recherche arborescente."""
    if think_time <= 0:
        # Mode instantané (comme avant)
        top = get_top_moves(board, W1, b1, W2, b2, token_to_move, k=1)
        if top:
            return top[0][0], top[0][1]
        import random
        return random.choice(list(board.legal_moves)), 0.0

    import time as _time
    deadline = _time.time() + think_time

    # Approfondissement itératif
    best_move = None
    best_score = -99999
    model_color = board.turn
    depth_reached = 0

    for depth in range(1, 20):
        if _time.time() > deadline:
            break

        score, move = search(
            board, W1, b1, W2, b2, token_to_move, model_color,
            depth, deadline, top_k=7
        )

        if move is not None:
            best_move = move
            best_score = score
            depth_reached = depth

    if best_move is None:
        top = get_top_moves(board, W1, b1, W2, b2, token_to_move, k=1)
        best_move = top[0][0] if top else list(board.legal_moves)[0]

    return best_move, depth_reached


# ── Partie ──────────────────────────────────────────────────────────────────

def play_game(W1, b1, W2, b2, token_to_move, engine, model_color,
              sf_time=0.1, think_time=0):
    """Joue une partie complète. Retourne (score, pgn_string)."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "Réseau de neurones" if model_color == chess.WHITE else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_color == chess.WHITE else "Réseau de neurones"
    node = game
    move_count = 0

    while not board.is_game_over() and move_count < 200:
        if board.turn == model_color:
            move, info = predict_move(board, W1, b1, W2, b2, token_to_move,
                                      think_time=think_time)
            node = node.add_variation(move)
            if think_time > 0:
                node.comment = f"depth={info}"
            else:
                node.comment = f"p={info:.3f}"
        else:
            result = engine.play(board, chess.engine.Limit(time=sf_time))
            move = result.move
            node = node.add_variation(move)

        board.push(move)
        move_count += 1

    result = board.result()
    game.headers["Result"] = result

    if result == "1-0":
        score = 1.0 if model_color == chess.WHITE else 0.0
    elif result == "0-1":
        score = 0.0 if model_color == chess.WHITE else 1.0
    elif result == "1/2-1/2":
        score = 0.5
    else:
        score = 0.5

    return score, str(game)


# ── Main ────────────────────────────────────────────────────────────────────

def generate_html(pgns, results, stockfish_elo, out_path):
    """Génère un fichier HTML interactif avec échiquier navigable."""
    import json
    pgns_json = json.dumps(pgns)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>♟ Réseau de neurones vs Stockfish (Elo {stockfish_elo})</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
         display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  h1 {{ margin-bottom: 10px; }}
  .score {{ font-size: 1.2em; margin-bottom: 20px; color: #aaa; }}
  .game-select {{ margin-bottom: 15px; }}
  select, button {{ font-size: 16px; padding: 8px 16px; border-radius: 6px;
                    border: none; cursor: pointer; margin: 0 4px; }}
  select {{ background: #16213e; color: #eee; }}
  button {{ background: #0f3460; color: #eee; }}
  button:hover {{ background: #533483; }}
  button:disabled {{ opacity: 0.3; cursor: default; }}
  #board {{ font-size: 0; margin: 10px 0; }}
  .row {{ display: flex; }}
  .sq {{ width: 60px; height: 60px; display: flex; align-items: center;
         justify-content: center; font-size: 44px; user-select: none;
         text-shadow: none; }}
  .light {{ background: #f0d9b5; }}
  .dark {{ background: #b58863; }}
  .white-piece {{ color: #fff; text-shadow: 0 0 2px #000, 0 0 2px #000, 1px 1px 1px #000; }}
  .black-piece {{ color: #000; text-shadow: 0 0 2px rgba(255,255,255,0.3); }}
  .highlight {{ box-shadow: inset 0 0 0 3px #f5c542; }}
  .players {{ display: flex; justify-content: space-between; width: 480px;
              margin-bottom: 5px; font-size: 14px; }}
  .player-bar {{ display: flex; align-items: center; gap: 8px; padding: 6px 12px;
                 border-radius: 6px; }}
  .player-bar.top {{ background: #222; }}
  .player-bar.bottom {{ background: #ddd; }}
  .player-dot {{ width: 16px; height: 16px; border-radius: 50%; border: 2px solid #888; }}
  .player-dot.white {{ background: #fff; }}
  .player-dot.black {{ background: #000; }}
  .player-name {{ font-weight: bold; }}
  .player-bar.top .player-name {{ color: #ccc; }}
  .player-bar.bottom .player-name {{ color: #333; }}
  .player-tag {{ font-size: 11px; padding: 2px 6px; border-radius: 3px; font-weight: normal; }}
  .player-tag.model {{ background: #533483; color: #fff; }}
  .player-tag.stockfish {{ background: #0f3460; color: #fff; }}
  .player-bar .turn-indicator {{ font-size: 11px; color: #f5c542; font-weight: bold;
                                  margin-left: auto; }}
  .board-container {{ border: 2px solid #555; border-radius: 4px; overflow: hidden; }}
  .info {{ margin-top: 15px; font-size: 14px; color: #aaa; text-align: center; }}
  .move-list {{ max-width: 500px; margin-top: 10px; font-family: monospace;
                font-size: 13px; color: #ccc; line-height: 1.6; word-wrap: break-word;
                text-align: center; }}
  .move-list .current {{ color: #f5c542; font-weight: bold; }}
</style>
</head>
<body>
<h1>♟ Réseau de neurones vs Stockfish</h1>
<div class="score">{results['win']}W - {results['draw']}D - {results['loss']}L │ Elo Stockfish: ~{stockfish_elo}</div>

<div class="game-select">
  <select id="gameSelect" onchange="loadGame(this.value)"></select>
</div>
<div id="topPlayer" class="player-bar top" style="width:480px">
  <div class="player-dot black"></div>
  <span class="player-name" id="blackName">Noirs</span>
  <span class="player-tag" id="blackTag"></span>
  <span class="turn-indicator" id="blackTurn"></span>
</div>
<div class="board-container"><div id="board"></div></div>
<div id="bottomPlayer" class="player-bar bottom" style="width:480px">
  <div class="player-dot white"></div>
  <span class="player-name" id="whiteName">Blancs</span>
  <span class="player-tag" id="whiteTag"></span>
  <span class="turn-indicator" id="whiteTurn"></span>
</div>
<div>
  <button onclick="goTo(0)">⏮</button>
  <button onclick="step(-1)">◀</button>
  <button id="playBtn" onclick="togglePlay()">▶ Play</button>
  <button onclick="step(1)">▶</button>
  <button onclick="goTo(positions.length-1)">⏭</button>
</div>
<div class="info" id="info"></div>
<div class="move-list" id="moveList"></div>

<script>
const PIECES = {{
  'P':'♟','N':'♞','B':'♝','R':'♜','Q':'♛','K':'♚',
  'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'
}};
const WHITE_PIECES = new Set(['P','N','B','R','Q','K']);
const pgns = {pgns_json};
let positions = [], moves = [], currentPos = 0, playInterval = null, gameHeaders = {{}};

function fenToBoard(fen) {{
  const rows = fen.split(' ')[0].split('/');
  const board = [];
  for (const row of rows) {{
    const r = [];
    for (const c of row) {{
      if (c >= '1' && c <= '8') for (let i = 0; i < +c; i++) r.push('');
      else r.push(c);
    }}
    board.push(r);
  }}
  return board;
}}

function loadGame(idx) {{
  const pgn = pgns[idx];

  positions = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'];
  moves = [];

  if (typeof Chess !== 'undefined') {{
    const game = new Chess();
    game.load_pgn(pgn);
    const history = game.history();
    moves = history;

    // Rejouer pour capturer les FEN
    game.reset();
    for (const m of history) {{
      game.move(m);
      positions.push(game.fen());
    }}
  }}

  currentPos = 0;
  render();
  renderMoveList();

  // Headers
  gameHeaders = {{}};
  pgn.split('\\n').forEach(l => {{
    const m = l.match(/^\\[(\w+)\\s+"(.+)"\\]/);
    if (m) gameHeaders[m[1]] = m[2];
  }});
  const wName = gameHeaders.White || 'Blancs';
  const bName = gameHeaders.Black || 'Noirs';
  const isModelWhite = wName.includes('seau');
  document.getElementById('whiteName').textContent = wName;
  document.getElementById('blackName').textContent = bName;
  document.getElementById('whiteTag').textContent = isModelWhite ? '🤖 réseau de neurones' : '♟ stockfish';
  document.getElementById('whiteTag').className = 'player-tag ' + (isModelWhite ? 'model' : 'stockfish');
  document.getElementById('blackTag').textContent = isModelWhite ? '♟ stockfish' : '🤖 réseau de neurones';
  document.getElementById('blackTag').className = 'player-tag ' + (isModelWhite ? 'stockfish' : 'model');
  document.getElementById('info').textContent = `Résultat : ${{gameHeaders.Result || '?'}}`;
}}

function render() {{
  const fen = positions[currentPos] || positions[0];
  const board = fenToBoard(fen);
  const turn = (fen.split(' ')[1] || 'w') === 'w' ? 'white' : 'black';
  let html = '';

  for (let r = 0; r < 8; r++) {{
    html += '<div class="row">';
    for (let c = 0; c < 8; c++) {{
      const light = (r + c) % 2 === 0;
      const piece = board[r][c];
      const colorCls = piece ? (WHITE_PIECES.has(piece) ? 'white-piece' : 'black-piece') : '';
      html += `<div class="sq ${{light?'light':'dark'}} ${{colorCls}}">${{piece ? PIECES[piece] || '' : ''}}</div>`;
    }}
    html += '</div>';
  }}
  document.getElementById('board').innerHTML = html;

  // Mettre à jour qui joue
  document.getElementById('whiteTurn').textContent = turn === 'white' ? '◄ au trait' : '';
  document.getElementById('blackTurn').textContent = turn === 'black' ? '◄ au trait' : '';

  renderMoveList();
}}

function renderMoveList() {{
  let html = '';
  for (let i = 0; i < moves.length; i++) {{
    if (i % 2 === 0) html += `${{Math.floor(i/2)+1}}. `;
    const cls = i === currentPos - 1 ? 'current' : '';
    html += `<span class="${{cls}}" style="cursor:pointer" onclick="goTo(${{i+1}})">${{moves[i]}}</span> `;
  }}
  document.getElementById('moveList').innerHTML = html;
}}

function step(dir) {{
  const next = currentPos + dir;
  if (next >= 0 && next < positions.length) {{ currentPos = next; render(); }}
}}

function goTo(idx) {{
  if (idx >= 0 && idx < positions.length) {{ currentPos = idx; render(); }}
}}

function togglePlay() {{
  if (playInterval) {{
    clearInterval(playInterval);
    playInterval = null;
    document.getElementById('playBtn').textContent = '▶ Play';
  }} else {{
    document.getElementById('playBtn').textContent = '⏸ Pause';
    playInterval = setInterval(() => {{
      if (currentPos >= positions.length - 1) {{ togglePlay(); return; }}
      step(1);
    }}, 800);
  }}
}}

// Populate game selector
pgns.forEach((pgn, i) => {{
  const opt = document.createElement('option');
  const headers = {{}};
  pgn.split('\\n').forEach(l => {{
    const m = l.match(/^\\[(\w+)\\s+"(.+)"\\]/);
    if (m) headers[m[1]] = m[2];
  }});
  opt.value = i;
  opt.textContent = `Partie ${{i+1}}: ${{headers.White||'?'}} vs ${{headers.Black||'?'}} (${{headers.Result||'?'}})`;
  document.getElementById('gameSelect').appendChild(opt);
}});

// Load chess.js from CDN for move replay
const s = document.createElement('script');
s.src = 'https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js';
s.onload = () => loadGame(0);
document.head.appendChild(s);
</script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n  🌐 Visualisation sauvegardée : {out_path}")


def find_stockfish():
    """Cherche le binaire Stockfish."""
    import shutil
    path = shutil.which("stockfish")
    if path:
        return path
    candidates = [
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _play_game_worker(args):
    """Worker pour jouer une partie en parallèle (CPU only)."""
    W1, b1, W2, b2, token_to_move, sf_path, sf_elo, model_color, think_time = args

    # Chaque worker utilise NumPy (pas CuPy — pas de GPU partagé entre processes)
    import numpy as _xp

    # Convertir les poids en arrays locaux
    W1_l = _xp.array(W1)
    b1_l = _xp.array(b1)
    W2_l = _xp.array(W2)
    b2_l = _xp.array(b2)

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": max(sf_elo, 1350)})

    score, pgn = play_game(W1_l, b1_l, W2_l, b2_l, token_to_move, engine,
                           model_color, think_time=think_time)
    engine.quit()
    return score, pgn, model_color


def _auto_workers():
    """Détecte le nombre optimal de workers (CPU cores, min 1, max 8)."""
    import os
    cores = os.cpu_count() or 1
    return max(1, min(cores, 8))


def run(model_path, n_games=10, stockfish_elo=800, think_time=0, n_workers=0):
    """Lance l'évaluation contre Stockfish."""
    mode = "instantané" if think_time <= 0 else f"recherche {think_time}s/coup"
    if n_workers <= 0:
        n_workers = _auto_workers()
    parallel = n_workers > 1
    print(f"\n{'='*60}")
    print(f"  ♟  Évaluation : Réseau vs Stockfish (Elo ~{stockfish_elo})")
    print(f"{'='*60}")

    # Charger modèle
    print(f"  Modèle    : {model_path}")
    W1, b1, W2, b2, move_tokens = load_model(model_path)
    token_to_move = build_token_to_move(move_tokens)
    print(f"  Tokens    : {len(token_to_move)} coups")
    print(f"  Arch.     : {W1.shape[1]} → {W1.shape[0]} (ReLU) → {W2.shape[0]} (softmax)")
    print(f"  Parties   : {n_games}")
    print(f"  Mode      : {mode}")
    print(f"  Workers   : {n_workers}" + (" (parallèle)" if parallel else " (séquentiel)"))
    print(f"  Backend   : {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}" +
          (" → workers CPU" if parallel and GPU else ""))

    # Trouver Stockfish
    sf_path = find_stockfish()
    if sf_path is None:
        print("\n❌ Stockfish non trouvé. Installe-le :")
        print("   apt install stockfish    # Linux/Colab")
        print("   brew install stockfish   # macOS")
        sys.exit(1)
    print(f"  Stockfish : {sf_path}")
    print(f"{'='*60}\n")

    # Ramener les poids sur CPU pour les workers
    W1_cpu = W1.get() if GPU else W1
    b1_cpu = b1.get() if GPU else b1
    W2_cpu = W2.get() if GPU else W2
    b2_cpu = b2.get() if GPU else b2

    results = {"win": 0, "draw": 0, "loss": 0}
    scores = []
    pgns = []

    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for i in range(n_games):
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            tasks.append((W1_cpu, b1_cpu, W2_cpu, b2_cpu, token_to_move,
                         sf_path, stockfish_elo, model_color, think_time))

        game_results = [None] * n_games
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_play_game_worker, t): i
                      for i, t in enumerate(tasks)}
            done_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                score, pgn, model_color = future.result()
                game_results[idx] = (score, pgn, model_color)
                done_count += 1

                if score == 1.0:
                    results["win"] += 1
                    symbol = "✅"
                elif score == 0.0:
                    results["loss"] += 1
                    symbol = "❌"
                else:
                    results["draw"] += 1
                    symbol = "🤝"

                color_str = "Blancs" if model_color == chess.WHITE else "Noirs"
                win_rate = (results["win"] + results["draw"] * 0.5) / done_count * 100
                print(f"  Partie {done_count:2d}/{n_games} │ {color_str:7s} │ {symbol} │ "
                      f"Score: {results['win']}W-{results['draw']}D-{results['loss']}L │ "
                      f"Win rate: {win_rate:.0f}%")

        for score, pgn, _ in game_results:
            scores.append(score)
            pgns.append(pgn)
    else:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": max(stockfish_elo, 1350)})

        for i in range(n_games):
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_str = "Blancs" if model_color == chess.WHITE else "Noirs"

            score, pgn = play_game(W1, b1, W2, b2, token_to_move, engine,
                                   model_color, think_time=think_time)
            scores.append(score)
            pgns.append(pgn)

            if score == 1.0:
                results["win"] += 1
                symbol = "✅"
            elif score == 0.0:
                results["loss"] += 1
                symbol = "❌"
            else:
                results["draw"] += 1
                symbol = "🤝"

            total = i + 1
            win_rate = sum(scores) / total * 100
            print(f"  Partie {total:2d}/{n_games} │ {color_str:7s} │ {symbol} │ "
                  f"Score: {results['win']}W-{results['draw']}D-{results['loss']}L │ "
                  f"Win rate: {win_rate:.0f}%")

        engine.quit()

    # Générer le HTML interactif
    html_path = os.path.splitext(model_path)[0] + "_games.html"
    generate_html(pgns, results, stockfish_elo, html_path)

    # Estimation Elo (formule FIDE inversée)
    import math
    total_score = sum(scores)
    win_rate = total_score / n_games
    if win_rate <= 0:
        elo_diff = -400
    elif win_rate >= 1:
        elo_diff = 400
    else:
        elo_diff = -400 * math.log10(1.0 / win_rate - 1)
    estimated_elo = round(stockfish_elo + elo_diff)

    # Résumé
    print(f"\n{'='*60}")
    print(f"  Résultat final : {results['win']}W - {results['draw']}D - {results['loss']}L")
    print(f"  Score          : {total_score:.1f}/{n_games} ({win_rate*100:.0f}%)")
    print(f"  Elo Stockfish  : ~{stockfish_elo}")
    print(f"  Elo estimé     : ~{estimated_elo}")
    print(f"  Mode           : {mode}")
    print(f"  Workers        : {n_workers}")
    print(f"  Visualisation  : {html_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation réseau vs Stockfish")
    parser.add_argument("model", help="Chemin vers le modèle .npz")
    parser.add_argument("--games", type=int, default=10, help="Nombre de parties (défaut: 10)")
    parser.add_argument("--stockfish-elo", type=int, default=1350, help="Elo de Stockfish (défaut: 1350, min: 1350)")
    parser.add_argument("--think-time", type=float, default=0, help="Temps de réflexion en secondes par coup (défaut: 0 = instantané)")
    parser.add_argument("--workers", type=int, default=0, help="Nombre de workers (défaut: 0 = auto-détection CPU cores)")
    args = parser.parse_args()
    run(args.model, args.games, args.stockfish_elo, args.think_time, args.workers)


if __name__ == "__main__":
    main()
