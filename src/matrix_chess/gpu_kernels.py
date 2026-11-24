"""Noyau CUDA fusionné pour gpu_engine.expand : un fil GPU par position parente.

Même résultat que le chemin PyTorch (gpu_engine.expand_torch, qui reste la référence et le repli
CPU), mêmes formats d'entrée et de sortie, enfants dans le même ordre (parent, plan, case de départ).
Dans le noyau, le plateau devient des bitboards (bit s = case s) et chaque plan de coups est un
masque 64 bits des cases de départ, déjà légal :
- pièces clouées : seulement le long de l'axe roi-cloueur ; échec simple : arrivée sur la pièce qui
  donne échec ou entre elle et le roi ; échec double : roi seul ;
- roi : arrivée hors des cases attaquées (calculées roi retiré) ; prise en passant : coup joué puis
  roi testé (rare).
Trois passes : comptage par parent (popcount, sans boucle par coup), somme préfixe côté torch, écriture
compactée d'un code par coup, puis un fil par enfant pour le plateau et l'état. Le comptage seul sert
au dernier niveau de perft.

Compilé à la volée par CuPy (nvrtc) : pip install cupy-cuda12x. Sans CuPy, gpu_engine garde le
chemin PyTorch. Suppose des positions légales (un roi par camp, camp qui ne joue pas hors d'échec).
"""

import numpy as np
import torch

from .engine import KING_DIRS, KNIGHT_DIRS

try:
    import cupy as cp
except ImportError:
    cp = None


def _table(name, dirs):
    return f"static const int {name}[8][2] = {{{', '.join('{%d, %d}' % d for d in dirs)}}};"


SOURCE = _table("KD", KING_DIRS) + _table("ND", KNIGHT_DIRS) + r"""
typedef unsigned long long U;
typedef long long I;
typedef signed char S;

// bit s -> bit s + 8·dr + dc, les bits qui sortent du plateau sont perdus
__device__ __forceinline__ U sh(U x, int dr, int dc) {
    U cols = (dc >= 0 ? (0xFFull << dc) & 0xFFull : 0xFFull >> -dc) * 0x0101010101010101ull;
    int s = 8 * dr + dc;
    return (s >= 0 ? x << s : x >> -s) & cols;
}
__device__ __forceinline__ U bit(int s) { return 1ull << s; }
__device__ __forceinline__ int axis(int dr, int dc) { return dc == 0 ? 0 : dr == 0 ? 1 : dr == dc ? 2 : 3; }

// le roi `king` (de couleur color) est-il attaqué par les pièces en[1..6], avec l'occupation occ ?
__device__ bool king_attacked(U king, U occ, const U* en, int color) {
    U hit = (sh(king, -color, -1) | sh(king, -color, 1)) & en[1];
    for (int j = 0; j < 8; j++) hit |= (sh(king, ND[j][0], ND[j][1]) & en[2]) | (sh(king, KD[j][0], KD[j][1]) & en[6]);
    if (hit) return true;
    for (int d = 0; d < 8; d++) {
        U sl = en[5] | (d < 4 ? en[3] : en[4]), x = king;
        for (int k = 0; k < 7; k++) {
            x = sh(x, KD[d][0], KD[d][1]);
            if (x & sl) return true;
            if (!x || (x & occ)) break;
        }
    }
    return false;
}

struct Out { I i, pos, ep; int wr; I* code; };

// droits de roque perdus quand la case s est quittée ou atteinte (bits K, Q, k, q)
__device__ __forceinline__ int lose(int s) {
    return s == 60 ? 3 : s == 63 ? 1 : s == 56 ? 2 : s == 4 ? 12 : s == 7 ? 4 : s == 0 ? 8 : 0;
}

// un plan : F = cases de départ, d = décalage d'index, promo = départs qui promeuvent (4 coups)
__device__ void emit(Out& o, U F, int d, int p, U promo) {
    if (!o.wr) { o.pos += __popcll(F) + 3 * __popcll(F & promo); return; }
    while (F) {  // code du coup : parent, puis roque dame, roque roi, e.p. (1 bit chacun), k (3 bits), t, f (6 bits)
        int f = __ffsll((I)F) - 1, t = f + d;
        F &= F - 1;
        I code = o.i << 18 | (p == 69) << 17 | (p == 68) << 16 | ((p == 66 || p == 67) && t == o.ep) << 15 | t << 6 | f;
        if ((promo >> f) & 1) for (int k = 5; k >= 2; k--) o.code[o.pos++] = code | k << 12;
        else o.code[o.pos++] = code | 1 << 12;
    }
}

// un fil par parent. wr = 0 : cnt[i] <- nombre d'enfants légaux ;
// wr = 1 : cnt[i] = rang du premier enfant, écrit le code de chaque enfant
extern "C" __global__ void moves(const S* B, const bool* CR, const I* EP, I n, int color, int capt, int wr,
                                 I* cnt, I* code) {
    I i = (I)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const S* b = B + 65 * i;
    const bool* cr = CR + 4 * i;
    Out o = {i, wr ? cnt[i] : 0, EP[i], wr, code};

    U own[7] = {0, 0, 0, 0, 0, 0, 0}, en[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int q = 0; q < 64; q++) {
        int v = b[q] * color;
        if (v > 0) own[v] |= bit(q); else if (v < 0) en[-v] |= bit(q);
    }
    U ownAll = own[1] | own[2] | own[3] | own[4] | own[5] | own[6];
    U enAll = en[1] | en[2] | en[3] | en[4] | en[5] | en[6];
    U occ = ownAll | enAll, empty = ~occ, king = own[6];

    // cases attaquées par l'adversaire, notre roi retiré (il ne s'abrite pas derrière lui-même)
    U att = sh(en[1], color, -1) | sh(en[1], color, 1);
    for (int j = 0; j < 8; j++) att |= sh(en[2], ND[j][0], ND[j][1]) | sh(en[6], KD[j][0], KD[j][1]);
    for (int d = 0; d < 8; d++) {
        U front = sh(en[5] | (d < 4 ? en[3] : en[4]), KD[d][0], KD[d][1]);
        while (front) { att |= front; front = sh(front & (empty | king), KD[d][0], KD[d][1]); }
    }

    // échecs (check = arrivées qui parent l'échec) et clouages par axe
    U chk = (sh(king, -color, -1) | sh(king, -color, 1)) & en[1];
    for (int j = 0; j < 8; j++) chk |= sh(king, ND[j][0], ND[j][1]) & en[2];
    int nchk = __popcll(chk);
    U pinned = 0, pin[4] = {0, 0, 0, 0};
    for (int d = 0; d < 8; d++) {
        U sl = en[5] | (d < 4 ? en[3] : en[4]), x = king, ray = 0, mine = 0;
        for (int k = 0; k < 7; k++) {
            x = sh(x, KD[d][0], KD[d][1]);
            if (!x) break;
            if (!(x & occ)) { ray |= x; continue; }
            if (x & sl) {
                if (mine) { pinned |= mine; pin[axis(KD[d][0], KD[d][1])] |= mine; }
                else { nchk++; chk |= ray | x; }
                break;
            }
            if (mine || !(x & ownAll)) break;
            mine = x;
        }
    }
    U check = nchk == 0 ? ~0ull : nchk == 1 ? chk : 0;

    // cibles par type de pièce ; en quiescence : victime au moins aussi chère (MVV/LVA de expand_torch)
    U tg[7];
    const int rank[7] = {0, 1, 3, 3, 5, 9, 0};
    for (int a = 1; a < 7; a++) {
        tg[a] = ~ownAll;
        if (capt) { tg[a] = 0; for (int v = 1; v < 7; v++) if (rank[v] >= rank[a]) tg[a] |= en[v]; }
    }

    for (int d = 0; d < 8; d++) {  // plans 0..55 : glisseurs, et le roi à distance 1
        int dr = KD[d][0], dc = KD[d][1], a = d < 4 ? 3 : 4;
        U ok = ~pinned | pin[axis(dr, dc)];
        U cA = own[a] & ok, cQ = own[5] & ok, tA = tg[a] & check, tQ = tg[5] & check, e = empty;
        for (int k = 1; k <= 7; k++) {
            tA = sh(tA, -dr, -dc); tQ = sh(tQ, -dr, -dc); e = sh(e, -dr, -dc);
            U F = (cA & tA) | (cQ & tQ);
            if (k == 1) F |= king & sh(tg[6] & ~att, -dr, -dc);
            emit(o, F, (8 * dr + dc) * k, 7 * d + k - 1, 0);
            cA &= e; cQ &= e;
            if (!(cA | cQ)) break;
        }
    }
    for (int j = 0; j < 8; j++)  // 56..63 : cavaliers
        emit(o, own[2] & ~pinned & sh(tg[2] & check, -ND[j][0], -ND[j][1]), 8 * ND[j][0] + ND[j][1], 56 + j, 0);

    int fwd = -color;  // 64..67 : pions
    U promo = 0xFFull << (color == 1 ? 8 : 48), start = 0xFFull << (color == 1 ? 48 : 8);
    U push = own[1] & (~pinned | pin[0]) & sh(empty, -fwd, 0);
    emit(o, push & sh(check, -fwd, 0) & (capt ? promo : ~0ull), 8 * fwd, 64, promo);
    emit(o, capt ? 0 : push & start & sh(empty & check, -2 * fwd, 0), 16 * fwd, 65, 0);
    for (int dc = -1; dc <= 1; dc += 2) {
        U F = own[1] & (~pinned | pin[axis(fwd, dc)]) & sh(enAll & check, -fwd, -dc) & (sh(tg[1], -fwd, -dc) | promo);
        if (o.ep < 64 && !(occ & bit((int)o.ep))) {
            U fe = own[1] & sh(bit((int)o.ep), -fwd, -dc);
            if (fe) {  // prise en passant : jouer le coup, puis tester le roi
                U gone = bit((__ffsll((I)fe) - 1) / 8 * 8 + (int)o.ep % 8), e2[7];
                for (int v = 0; v < 7; v++) e2[v] = en[v] & ~gone;
                if (!king_attacked(king, (occ & ~fe & ~gone) | bit((int)o.ep), e2, color)) F |= fe;
            }
        }
        emit(o, F, 8 * fwd + dc, dc < 0 ? 66 : 67, promo);
    }

    if (!capt) {  // 68, 69 : roques
        int row = color == 1 ? 56 : 0, c0 = color == 1 ? 0 : 2;
        bool base = b[row + 4] * color == 6 && !((att >> (row + 4)) & 1);
        bool ks = base && cr[c0] && b[row + 7] * color == 4 && !(occ & (3ull << (row + 5))) && !(att & (3ull << (row + 5)));
        bool qs = base && cr[c0 + 1] && b[row] * color == 4 && !(occ & (7ull << (row + 1))) && !(att & (3ull << (row + 2)));
        emit(o, ks ? bit(row + 4) : 0, 2, 68, 0);
        emit(o, qs ? bit(row + 4) : 0, -2, 69, 0);
    }
    if (!wr) cnt[i] = o.pos;
}

// écrit l'octet v à la case q dans le mot k du plateau (q = 64, la sentinelle, n'est dans aucun mot)
__device__ __forceinline__ void poke(U& w, int k, int q, U v) {
    if ((q >> 3) == k) w = (w & ~(0xFFull << (8 * (q & 7)))) | (v << (8 * (q & 7)));
}

// un fil par enfant : b' = T @ b (mêmes écritures que reduced_gpu.apply_moves), droits de roque, case e.p.
// Le plateau passe par 8 mots de 64 bits : les lignes de 65 octets ne sont pas alignées, donc lectures
// de mots alignés recollés par décalage, et écriture des seuls mots alignés entiers de la ligne.
// parent[j] porte en entrée le code du coup (noyau moves), en sortie l'indice du parent.
extern "C" __global__ void children(const S* B, const bool* CR, I m, I* parent, I* mv, S* cb, bool* ccr, I* cep) {
    I j = (I)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    I code = parent[j], i = code >> 18;
    int f = code & 63, t = (code >> 6) & 63, k = (code >> 12) & 7, row = f - 4;
    int ec = (code >> 15) & 1 ? (f / 8) * 8 + t % 8 : 64;
    int f2 = (code >> 16) & 1 ? row + 7 : (code >> 17) & 1 ? row : 64;
    int t2 = (code >> 16) & 1 ? row + 5 : (code >> 17) & 1 ? row + 3 : 64;
    parent[j] = i;
    I* v = mv + 6 * j;
    v[0] = f; v[1] = t; v[2] = k; v[3] = ec; v[4] = f2; v[5] = t2;
    const S* b = B + 65 * i;
    const bool* cr = CR + 4 * i;
    S bf = b[f];
    U moved = (unsigned char)(S)(k * bf), rook = (unsigned char)b[f2];

    int a = (int)((size_t)b & 7);
    const U* in = (const U*)(b - a);
    U w[8], next = in[0];
    for (int k = 0; k < 8; k++) {
        U cur = next;
        next = in[k + 1];  // contient au plus loin la sentinelle : reste dans le tampon
        w[k] = (cur >> (8 * a)) | ((next << 1) << (63 - 8 * a));
        poke(w[k], k, t, moved); poke(w[k], k, t2, rook); poke(w[k], k, f, 0); poke(w[k], k, ec, 0); poke(w[k], k, f2, 0);
    }
    S* c = cb + 65 * j;
    int h = (8 - (int)((size_t)c & 7)) & 7;  // octets avant le premier mot aligné
    for (int q = 0; q < h; q++) c[q] = (S)(w[0] >> (8 * q));
    U* out = (U*)(c + h);
    for (int k = 0; k < 7; k++) out[k] = (w[k] >> (8 * h)) | ((w[k + 1] << 1) << (63 - 8 * h));
    for (int q = h + 56; q < 64; q++) c[q] = (S)(w[7] >> (8 * (q - 56)));
    c[64] = 0;

    int l = lose(f) | lose(t);
    for (int q = 0; q < 4; q++) ccr[4 * j + q] = cr[q] && !((l >> q) & 1);
    cep[j] = (bf == 1 || bf == -1) && (t - f == 16 || f - t == 16) ? (f + t) / 2 : 64;
}
"""

_kernels = None


def available(dev):
    """Le noyau fusionné est-il utilisable sur ce périphérique ?"""
    return cp is not None and torch.device(dev).type == "cuda"


def expand(B, cr, ep, color, captures_only=False, count_only=False):
    """Même contrat que gpu_engine.expand_torch, tenseurs sur cuda."""
    global _kernels
    if _kernels is None:
        mod = cp.RawModule(code=SOURCE)
        _kernels = mod.get_function("moves"), mod.get_function("children")
    moves, children = _kernels
    n, dev = len(B), B.device

    def out(m):  # parent, coups, B', cr', ep'
        return [torch.empty(m, dtype=torch.int64, device=dev), torch.empty(m, 6, dtype=torch.int64, device=dev),
                torch.empty(m, 65, dtype=torch.int8, device=dev), torch.empty(m, 4, dtype=torch.bool, device=dev),
                torch.empty(m, dtype=torch.int64, device=dev)]

    def grid(m):
        return ((m + 127) // 128,), (128,)

    if n == 0:
        return 0 if count_only else tuple(out(0))
    cnt = torch.empty(n, dtype=torch.int64, device=dev)
    with cp.cuda.Device(dev.index or 0):
        Bc, crc, epc = (cp.asarray(x.contiguous()) for x in (B, cr, ep))
        head = (Bc, crc, epc, np.int64(n), np.int32(color), np.int32(captures_only))
        moves(*grid(n), (*head, np.int32(0), cp.asarray(cnt), epc))  # comptage : aucun code écrit
        total = int(cnt.sum())
        if count_only or total == 0:
            return total if count_only else tuple(out(0))
        res = out(total)
        parent, mv, child, ccr, cep = (cp.asarray(x) for x in res)
        moves(*grid(n), (*head, np.int32(1), cp.asarray(cnt.cumsum(0) - cnt), parent))
        children(*grid(total), (Bc, crc, np.int64(total), parent, mv, child, ccr, cep))
    return tuple(res)
