"""BP+OSD decoder for the [[18,4,4]] memory experiments, with OSD started from several BP rankings.

Pipeline for one decode() call:
  1. Turn the circuit into a detector error model (DEM): H (detectors x error mechanisms), the observables each
     mechanism flips, and its prior probability. Mechanisms with identical detector and observable sets are merged.
  2. Deduplicate the detector patterns: many shots share a pattern, so each distinct pattern is decoded once.
  3. Run normalised min-sum BP (scaling 0.5) for max(RANKINGS) iterations, keeping the posterior LLRs averaged over
     the first t iterations for every t in RANKINGS (default 1, 3 and 5). BP is only used to rank the mechanisms.
  4. For each of those rankings, run OSD-CS: take the most reliable independent set of mechanisms (GF(2)
     elimination) as the base solution and also try every single flip and the pairs among the 150 most reliable
     remaining mechanisms (about 13,600 candidate error sets per ranking).
  5. Merge the candidates of all rankings (the same error set found twice is counted once, via a 64-bit Zobrist
     hash), add up their prior probabilities per logical class, and predict each of the 4 observables by majority
     of that probability.

There are only 70 detectors, so every column of H fits in two 64-bit words and all of the elimination is done
with bit operations. Everything numba compiles is warmed up at import, so compilation does not count towards the
timed decode calls.

LER on all 200,000 labelled shots of data/train: 0.0348 with RANKINGS = (1, 3, 5), 0.0352 with (3,).
"""
import numpy as np
import stim
from numba import njit, prange

# ----------------------------------------------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------------------------------------------
RANKINGS = (1, 3, 5)     # OSD runs from the BP posteriors averaged over the first t iterations, for each t
MS_SCALING = 0.5         # normalisation factor of the min-sum check update
OSD_SINGLES = 2500       # OSD-CS: number of non-pivot columns tried as single flips (2,500 = all of them)
OSD_PAIRS = 150          # OSD-CS: number of non-pivot columns whose pairs are tried


# ----------------------------------------------------------------------------------------------------------------
# Detector error model
# ----------------------------------------------------------------------------------------------------------------
def build_dem(circuit: stim.Circuit):
    """Return (det_lists, obs_masks, priors, num_detectors) with duplicate mechanisms merged."""
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    merged = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = [], 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs ^= 1 << t.val
        # A target listed twice cancels out.
        dets = tuple(sorted(d for d in set(dets) if dets.count(d) % 2 == 1))
        if not dets:
            continue  # undetectable: cannot be corrected, irrelevant to decoding
        key = (dets, obs)
        q = merged.get(key, 0.0)
        merged[key] = q * (1 - p) + p * (1 - q)  # XOR of two independent events
    keys = list(merged)
    det_lists = [k[0] for k in keys]
    obs_masks = np.array([k[1] for k in keys], dtype=np.uint8)
    priors = np.array([merged[k] for k in keys], dtype=np.float64)
    return det_lists, obs_masks, priors, dem.num_detectors


def build_graph(det_lists, num_det):
    """Tanner graph as edge lists in check-major (CSR) order plus a var -> edges index."""
    n = len(det_lists)
    edge_chk, edge_var = [], []
    for j, dets in enumerate(det_lists):
        for d in dets:
            edge_chk.append(d)
            edge_var.append(j)
    edge_chk = np.array(edge_chk, dtype=np.int32)
    edge_var = np.array(edge_var, dtype=np.int32)
    order = np.lexsort((edge_var, edge_chk))
    edge_chk, edge_var = edge_chk[order], edge_var[order]
    chk_ptr = np.zeros(num_det + 1, dtype=np.int32)
    np.add.at(chk_ptr, edge_chk + 1, 1)
    chk_ptr = np.cumsum(chk_ptr).astype(np.int32)
    var_order = np.argsort(edge_var, kind="stable").astype(np.int32)  # edges grouped by variable
    var_ptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(var_ptr, edge_var + 1, 1)
    var_ptr = np.cumsum(var_ptr).astype(np.int32)
    # Columns of H as two 64-bit words.
    cols = np.zeros((n, 2), dtype=np.uint64)
    for j, dets in enumerate(det_lists):
        for d in dets:
            cols[j, d >> 6] |= np.uint64(1) << np.uint64(d & 63)
    return edge_chk, edge_var, chk_ptr, var_order, var_ptr, cols


# ----------------------------------------------------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _bit(lo, hi, k):
    if k < 64:
        return (lo >> np.uint64(k)) & np.uint64(1)
    return (hi >> np.uint64(k - 64)) & np.uint64(1)


@njit(cache=True)
def _lowest_bit(lo, hi):
    if lo != 0:
        k = 0
        while ((lo >> np.uint64(k)) & np.uint64(1)) == 0:
            k += 1
        return k
    k = 0
    while ((hi >> np.uint64(k)) & np.uint64(1)) == 0:
        k += 1
    return 64 + k


@njit(cache=True)
def _build_tables(rank, piv_col, weights, obs_masks, zob, cost_tab, obs_tab, zob_tab):
    """For byte b of a pivot bitmask with value v: total weight, observable XOR and Zobrist hash of those pivots."""
    for b in range(cost_tab.shape[0]):
        cost_tab[b, 0] = 0.0
        obs_tab[b, 0] = 0
        zob_tab[b, 0] = 0
        for v in range(1, 256):
            low = v & (-v)
            k = 0
            while (low >> k) != 1:
                k += 1
            p = b * 8 + k
            prev = v ^ low
            if p < rank:
                j = piv_col[p]
                cost_tab[b, v] = cost_tab[b, prev] + weights[j]
                obs_tab[b, v] = obs_tab[b, prev] ^ obs_masks[j]
                zob_tab[b, v] = zob_tab[b, prev] ^ zob[j]
            else:
                cost_tab[b, v] = cost_tab[b, prev]
                obs_tab[b, v] = obs_tab[b, prev]
                zob_tab[b, v] = zob_tab[b, prev]


@njit(cache=True, inline="always")
def _eval(c_lo, c_hi, cost_tab, obs_tab, zob_tab):
    cost = 0.0
    obs = 0
    z = np.uint64(0)
    m = np.uint64(255)
    for b in range(cost_tab.shape[0]):
        if b < 8:
            v = (c_lo >> np.uint64(8 * b)) & m
        else:
            v = (c_hi >> np.uint64(8 * (b - 8))) & m
        cost += cost_tab[b, v]
        obs ^= obs_tab[b, v]
        z ^= zob_tab[b, v]
    return cost, obs, z


@njit(cache=True)
def _osd_candidates(post, syn, cols, weights, obs_masks, zob, rank_h, n_single, n_pair, costs, obss, hashes, q):
    """OSD-CS from the ranking given by post (most likely first). Appends candidates from index q; returns new q."""
    n = post.shape[0]
    n_det = syn.shape[0]
    order = np.argsort(post)
    piv_lo = np.zeros(rank_h, dtype=np.uint64)
    piv_hi = np.zeros(rank_h, dtype=np.uint64)
    pc_lo = np.zeros(rank_h, dtype=np.uint64)   # which pivots (by index) make up each reduced pivot vector
    pc_hi = np.zeros(rank_h, dtype=np.uint64)
    piv_bit = np.zeros(rank_h, dtype=np.int64)
    piv_col = np.zeros(rank_h, dtype=np.int64)
    nonpiv = np.empty(n_single, dtype=np.int64)
    np_lo = np.zeros(n_single, dtype=np.uint64)  # each non-pivot column written in the pivot basis
    np_hi = np.zeros(n_single, dtype=np.uint64)
    n_np = 0
    rank = 0
    one = np.uint64(1)
    for idx in range(n):
        if rank == rank_h and n_np == n_single:
            break
        j = order[idx]
        v_lo = cols[j, 0]
        v_hi = cols[j, 1]
        c_lo = np.uint64(0)
        c_hi = np.uint64(0)
        for k in range(rank):
            if _bit(v_lo, v_hi, piv_bit[k]):
                v_lo ^= piv_lo[k]
                v_hi ^= piv_hi[k]
                c_lo ^= pc_lo[k]
                c_hi ^= pc_hi[k]
        if v_lo != 0 or v_hi != 0:
            if rank < rank_h:
                piv_lo[rank] = v_lo
                piv_hi[rank] = v_hi
                if rank < 64:
                    c_lo ^= one << np.uint64(rank)
                else:
                    c_hi ^= one << np.uint64(rank - 64)
                pc_lo[rank] = c_lo
                pc_hi[rank] = c_hi
                piv_bit[rank] = _lowest_bit(v_lo, v_hi)
                piv_col[rank] = j
                rank += 1
        elif n_np < n_single:
            # A dependent column's representation only uses pivots that already existed, which never change.
            nonpiv[n_np] = j
            np_lo[n_np] = c_lo
            np_hi[n_np] = c_hi
            n_np += 1
    # The syndrome in the pivot basis: the base solution.
    s_lo = np.uint64(0)
    s_hi = np.uint64(0)
    for c in range(n_det):
        if syn[c]:
            if c < 64:
                s_lo |= one << np.uint64(c)
            else:
                s_hi |= one << np.uint64(c - 64)
    cs_lo = np.uint64(0)
    cs_hi = np.uint64(0)
    for k in range(rank):
        if _bit(s_lo, s_hi, piv_bit[k]):
            s_lo ^= piv_lo[k]
            s_hi ^= piv_hi[k]
            cs_lo ^= pc_lo[k]
            cs_hi ^= pc_hi[k]
    nb = max(9, (rank + 7) // 8)
    cost_tab = np.empty((nb, 256))
    obs_tab = np.empty((nb, 256), dtype=np.int64)
    zob_tab = np.empty((nb, 256), dtype=np.uint64)
    _build_tables(rank, piv_col, weights, obs_masks, zob, cost_tab, obs_tab, zob_tab)
    c, o, z = _eval(cs_lo, cs_hi, cost_tab, obs_tab, zob_tab)
    costs[q] = c
    obss[q] = o
    hashes[q] = z
    q += 1
    for a in range(n_np):
        ja = nonpiv[a]
        c, o, z = _eval(cs_lo ^ np_lo[a], cs_hi ^ np_hi[a], cost_tab, obs_tab, zob_tab)
        costs[q] = c + weights[ja]
        obss[q] = o ^ obs_masks[ja]
        hashes[q] = z ^ zob[ja]
        q += 1
    n_p = min(n_pair, n_np)
    for a in range(n_p):
        for b in range(a + 1, n_p):
            ja = nonpiv[a]
            jb = nonpiv[b]
            c, o, z = _eval(cs_lo ^ np_lo[a] ^ np_lo[b], cs_hi ^ np_hi[a] ^ np_hi[b], cost_tab, obs_tab, zob_tab)
            costs[q] = c + weights[ja] + weights[jb]
            obss[q] = o ^ obs_masks[ja] ^ obs_masks[jb]
            hashes[q] = z ^ zob[ja] ^ zob[jb]
            q += 1
    return q


@njit(cache=True)
def _decode_one(syn, edge_var, chk_ptr, var_order, var_ptr, cols, llr, obs_masks, zob, rank_h, alpha, rankings,
                n_single, n_pair, v2c, c2v, post, snaps, costs, obss, hashes, table, stamp, cur):
    """Decode one syndrome. Returns the 4-bit observable prediction."""
    n = llr.shape[0]
    n_det = chk_ptr.shape[0] - 1
    n_rank = rankings.shape[0]
    max_it = 0
    for r in range(n_rank):
        if rankings[r] > max_it:
            max_it = rankings[r]

    # ---------------- min-sum BP; keep the averaged posterior after each requested iteration ----------------
    for e in range(edge_var.shape[0]):
        v2c[e] = llr[edge_var[e]]
    for it in range(max_it):
        for c in range(n_det):
            sgn = -1.0 if syn[c] else 1.0
            m1 = 1e300
            m2 = 1e300
            arg1 = -1
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                x = v2c[e]
                if x < 0:
                    sgn = -sgn
                    x = -x
                if x < m1:
                    m2 = m1
                    m1 = x
                    arg1 = e
                elif x < m2:
                    m2 = x
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                x = v2c[e]
                s = sgn * (-1.0 if x < 0 else 1.0)
                c2v[e] = alpha * s * (m2 if e == arg1 else m1)
        for j in range(n):
            tot = llr[j]
            for k in range(var_ptr[j], var_ptr[j + 1]):
                tot += c2v[var_order[k]]
            post[j] = (post[j] * it + tot) / (it + 1) if it > 0 else tot
            for k in range(var_ptr[j], var_ptr[j + 1]):
                e = var_order[k]
                v2c[e] = tot - c2v[e]
        for r in range(n_rank):
            if rankings[r] == it + 1:
                snaps[r] = post

    # ---------------- OSD-CS from every ranking ----------------
    q = 0
    for r in range(n_rank):
        q = _osd_candidates(snaps[r], syn, cols, llr, obs_masks, zob, rank_h, n_single, n_pair,
                            costs, obss, hashes, q)

    # ---------------- probability per logical class, each distinct error set counted once ----------------
    cmin = 1e300
    for i in range(q):
        if costs[i] < cmin:
            cmin = costs[i]
    mask = table.shape[0] - 1
    mass = np.zeros(16)
    for i in range(q):
        if n_rank > 1:
            h = hashes[i]
            slot = np.int64(h & np.uint64(mask))
            dup = False
            while stamp[slot] == cur:
                if table[slot] == h:
                    dup = True
                    break
                slot = (slot + 1) & mask
            if dup:
                continue
            stamp[slot] = cur
            table[slot] = h
        mass[obss[i]] += np.exp(cmin - costs[i])
    res = 0
    for k in range(4):
        m1 = 0.0
        m0 = 0.0
        for o in range(16):
            if (o >> k) & 1:
                m1 += mass[o]
            else:
                m0 += mass[o]
        if m1 > m0:
            res |= 1 << k
    return res


@njit(cache=True, parallel=True)
def _decode_batch(syns, edge_var, chk_ptr, var_order, var_ptr, cols, llr, obs_masks, zob, rank_h, alpha, rankings,
                  n_single, n_pair):
    m = syns.shape[0]
    out = np.zeros(m, dtype=np.uint8)
    n_chunks = min(m, 256)
    n_edge = edge_var.shape[0]
    n = llr.shape[0]
    cap = rankings.shape[0] * (1 + n_single + n_pair * (n_pair - 1) // 2)
    size = 1
    while size < 2 * cap:
        size *= 2
    for ch in prange(n_chunks):
        v2c = np.empty(n_edge)
        c2v = np.empty(n_edge)
        post = np.empty(n)
        snaps = np.empty((rankings.shape[0], n))
        costs = np.empty(cap)
        obss = np.empty(cap, dtype=np.int64)
        hashes = np.empty(cap, dtype=np.uint64)
        table = np.empty(size, dtype=np.uint64)   # open-addressing hash set; a slot is in use if stamp == cur
        stamp = np.zeros(size, dtype=np.int64)
        cur = 0
        for i in range(ch, m, n_chunks):
            cur += 1
            out[i] = _decode_one(syns[i], edge_var, chk_ptr, var_order, var_ptr, cols, llr, obs_masks, zob, rank_h,
                                 alpha, rankings, n_single, n_pair, v2c, c2v, post, snaps, costs, obss, hashes,
                                 table, stamp, cur)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Python driver
# ----------------------------------------------------------------------------------------------------------------
def _gf2_rank(cols):
    rows = [int(c[0]) | (int(c[1]) << 64) for c in cols]
    basis = {}
    for v in rows:
        while v:
            b = v.bit_length() - 1
            if b in basis:
                v ^= basis[b]
            else:
                basis[b] = v
                break
    return len(basis)


_MODEL_CACHE = {}


def _model(circuit: stim.Circuit):
    key = str(circuit)
    if key not in _MODEL_CACHE:
        det_lists, obs_masks, priors, num_det = build_dem(circuit)
        edge_chk, edge_var, chk_ptr, var_order, var_ptr, cols = build_graph(det_lists, num_det)
        priors = np.clip(priors, 1e-12, 0.5 - 1e-12)
        llr = np.log((1 - priors) / priors)
        zob = np.random.default_rng(12345).integers(0, 2 ** 63, size=len(llr), dtype=np.int64).astype(np.uint64)
        _MODEL_CACHE[key] = dict(edge_var=edge_var, chk_ptr=chk_ptr, var_order=var_order, var_ptr=var_ptr,
                                 cols=cols, llr=llr, obs_masks=obs_masks, zob=zob, rank_h=_gf2_rank(cols))
    return _MODEL_CACHE[key]


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    m = _model(circuit)
    det = np.ascontiguousarray(np.asarray(detector_events, dtype=bool))
    packed = np.packbits(det, axis=1)
    uniq, inverse = np.unique(packed, axis=0, return_inverse=True)
    syns = np.unpackbits(uniq, axis=1, count=det.shape[1]).astype(np.bool_)
    nonzero = syns.any(axis=1)
    obs_u = np.zeros(len(uniq), dtype=np.uint8)
    if nonzero.any():
        obs_u[nonzero] = _decode_batch(
            np.ascontiguousarray(syns[nonzero]), m["edge_var"], m["chk_ptr"], m["var_order"], m["var_ptr"],
            m["cols"], m["llr"], m["obs_masks"], m["zob"], m["rank_h"], MS_SCALING,
            np.array(RANKINGS, dtype=np.int64), min(OSD_SINGLES, len(m["llr"])), OSD_PAIRS)
    obs = obs_u[inverse.reshape(-1)]
    return ((obs[:, None] >> np.arange(4)) & 1).astype(np.uint8)


def _warmup():
    """Compile the numba kernels on a tiny circuit so compilation happens at import, not inside decode()."""
    c = stim.Circuit.generated("repetition_code:memory", rounds=2, distance=3, before_round_data_depolarization=0.01)
    det = c.compile_detector_sampler(seed=0).sample(8)
    decode(c, det)
    _MODEL_CACHE.clear()


_warmup()
