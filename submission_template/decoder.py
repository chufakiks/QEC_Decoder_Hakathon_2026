"""Neural BP decoder with a blossom-matching (or OSD) fallback, for the [[18,4,4]] memory experiments.

Pipeline for one decode() call:
  1. Turn the circuit into a detector error model (DEM): H (detectors x error mechanisms), the observables each
     mechanism flips, and its prior probability. Mechanisms with identical detector and observable sets are merged.
  2. Deduplicate the detector patterns: many shots share a pattern, so each distinct pattern is decoded once.
  3. Run neural min-sum BP: every iteration t scales each check-to-variable message by a trained weight
     alpha[t, edge] and each prior LLR by a trained weight beta[t, mechanism] (see train_nbp.py; the weights are in
     nbp_weights.npz). If the hard decision reproduces the detector pattern, BP has converged and its error set gives
     the prediction.
  4. If BP is stuck (no valid error set after the last iteration), fall back to
       FALLBACK = "matching": minimum-weight perfect matching with our own Edmonds blossom implementation
                              (matching.py), on a graph that keeps the error pieces with at most two detectors, or
       FALLBACK = "osd":      ordered statistics decoding (OSD-CS) ordered by BP's beliefs, summing the probability
                              of single- and pair-flip candidates per logical class.
  5. The prediction is the XOR of the observable flips of the chosen mechanisms.

There are only 70 detectors, so every column of H fits in two 64-bit words and the OSD elimination is done with bit
operations. Everything numba compiles is warmed up at import, so compilation does not count towards the timed calls.
"""
import hashlib
import os

import numpy as np
import stim
from numba import njit, prange

from matching import MatchingGraph

# ----------------------------------------------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------------------------------------------
BP_MAX_ITER = 3          # min-sum iterations when no trained weights are used
MS_SCALING = 0.5         # normalisation factor of the min-sum check update
OSD_SINGLES = 2500       # OSD-CS: number of non-pivot columns tried as single flips
OSD_PAIRS = 150          # OSD-CS: number of non-pivot columns whose pairs are tried
OSD_TRIPLES = 0          # OSD-CS: number of non-pivot columns whose triples are tried
ALWAYS_OSD = False       # run OSD even when BP converges (needs FALLBACK = "osd")
POST_AVG = True          # order OSD by the posterior averaged over BP iterations
USE_NBP = True           # use the trained neural-BP weights (nbp_weights.npz) when they match the circuit
NBP_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbp_weights.npz")
FALLBACK = "belief-matching"  # when BP does not converge: "belief-matching" (blossom MWPM on a graph reweighted
                              # with the BP posteriors), "matching" (MWPM with prior weights) or "osd" (soft OSD-CS)
SOFT_OSD = True          # sum candidate probabilities per logical class instead of keeping the single best


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
    """Tanner graph (checks x mechanisms) as edge lists in check-major (CSR) order plus a var -> edges index."""
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
    return edge_chk, edge_var, chk_ptr, var_order, var_ptr


def _cols(det_lists):
    """Columns of H (detectors only) as two 64-bit words each."""
    cols = np.zeros((len(det_lists), 2), dtype=np.uint64)
    for j, dets in enumerate(det_lists):
        for d in dets:
            cols[j, d >> 6] |= np.uint64(1) << np.uint64(d & 63)
    return cols


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
def _build_tables(rank, piv_col, weights, obs_masks, cost_tab, obs_tab):
    """cost_tab[b, v] / obs_tab[b, v]: total weight / observable XOR of the pivots selected by byte b of a mask."""
    for b in range(cost_tab.shape[0]):
        cost_tab[b, 0] = 0.0
        obs_tab[b, 0] = 0
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
            else:
                cost_tab[b, v] = cost_tab[b, prev]
                obs_tab[b, v] = obs_tab[b, prev]


@njit(cache=True, inline="always")
def _eval(c_lo, c_hi, cost_tab, obs_tab):
    cost = 0.0
    obs = 0
    m = np.uint64(255)
    for b in range(8):
        v = (c_lo >> np.uint64(8 * b)) & m
        cost += cost_tab[b, v]
        obs ^= obs_tab[b, v]
    for b in range(8, cost_tab.shape[0]):
        v = (c_hi >> np.uint64(8 * (b - 8))) & m
        cost += cost_tab[b, v]
        obs ^= obs_tab[b, v]
    return cost, obs


@njit(cache=True)
def _bp(syn, edge_var, chk_ptr, var_order, var_ptr, llr, alpha_w, beta_w, post_avg, v2c, c2v, post, hard):
    """Weighted min-sum BP on one syndrome (one bit per check, redundant checks included).

    alpha_w[t, e] / beta_w[t, j] are the (trained) weights of iteration t. Fills post (posterior LLRs, averaged over
    the iterations if post_avg) and hard (hard decision); returns True if the hard decision satisfies every check.
    """
    n_det = chk_ptr.shape[0] - 1
    n = llr.shape[0]
    n_edge = edge_var.shape[0]

    # ---------------- belief propagation (normalised min-sum) ----------------
    for e in range(n_edge):
        v2c[e] = llr[edge_var[e]]
    for j in range(n):
        post[j] = llr[j]
    converged = False
    max_iter = alpha_w.shape[0]
    for it in range(max_iter):
        # check -> variable
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
                mag = m2 if e == arg1 else m1
                c2v[e] = alpha_w[it, e] * s * mag
        # variable -> check, posterior, hard decision
        for j in range(n):
            tot = beta_w[it, j] * llr[j]
            for k in range(var_ptr[j], var_ptr[j + 1]):
                tot += c2v[var_order[k]]
            if post_avg:
                post[j] = (post[j] * it + tot) / (it + 1) if it > 0 else tot
            else:
                post[j] = tot
            hard[j] = tot < 0
            for k in range(var_ptr[j], var_ptr[j + 1]):
                e = var_order[k]
                v2c[e] = tot - c2v[e]
        # syndrome check
        ok = True
        for c in range(n_det):
            par = syn[c]
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                if hard[edge_var[e]]:
                    par = not par
            if par:
                ok = False
                break
        if ok:
            converged = True
            break

    return converged


@njit(cache=True)
def _decode_one(syn, n_det, edge_chk, edge_var, chk_ptr, var_order, var_ptr, cols, llr, weights, obs_masks,
                rank_h, alpha_w, beta_w, n_single, n_pair, n_triple, always_osd, soft, post_avg, osd_fallback,
                v2c, c2v, post, hard):
    """Decode one syndrome (bool array: the n_det detectors, then the redundant checks if any).

    Returns the 4-bit observable prediction, or -1 if BP did not converge and osd_fallback is off (the caller then
    decodes the syndrome with matching).
    """
    n = llr.shape[0]
    converged = _bp(syn, edge_var, chk_ptr, var_order, var_ptr, llr, alpha_w, beta_w, post_avg, v2c, c2v, post, hard)

    if converged and not always_osd:
        bp_obs = 0
        for j in range(n):
            if hard[j]:
                bp_obs ^= obs_masks[j]
        return bp_obs
    if not converged and not osd_fallback:
        return -1

    # ---------------- OSD-CS ----------------
    order = np.argsort(post)  # most likely errors first
    piv_lo = np.zeros(rank_h, dtype=np.uint64)
    piv_hi = np.zeros(rank_h, dtype=np.uint64)
    pc_lo = np.zeros(rank_h, dtype=np.uint64)   # which pivots (by index) make up each reduced pivot vector
    pc_hi = np.zeros(rank_h, dtype=np.uint64)
    piv_bit = np.zeros(rank_h, dtype=np.int64)
    piv_col = np.zeros(rank_h, dtype=np.int64)
    nonpiv = np.empty(n_single, dtype=np.int64)
    np_lo = np.zeros(n_single, dtype=np.uint64)  # representation of each non-pivot column in the pivot basis
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
        else:
            # Dependent column: only columns after the information set is complete are proper OSD candidates,
            # but dependent columns found earlier are just as valid (and more reliable).
            if n_np < n_single:
                nonpiv[n_np] = j
                np_lo[n_np] = c_lo
                np_hi[n_np] = c_hi
                n_np += 1
    # Columns seen before the basis was complete may need re-expressing? No: a dependent column's representation
    # only uses pivots that existed when it was reduced, and those pivots never change afterwards.

    # Syndrome in the pivot basis.
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

    # Enumerate candidate error sets (base solution, single and pair flips of non-pivot columns, triples among the
    # first few) and add up their probability per logical class. Each candidate is a distinct error set, so the
    # sums approximate the probability of each of the 16 observable outcomes.
    n_bytes = (rank + 7) // 8
    if n_bytes < 9:
        n_bytes = 9
    cost_tab = np.empty((n_bytes, 256))
    obs_tab = np.empty((n_bytes, 256), dtype=np.int64)
    _build_tables(rank, piv_col, weights, obs_masks, cost_tab, obs_tab)
    n_p = min(n_pair, n_np)
    n_t = min(n_triple, n_np)
    n_cand = 1 + n_np + n_p * (n_p - 1) // 2 + n_t * (n_t - 1) * (n_t - 2) // 6
    costs = np.empty(n_cand)
    obss = np.empty(n_cand, dtype=np.int64)
    q = 0
    cost, obs = _eval(cs_lo, cs_hi, cost_tab, obs_tab)
    costs[q] = cost
    obss[q] = obs
    q += 1
    for a in range(n_np):
        ja = nonpiv[a]
        cost, obs = _eval(cs_lo ^ np_lo[a], cs_hi ^ np_hi[a], cost_tab, obs_tab)
        costs[q] = cost + weights[ja]
        obss[q] = obs ^ obs_masks[ja]
        q += 1
    for a in range(n_p):
        for b in range(a + 1, n_p):
            ja = nonpiv[a]
            jb = nonpiv[b]
            cost, obs = _eval(cs_lo ^ np_lo[a] ^ np_lo[b], cs_hi ^ np_hi[a] ^ np_hi[b], cost_tab, obs_tab)
            costs[q] = cost + weights[ja] + weights[jb]
            obss[q] = obs ^ obs_masks[ja] ^ obs_masks[jb]
            q += 1
    for a in range(n_t):
        for b in range(a + 1, n_t):
            for c in range(b + 1, n_t):
                cost, obs = _eval(cs_lo ^ np_lo[a] ^ np_lo[b] ^ np_lo[c],
                                           cs_hi ^ np_hi[a] ^ np_hi[b] ^ np_hi[c], cost_tab, obs_tab)
                costs[q] = cost + weights[nonpiv[a]] + weights[nonpiv[b]] + weights[nonpiv[c]]
                obss[q] = obs ^ obs_masks[nonpiv[a]] ^ obs_masks[nonpiv[b]] ^ obs_masks[nonpiv[c]]
                q += 1
    if not soft:
        best = 0
        for i in range(q):
            if costs[i] < costs[best]:
                best = i
        return obss[best]
    cmin = costs[0]
    for i in range(q):
        if costs[i] < cmin:
            cmin = costs[i]
    mass = np.zeros(16)
    for i in range(q):
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
def _decode_batch(syns, n_det, edge_chk, edge_var, chk_ptr, var_order, var_ptr, cols, llr, weights, obs_masks,
                  rank_h, alpha_w, beta_w, n_single, n_pair, n_triple, always_osd, soft, post_avg, osd_fallback):
    m = syns.shape[0]
    out = np.zeros(m, dtype=np.int64)
    n_chunks = min(m, 256)
    n_edge = edge_var.shape[0]
    n = llr.shape[0]
    for ch in prange(n_chunks):
        v2c = np.empty(n_edge)
        c2v = np.empty(n_edge)
        post = np.empty(n)
        hard = np.empty(n, dtype=np.bool_)
        for i in range(ch, m, n_chunks):
            out[i] = _decode_one(syns[i], n_det, edge_chk, edge_var, chk_ptr, var_order, var_ptr, cols, llr, weights,
                                 obs_masks, rank_h, alpha_w, beta_w, n_single, n_pair, n_triple, always_osd, soft, post_avg,
                                 osd_fallback, v2c, c2v, post, hard)
    return out


@njit(cache=True, parallel=True)
def _bp_post_batch(syns, edge_var, chk_ptr, var_order, var_ptr, llr, alpha_w, beta_w):
    """Final-iteration BP posterior LLRs for each syndrome (used to reweight the matching graph)."""
    m = syns.shape[0]
    n = llr.shape[0]
    n_edge = edge_var.shape[0]
    out = np.empty((m, n))
    n_chunks = min(m, 256)
    for ch in prange(n_chunks):
        v2c = np.empty(n_edge)
        c2v = np.empty(n_edge)
        post = np.empty(n)
        hard = np.empty(n, dtype=np.bool_)
        for i in range(ch, m, n_chunks):
            _bp(syns[i], edge_var, chk_ptr, var_order, var_ptr, llr, alpha_w, beta_w, False, v2c, c2v, post, hard)
            out[i] = post
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
_MATCHING_CACHE = {}
_WEIGHTS = None


def fingerprint(m):
    """Identifies an error model, so trained weights are only used with the circuit they were trained on."""
    h = hashlib.sha1()
    for d in m["det_lists"]:
        h.update(np.asarray(d, dtype=np.int64).tobytes())
        h.update(b"|")
    h.update(np.ascontiguousarray(m["obs_masks"]).tobytes())
    return h.hexdigest()


def extended_model(circuit: stim.Circuit, pairs=()):
    """Error model plus Tanner graph. pairs: (a, b) detector pairs whose XOR is added as a redundant check."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    key = (str(circuit), pairs.tobytes())
    if key not in _MODEL_CACHE:
        det_lists, obs_masks, priors, num_det = build_dem(circuit)
        check_dets = [(d,) for d in range(num_det)] + [tuple(p) for p in pairs.tolist()]
        ext_lists = []
        for dets in det_lists:
            s = set(dets)
            ext_lists.append(tuple(dets) + tuple(num_det + r for r, (a, b) in enumerate(pairs.tolist())
                                                 if (a in s) != (b in s)))
        edge_chk, edge_var, chk_ptr, var_order, var_ptr = build_graph(ext_lists, len(check_dets))
        cols = _cols(det_lists)
        priors = np.clip(priors, 1e-12, 0.5 - 1e-12)
        llr = np.log((1 - priors) / priors)
        _MODEL_CACHE[key] = dict(edge_chk=edge_chk, edge_var=edge_var, chk_ptr=chk_ptr, var_order=var_order,
                                 var_ptr=var_ptr, cols=cols, llr=llr, weights=llr.copy(), obs_masks=obs_masks,
                                 rank_h=_gf2_rank(cols), num_det=len(check_dets), base_num_det=num_det,
                                 det_lists=det_lists, check_dets=check_dets, red_pairs=pairs)
    return _MODEL_CACHE[key]


def _model(circuit: stim.Circuit):
    return extended_model(circuit)


def _trained(circuit):
    """(model, alpha_w, beta_w): the trained neural-BP weights and the graph they belong to, or plain min-sum."""
    global _WEIGHTS
    base = _model(circuit)
    if USE_NBP:
        if _WEIGHTS is None:
            _WEIGHTS = dict(np.load(NBP_WEIGHTS)) if os.path.exists(NBP_WEIGHTS) else {}
        fp = fingerprint(base)
        for b in "XZ":
            if str(_WEIGHTS.get(f"{b}_fingerprint", "")) == fp:
                m = extended_model(circuit, _WEIGHTS.get(f"{b}_pairs", ()))
                return m, _WEIGHTS[f"{b}_alpha"], _WEIGHTS[f"{b}_beta"]
    n_edge, n = len(base["edge_var"]), len(base["llr"])
    return base, np.full((BP_MAX_ITER, n_edge), MS_SCALING), np.ones((BP_MAX_ITER, n))


def _matching(circuit, m):
    key = str(circuit)
    if key not in _MATCHING_CACHE:
        _MATCHING_CACHE[key] = MatchingGraph(circuit, m)
    return _MATCHING_CACHE[key]


LAST_STATS = {}


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    m, alpha_w, beta_w = _trained(circuit)
    det = np.ascontiguousarray(np.asarray(detector_events, dtype=bool))
    shots = det.shape[0]
    packed = np.packbits(det, axis=1)
    uniq, inverse = np.unique(packed, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    syns = np.unpackbits(uniq, axis=1, count=det.shape[1]).astype(np.bool_)
    pairs = m["red_pairs"]
    if len(pairs):
        syns = np.concatenate([syns, syns[:, pairs[:, 0]] ^ syns[:, pairs[:, 1]]], axis=1)
    nonzero = syns.any(axis=1)
    obs_u = np.zeros(len(uniq), dtype=np.int64)
    if nonzero.any():
        obs_u[nonzero] = _decode_batch(
            np.ascontiguousarray(syns[nonzero]), m["base_num_det"], m["edge_chk"], m["edge_var"], m["chk_ptr"],
            m["var_order"], m["var_ptr"], m["cols"], m["llr"], m["weights"], m["obs_masks"], m["rank_h"],
            alpha_w, beta_w, OSD_SINGLES, OSD_PAIRS, OSD_TRIPLES, ALWAYS_OSD, SOFT_OSD, POST_AVG,
            FALLBACK == "osd")
    stuck = np.flatnonzero(obs_u < 0)
    if len(stuck):
        mg = _matching(circuit, m)
        posts = None
        if FALLBACK == "belief-matching":
            posts = _bp_post_batch(np.ascontiguousarray(syns[stuck]), m["edge_var"], m["chk_ptr"], m["var_order"],
                                   m["var_ptr"], m["llr"], alpha_w, beta_w)
        for r, i in enumerate(stuck):
            fired = np.flatnonzero(syns[i, : m["base_num_det"]])
            obs_u[i] = mg.decode_syndrome(fired, None if posts is None else posts[r])
    LAST_STATS.update(unique=len(uniq), stuck_unique=len(stuck),
                      stuck_shots=int(np.isin(inverse, stuck).sum()), shots=shots)
    obs = obs_u[inverse]
    out = np.zeros((shots, 4), dtype=np.uint8)
    for k in range(4):
        out[:, k] = (obs >> k) & 1
    return out


def _warmup():
    """Compile the numba kernels on a tiny circuit so compilation happens at import, not inside decode()."""
    global FALLBACK
    c = stim.Circuit.generated("repetition_code:memory", rounds=2, distance=3, before_round_data_depolarization=0.01)
    det = c.compile_detector_sampler(seed=0).sample(8)
    saved = FALLBACK
    for fb in ("osd", "belief-matching"):
        FALLBACK = fb
        decode(c, det)
    FALLBACK = saved
    _MODEL_CACHE.clear()
    _MATCHING_CACHE.clear()


if not os.environ.get("NBP_SKIP_WARMUP"):
    _warmup()
