"""Minimum-weight perfect matching decoder, with our own implementation of Edmonds' blossom algorithm.

Used as the fallback when the neural BP does not converge.

Matching graph
    Stim splits each error mechanism into graphlike pieces (one or two detectors) where it can. Every piece becomes
    an edge between its two detectors, or between its detector and the boundary. Edges with the same endpoints are
    merged: their probabilities are combined and the edge keeps the observable flips of its most likely piece.
    Pieces with three or more detectors (stim cannot split about 9% of mechanisms in this code) are dropped, so the
    matching graph only approximates the full error model.
    Edge weight = log((1 - p) / p). Floyd-Warshall gives the shortest path and its observable flips between every
    pair of nodes (70 detectors plus the boundary).

Decoding one syndrome
    The k fired detectors each get a boundary twin. Detector-detector edges have the shortest-path weight,
    detector-twin edges the distance to the boundary, twin-twin edges weight 0. A minimum-weight perfect matching
    of these 2k nodes pairs every fired detector with another fired detector or with the boundary; the prediction
    is the XOR of the observable flips along the matched paths.

Blossom algorithm
    `max_weight_matching` is the primal-dual O(n^3) algorithm of Edmonds (1965) in the form described by Galil,
    "Efficient algorithms for finding maximum matching in graphs" (1986): grow alternating trees from free vertices,
    shrink odd cycles into blossoms, expand blossoms whose dual variable reaches zero, and adjust the dual variables
    by the smallest slack until an augmenting path appears. Minimum-weight perfect matching is maximum-weight
    maximum-cardinality matching on the weights (C - w).
"""
import numpy as np
import stim


# ----------------------------------------------------------------------------------------------------------------
# Edmonds' blossom algorithm (maximum-weight matching on a general graph)
# ----------------------------------------------------------------------------------------------------------------
def max_weight_matching(edges, maxcardinality=False):
    """edges: list of (i, j, w) with integer weights. Returns mate[v] (the partner of v, or -1)."""
    if not edges:
        return []
    nedge = len(edges)
    nvertex = 1 + max(max(i, j) for i, j, _ in edges)
    maxweight = max(0, max(w for _, _, w in edges))

    # Endpoint p of edge k = p // 2 is edges[k][p % 2]; p ^ 1 is the other end.
    endpoint = [edges[p // 2][p % 2] for p in range(2 * nedge)]
    neighbend = [[] for _ in range(nvertex)]
    for k, (i, j, _) in enumerate(edges):
        neighbend[i].append(2 * k + 1)
        neighbend[j].append(2 * k)

    mate = [-1] * nvertex                     # remote endpoint of the matched edge
    label = [0] * (2 * nvertex)               # 0 free, 1 S (outer), 2 T (inner)
    labelend = [-1] * (2 * nvertex)           # endpoint through which the label was assigned
    inblossom = list(range(nvertex))          # top-level blossom of each vertex
    blossomparent = [-1] * (2 * nvertex)
    blossomchilds = [None] * (2 * nvertex)
    blossombase = list(range(nvertex)) + [-1] * nvertex
    blossomendps = [None] * (2 * nvertex)
    bestedge = [-1] * (2 * nvertex)           # least-slack edge to an S blossom
    blossombestedges = [None] * (2 * nvertex)
    unusedblossoms = list(range(nvertex, 2 * nvertex))
    dualvar = [maxweight] * nvertex + [0] * nvertex
    allowedge = [False] * nedge               # edge has zero slack
    queue = []

    def slack(k):
        i, j, w = edges[k]
        return dualvar[i] + dualvar[j] - 2 * w

    def leaves(b):
        if b < nvertex:
            yield b
        else:
            for t in blossomchilds[b]:
                if t < nvertex:
                    yield t
                else:
                    yield from leaves(t)

    def assign_label(w, t, p):
        b = inblossom[w]
        label[w] = label[b] = t
        labelend[w] = labelend[b] = p
        bestedge[w] = bestedge[b] = -1
        if t == 1:
            queue.extend(leaves(b))
        else:
            base = blossombase[b]
            assign_label(endpoint[mate[base]], 1, mate[base] ^ 1)

    def scan_blossom(v, w):
        """Trace back from v and w; return the base of the new blossom, or -1 for an augmenting path."""
        path = []
        base = -1
        while v != -1 or w != -1:
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            path.append(b)
            label[b] = 5
            if labelend[b] == -1:
                v = -1
            else:
                v = endpoint[labelend[b]]
                b = inblossom[v]
                v = endpoint[labelend[b]]
            if w != -1:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base

    def add_blossom(base, k):
        v, w, _ = edges[k]
        bb, bv, bw = inblossom[base], inblossom[v], inblossom[w]
        b = unusedblossoms.pop()
        blossombase[b] = base
        blossomparent[b] = -1
        blossomparent[bb] = b
        path, endps = [], []
        blossomchilds[b], blossomendps[b] = path, endps
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            endps.append(labelend[bv])
            v = endpoint[labelend[bv]]
            bv = inblossom[v]
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        while bw != bb:
            blossomparent[bw] = b
            path.append(bw)
            endps.append(labelend[bw] ^ 1)
            w = endpoint[labelend[bw]]
            bw = inblossom[w]
        label[b] = 1
        labelend[b] = labelend[bb]
        dualvar[b] = 0
        for v in leaves(b):
            if label[inblossom[v]] == 2:
                queue.append(v)  # former T vertices become S
            inblossom[v] = b
        bestedgeto = [-1] * (2 * nvertex)
        for bv in path:
            if blossombestedges[bv] is None:
                nblists = [[p // 2 for p in neighbend[v]] for v in leaves(bv)]
            else:
                nblists = [blossombestedges[bv]]
            for nblist in nblists:
                for kk in nblist:
                    i, j, _ = edges[kk]
                    if inblossom[j] == b:
                        i, j = j, i
                    bj = inblossom[j]
                    if (bj != b and label[bj] == 1
                            and (bestedgeto[bj] == -1 or slack(kk) < slack(bestedgeto[bj]))):
                        bestedgeto[bj] = kk
            blossombestedges[bv] = None
            bestedge[bv] = -1
        blossombestedges[b] = [kk for kk in bestedgeto if kk != -1]
        bestedge[b] = -1
        for kk in blossombestedges[b]:
            if bestedge[b] == -1 or slack(kk) < slack(bestedge[b]):
                bestedge[b] = kk

    def expand_blossom(b, endstage):
        for s in blossomchilds[b]:
            blossomparent[s] = -1
            if s < nvertex:
                inblossom[s] = s
            elif endstage and dualvar[s] == 0:
                expand_blossom(s, endstage)
            else:
                for v in leaves(s):
                    inblossom[v] = s
        if not endstage and label[b] == 2:
            # Relabel the sub-blossoms along the even-length path from the entry child to the base.
            entrychild = inblossom[endpoint[labelend[b] ^ 1]]
            j = blossomchilds[b].index(entrychild)
            if j & 1:
                j -= len(blossomchilds[b])
                jstep, endptrick = 1, 0
            else:
                jstep, endptrick = -1, 1
            p = labelend[b]
            while j != 0:
                label[endpoint[p ^ 1]] = 0
                label[endpoint[blossomendps[b][j - endptrick] ^ endptrick ^ 1]] = 0
                assign_label(endpoint[p ^ 1], 2, p)
                allowedge[blossomendps[b][j - endptrick] // 2] = True
                j += jstep
                p = blossomendps[b][j - endptrick] ^ endptrick
                allowedge[p // 2] = True
                j += jstep
            bv = blossomchilds[b][j]
            label[endpoint[p ^ 1]] = label[bv] = 2
            labelend[endpoint[p ^ 1]] = labelend[bv] = p
            bestedge[bv] = -1
            j += jstep
            while blossomchilds[b][j] != entrychild:
                bv = blossomchilds[b][j]
                if label[bv] == 1:
                    j += jstep
                    continue
                v = -1
                for v in leaves(bv):
                    if label[v] != 0:
                        break
                if label[v] != 0:
                    label[v] = 0
                    label[endpoint[mate[blossombase[bv]]]] = 0
                    assign_label(v, 2, labelend[v])
                j += jstep
        label[b] = labelend[b] = -1
        blossomchilds[b] = blossomendps[b] = None
        blossombase[b] = -1
        blossombestedges[b] = None
        bestedge[b] = -1
        unusedblossoms.append(b)

    def augment_blossom(b, v):
        """Swap matched/unmatched edges inside blossom b so that v becomes its base."""
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        if t >= nvertex:
            augment_blossom(t, v)
        i = j = blossomchilds[b].index(t)
        if i & 1:
            j -= len(blossomchilds[b])
            jstep, endptrick = 1, 0
        else:
            jstep, endptrick = -1, 1
        while j != 0:
            j += jstep
            t = blossomchilds[b][j]
            p = blossomendps[b][j - endptrick] ^ endptrick
            if t >= nvertex:
                augment_blossom(t, endpoint[p])
            j += jstep
            t = blossomchilds[b][j]
            if t >= nvertex:
                augment_blossom(t, endpoint[p ^ 1])
            mate[endpoint[p]] = p ^ 1
            mate[endpoint[p ^ 1]] = p
        blossomchilds[b] = blossomchilds[b][i:] + blossomchilds[b][:i]
        blossomendps[b] = blossomendps[b][i:] + blossomendps[b][:i]
        blossombase[b] = blossombase[blossomchilds[b][0]]

    def augment_matching(k):
        v, w, _ = edges[k]
        for s, p in ((v, 2 * k + 1), (w, 2 * k)):
            while True:
                bs = inblossom[s]
                if bs >= nvertex:
                    augment_blossom(bs, s)
                mate[s] = p
                if labelend[bs] == -1:
                    break  # reached a free vertex (root of the tree)
                t = endpoint[labelend[bs]]
                bt = inblossom[t]
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ^ 1]
                if bt >= nvertex:
                    augment_blossom(bt, j)
                mate[j] = labelend[bt]
                p = labelend[bt] ^ 1

    for _ in range(nvertex):
        # One stage: find an augmenting path (or stop).
        label[:] = [0] * (2 * nvertex)
        bestedge[:] = [-1] * (2 * nvertex)
        blossombestedges[nvertex:] = [None] * nvertex
        allowedge[:] = [False] * nedge
        queue[:] = []
        for v in range(nvertex):
            if mate[v] == -1 and label[inblossom[v]] == 0:
                assign_label(v, 1, -1)
        augmented = False
        while True:
            while queue and not augmented:
                v = queue.pop()
                for p in neighbend[v]:
                    k = p // 2
                    w = endpoint[p]
                    if inblossom[v] == inblossom[w]:
                        continue
                    if not allowedge[k]:
                        kslack = slack(k)
                        if kslack <= 0:
                            allowedge[k] = True
                    if allowedge[k]:
                        if label[inblossom[w]] == 0:
                            assign_label(w, 2, p ^ 1)
                        elif label[inblossom[w]] == 1:
                            base = scan_blossom(v, w)
                            if base >= 0:
                                add_blossom(base, k)
                            else:
                                augment_matching(k)
                                augmented = True
                                break
                        elif label[w] == 0:
                            label[w] = 2
                            labelend[w] = p ^ 1
                    elif label[inblossom[w]] == 1:
                        b = inblossom[v]
                        if bestedge[b] == -1 or kslack < slack(bestedge[b]):
                            bestedge[b] = k
                    elif label[w] == 0:
                        if bestedge[w] == -1 or kslack < slack(bestedge[w]):
                            bestedge[w] = k
            if augmented:
                break
            # No augmenting path with the current duals: change them by the smallest delta.
            deltatype, delta, deltaedge, deltablossom = -1, None, None, None
            if not maxcardinality:
                deltatype, delta = 1, min(dualvar[:nvertex])
            for v in range(nvertex):
                if label[inblossom[v]] == 0 and bestedge[v] != -1:
                    d = slack(bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta, deltatype, deltaedge = d, 2, bestedge[v]
            for b in range(2 * nvertex):
                if blossomparent[b] == -1 and label[b] == 1 and bestedge[b] != -1:
                    d = slack(bestedge[b]) // 2
                    if deltatype == -1 or d < delta:
                        delta, deltatype, deltaedge = d, 3, bestedge[b]
            for b in range(nvertex, 2 * nvertex):
                if (blossombase[b] >= 0 and blossomparent[b] == -1 and label[b] == 2
                        and (deltatype == -1 or dualvar[b] < delta)):
                    delta, deltatype, deltablossom = dualvar[b], 4, b
            if deltatype == -1:
                deltatype, delta = 1, max(0, min(dualvar[:nvertex]))
            for v in range(nvertex):
                lb = label[inblossom[v]]
                if lb == 1:
                    dualvar[v] -= delta
                elif lb == 2:
                    dualvar[v] += delta
            for b in range(nvertex, 2 * nvertex):
                if blossombase[b] >= 0 and blossomparent[b] == -1:
                    if label[b] == 1:
                        dualvar[b] += delta
                    elif label[b] == 2:
                        dualvar[b] -= delta
            if deltatype == 1:
                break  # optimum reached
            elif deltatype == 2:
                allowedge[deltaedge] = True
                i, j, _ = edges[deltaedge]
                if label[inblossom[i]] == 0:
                    i, j = j, i
                queue.append(i)
            elif deltatype == 3:
                allowedge[deltaedge] = True
                i, j, _ = edges[deltaedge]
                queue.append(i)
            else:
                expand_blossom(deltablossom, False)
        if not augmented:
            break
        for b in range(nvertex, 2 * nvertex):
            if (blossomparent[b] == -1 and blossombase[b] >= 0 and label[b] == 1 and dualvar[b] == 0):
                expand_blossom(b, True)

    return [endpoint[m] if m != -1 else -1 for m in mate]


def min_weight_perfect_matching(n, edges):
    """Minimum-weight perfect matching of nodes 0..n-1; edges (i, j, w) with integer w >= 0."""
    big = 1 + max(w for _, _, w in edges)
    mate = max_weight_matching([(i, j, big - w) for i, j, w in edges], maxcardinality=True)
    mate += [-1] * (n - len(mate))
    return mate


# ----------------------------------------------------------------------------------------------------------------
# Matching graph from the circuit
# ----------------------------------------------------------------------------------------------------------------
WEIGHT_SCALE = 1000  # weights are rounded to integers (in units of 1/1000 of an LLR) for the blossom algorithm
P_MAX = 0.5 - 1e-6   # posteriors above 1/2 are clipped so that every edge weight stays non-negative


class MatchingGraph:
    """Matching graph of the detector error model; edge weights from the priors or from per-shot BP posteriors.

    m is the decoder's error model (decoder.extended_model): mechanism j has detectors m["det_lists"][j] and
    observable mask m["obs_masks"][j]. Stim's decomposition splits each mechanism into graphlike pieces; a piece with
    three or more detectors (stim cannot split about 9% of the mechanisms here) is split greedily into pairs that
    are already edges of the graph, and any leftover detector is joined to the boundary (belief-matching's
    "decompose each hyperedge into edges and redistribute probability mass"). Every piece carries the probability
    of its mechanism, so reweighting the mechanisms by BP posteriors reweights the graph.
    """

    def __init__(self, circuit: stim.Circuit, m):
        n_det = m["base_num_det"]
        self.n_node = n_det + 1
        self.boundary = B = n_det
        index = {(tuple(d), int(o)): j for j, (d, o) in enumerate(zip(m["det_lists"], m["obs_masks"]))}
        dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        pieces = []  # (mechanism, detector list, obs)
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            comps = [[[], 0]]
            for t in inst.targets_copy():
                if t.is_separator():
                    comps.append([[], 0])
                elif t.is_relative_detector_id():
                    comps[-1][0].append(t.val)
                elif t.is_logical_observable_id():
                    comps[-1][1] ^= 1 << t.val
            full, fobs = set(), 0
            for dets, o in comps:
                full ^= set(dets)
                fobs ^= o
            j = index.get((tuple(sorted(full)), fobs))
            if j is None:
                continue  # undetectable mechanism
            for dets, o in comps:
                if dets:
                    pieces.append((j, sorted(set(dets)), o))
        graphlike = {tuple(d) if len(d) == 2 else (d[0], B) for _, d, _ in pieces if len(d) <= 2}
        mech, edge, obs = [], [], []
        self.edge_index = {}
        self.split = 0

        def add(j, u, v, o):
            e = self.edge_index.setdefault((min(u, v), max(u, v)), len(self.edge_index))
            mech.append(j)
            edge.append(e)
            obs.append(o)

        for j, dets, o in pieces:
            if len(dets) == 1:
                add(j, dets[0], B, o)
            elif len(dets) == 2:
                add(j, dets[0], dets[1], o)
            else:
                self.split += 1
                left = list(dets)
                first = True
                while left:
                    a = left.pop(0)
                    partner = next((b for b in left if (min(a, b), max(a, b)) in graphlike), None)
                    if partner is None and left:
                        partner = left[0]
                    if partner is None:
                        add(j, a, B, o if first else 0)
                    else:
                        left.remove(partner)
                        add(j, a, partner, o if first else 0)
                    first = False  # the observable flips go with the first piece
        self.piece_mech = np.array(mech)
        self.piece_edge = np.array(edge)
        self.n_edge = len(self.edge_index)
        ends = np.array(list(self.edge_index.keys()))
        self.eu, self.ev = ends[:, 0], ends[:, 1]
        # Observable flips of an edge: those of its most likely piece under the priors.
        prior = 1 / (1 + np.exp(m["llr"]))
        self.prior = prior
        best = {}
        for pm, pe, po in zip(mech, edge, obs):
            if pe not in best or prior[pm] > best[pe][0]:
                best[pe] = (prior[pm], po)
        self.edge_obs = np.array([best[e][1] for e in range(self.n_edge)], dtype=np.int64)
        self.obs_mat = np.zeros((self.n_node, self.n_node), dtype=np.int64)
        self.obs_mat[self.eu, self.ev] = self.edge_obs
        self.obs_mat[self.ev, self.eu] = self.edge_obs
        self.prior_paths = self._all_pairs(self._edge_weights(prior))

    def _edge_weights(self, p_mech):
        p = np.clip(p_mech, 1e-12, P_MAX)[self.piece_mech]
        s = np.bincount(self.piece_edge, weights=np.log1p(-2 * p), minlength=self.n_edge)
        pe = np.clip((1 - np.exp(s)) / 2, 1e-12, P_MAX)  # independent pieces on one edge combine as an XOR
        return np.log((1 - pe) / pe)

    def _all_pairs(self, w):
        """Floyd-Warshall over all nodes: (distance, observable flips of the shortest path)."""
        n = self.n_node
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        dist[self.eu, self.ev] = dist[self.ev, self.eu] = w
        obs = self.obs_mat.copy()
        for k in range(n):
            via = dist[:, k:k + 1] + dist[k:k + 1, :]
            better = via < dist
            dist = np.where(better, via, dist)
            obs = np.where(better, obs[:, k:k + 1] ^ obs[k:k + 1, :], obs)
        return dist, obs

    def decode_syndrome(self, fired, posterior_llr=None):
        """fired: indices of fired detectors. posterior_llr: BP posteriors per mechanism (None = use priors).
        Returns the 4-bit observable prediction."""
        fired = [int(f) for f in fired]
        k = len(fired)
        if k == 0:
            return 0
        B = self.boundary
        if posterior_llr is None:
            dist, obs = self.prior_paths
            D = dist[np.ix_(fired + [B], fired + [B])]
            O = obs[np.ix_(fired + [B], fired + [B])]
        else:
            dist, obs = self._all_pairs(self._edge_weights(1 / (1 + np.exp(posterior_llr))))
            D = dist[np.ix_(fired + [B], fired + [B])]
            O = obs[np.ix_(fired + [B], fired + [B])]
        D = np.where(np.isfinite(D), np.round(D * WEIGHT_SCALE), 10 ** 9).astype(np.int64)
        edges = []
        for a in range(k):
            edges.append((a, k + a, int(D[a, k])))
            for b in range(a + 1, k):
                edges.append((a, b, int(D[a, b])))
                edges.append((k + a, k + b, 0))
        mate = min_weight_perfect_matching(2 * k, edges)
        res = 0
        for a in range(k):
            mt = mate[a]
            if mt == k + a:
                res ^= int(O[a, k])
            elif a < mt < k:
                res ^= int(O[a, mt])
        return res
