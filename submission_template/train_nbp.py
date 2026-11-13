"""Train the weights of the neural (weighted) min-sum BP used by decoder.py.

Follows "Neural Belief-Matching Decoding for Topological Quantum Error Correction Codes" (Menti & Lazaro,
arXiv:2603.21730), adapted from the toric code with phenomenological noise to our circuit-level error model:

  * Neural BP (Sec. II-D). The BP is unrolled for T iterations. Iteration t scales each check-to-variable message by
    a trainable weight alpha[t, edge] and each prior LLR by beta[t, mechanism] (eq. 5).
  * Overcomplete check matrix (RNBP, footnote 2). Optionally add redundant checks: for every detector, the XOR of it
    with the two detectors whose error sets overlap it most (the analogue of the toric code's 2n weight-6 checks,
    which are products of each stabilizer with its two neighbours). Their syndrome bit is the XOR of the two.
  * Weight sharing (Sec. III-B). The paper ties weights across spatial translations of the toric lattice. Our
    circuit is translation invariant in time instead: rounds 1-4 repeat the same gates. Mechanisms that are time
    translates of each other (same detector pattern relative to their earliest round, same observables, same prior)
    share one weight, and so do the corresponding edges. 2,470 mechanisms collapse to about 1,100 classes.
  * Degeneracy-aware loss (eq. 6). An estimate is correct iff e + e_hat has no syndrome and flips no observable, so
    the loss is sum over rows r of H and of the observable matrix of f(sum_{i in r} P(e_hat_i != e_i)), with
    f(x) = |sin(pi x / 2)|. It does not punish an estimate for differing from the true error by a stabilizer.
    "--loss comb" adds the bit-wise cross-entropy to it with weight --bce-weight.

Training data is sampled straight from the detector error model: each mechanism fires independently with its
probability, the syndrome is H e mod 2. The loss is averaged over all iterations so that early iterations are good
too.

Run from the repository root (writes submission_template/nbp_weights.npz):

    python submission_template/train_nbp.py [--steps 1500] [--iters 8] [--share] [--redundant 2] [--loss comb]
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import stim
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("NBP_SKIP_WARMUP", "1")
import decoder as D  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------
# Redundant checks and weight-sharing classes
# ----------------------------------------------------------------------------------------------------------------
def redundant_pairs(base, per_detector):
    """Pairs (a, b) of detectors whose XOR is added as a redundant check: each detector with the `per_detector`
    detectors it shares the most error mechanisms with."""
    if per_detector <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    nd = base["num_det"]
    H = np.zeros((nd, len(base["llr"])), dtype=np.int64)
    H[base["edge_chk"], base["edge_var"]] = 1
    overlap = H @ H.T
    np.fill_diagonal(overlap, -1)
    pairs = set()
    for a in range(nd):
        for b in np.argsort(-overlap[a], kind="stable")[:per_detector]:
            pairs.add((min(a, int(b)), max(a, int(b))))
    return np.array(sorted(pairs), dtype=np.int64)


def sharing_classes(circuit, model):
    """Class index for every mechanism and every Tanner-graph edge, tying time translates together."""
    coords = circuit.get_detector_coordinates()
    det_lists = model["det_lists"]
    obs = model["obs_masks"]
    llr = model["llr"]

    def rel(d, t0):
        c = coords[d]
        return tuple(c[:-1]) + (round(c[-1] - t0, 6),)

    var_keys, var_t0 = [], []
    for j, dets in enumerate(det_lists):
        t0 = min(coords[d][-1] for d in dets)
        var_t0.append(t0)
        var_keys.append((tuple(sorted(rel(d, t0) for d in dets)), int(obs[j]), round(float(llr[j]), 6)))
    var_ids = {k: i for i, k in enumerate(dict.fromkeys(var_keys))}
    var_cls = np.array([var_ids[k] for k in var_keys], dtype=np.int64)

    check_dets = model["check_dets"]  # detectors making up each (possibly redundant) check
    edge_keys = []
    for c, j in zip(model["edge_chk"], model["edge_var"]):
        edge_keys.append((var_cls[j], tuple(sorted(rel(d, var_t0[j]) for d in check_dets[c]))))
    edge_ids = {k: i for i, k in enumerate(dict.fromkeys(edge_keys))}
    edge_cls = np.array([edge_ids[k] for k in edge_keys], dtype=np.int64)
    return var_cls, edge_cls


# ----------------------------------------------------------------------------------------------------------------
# Neural min-sum BP
# ----------------------------------------------------------------------------------------------------------------
class NeuralMinSum(torch.nn.Module):
    def __init__(self, model, iters, alpha0, var_cls, edge_cls):
        super().__init__()
        chk_ptr = model["chk_ptr"]
        n_chk = len(chk_ptr) - 1
        n_edge = len(model["edge_var"])
        deg = np.diff(chk_ptr)
        slot = np.full((n_chk, int(deg.max())), n_edge, dtype=np.int64)  # (check, slot) -> edge, n_edge = padding
        for c in range(n_chk):
            slot[c, : deg[c]] = np.arange(chk_ptr[c], chk_ptr[c + 1])
        self.register_buffer("slot", torch.as_tensor(slot))
        self.register_buffer("flat_valid", torch.as_tensor(np.flatnonzero((slot != n_edge).ravel())))
        self.register_buffer("edge_var", torch.as_tensor(model["edge_var"], dtype=torch.long))
        self.register_buffer("llr", torch.as_tensor(model["llr"], dtype=torch.float32))
        self.register_buffer("var_cls", torch.as_tensor(var_cls))
        self.register_buffer("edge_cls", torch.as_tensor(edge_cls))
        self.iters = iters
        self.alpha = torch.nn.Parameter(torch.full((iters, int(edge_cls.max()) + 1), float(alpha0)))
        self.beta = torch.nn.Parameter(torch.ones(iters, int(var_cls.max()) + 1))

    def full_weights(self):
        return self.alpha[:, self.edge_cls], self.beta[:, self.var_cls]

    def forward(self, syn):
        """syn: float (B, n_checks) of 0/1. Returns the posterior LLRs (B, n_var) of every iteration."""
        B = syn.shape[0]
        alpha, beta = self.full_weights()
        syn_sign = 1.0 - 2.0 * syn
        v2c = self.llr[self.edge_var].expand(B, -1)
        big = torch.full((B, 1), 1e4, dtype=v2c.dtype)
        outs = []
        for t in range(self.iters):
            x = torch.cat([v2c, big], dim=1)[:, self.slot]  # (B, C, D)
            neg = x < 0
            sgn = 1.0 - 2.0 * neg.float()
            mag = x.abs()
            prod = (1.0 - 2.0 * (neg.sum(dim=2) % 2).float()) * syn_sign
            m1, idx = mag.min(dim=2, keepdim=True)
            is_min = torch.zeros_like(neg).scatter_(2, idx, True)
            m2 = mag.masked_fill(is_min, 1e9).min(dim=2, keepdim=True).values
            c2v = (prod.unsqueeze(2) * sgn * torch.where(is_min, m2, m1)).reshape(B, -1)[:, self.flat_valid]
            c2v = c2v * alpha[t]
            tot = (beta[t] * self.llr).expand(B, -1).index_add(1, self.edge_var, c2v)
            outs.append(tot)
            v2c = tot[:, self.edge_var] - c2v
        return outs


# ----------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------
class Sampler:
    def __init__(self, model, seed):
        self.priors = 1 / (1 + np.exp(model["llr"]))
        nd = model["base_num_det"]
        Hb = np.zeros((nd, len(self.priors)), dtype=np.uint8)
        for j, dets in enumerate(model["det_lists"]):
            Hb[list(dets), j] = 1
        self.Hb = torch.as_tensor(Hb, dtype=torch.float32)
        self.L = torch.as_tensor(np.array([[(o >> k) & 1 for k in range(4)] for o in model["obs_masks"]]),
                                 dtype=torch.float32)  # (n, 4)
        self.pairs = torch.as_tensor(model["red_pairs"])
        self.rng = np.random.default_rng(seed)

    def sample(self, batch):
        err = torch.as_tensor(self.rng.random((batch, len(self.priors))) < self.priors, dtype=torch.float32)
        syn = (err @ self.Hb.T) % 2
        if len(self.pairs):
            syn = torch.cat([syn, (syn[:, self.pairs[:, 0]] + syn[:, self.pairs[:, 1]]) % 2], dim=1)
        return syn, err


def degeneracy_loss(post, err, Hb, L):
    """Eq. (6): sum over rows of H and L of |sin(pi/2 * sum_{i in row} P(e_hat_i != e_i))|, averaged over shots."""
    p1 = torch.sigmoid(-post)                       # P(e_hat_i = 1)
    q = torch.where(err > 0.5, 1.0 - p1, p1)        # P(e_hat_i != e_i)
    x = torch.cat([q @ Hb.T, q @ L], dim=1)
    return torch.sin(0.5 * torch.pi * x).abs().sum(dim=1).mean()


def train_one(circuit_path, args):
    circuit = stim.Circuit.from_file(circuit_path)
    base = D._model(circuit)
    pairs = redundant_pairs(base, args.redundant)
    model = D.extended_model(circuit, pairs)
    if args.share:
        var_cls, edge_cls = sharing_classes(circuit, model)
    else:
        var_cls, edge_cls = np.arange(len(model["llr"])), np.arange(len(model["edge_var"]))
    print(f"  checks {len(model['chk_ptr']) - 1} (redundant {len(pairs)}), edges {len(model['edge_var'])}, "
          f"weight classes: {var_cls.max() + 1} mechanism, {edge_cls.max() + 1} edge", flush=True)
    net = NeuralMinSum(model, args.iters, D.MS_SCALING, var_cls, edge_cls)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    sampler = Sampler(model, args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        syn, err = sampler.sample(args.batch)
        outs = net(syn)
        losses = []
        for o in outs:
            loss = 0.0
            if args.loss in ("degen", "comb"):
                loss = loss + degeneracy_loss(o, err, sampler.Hb, sampler.L)
            if args.loss in ("bce", "comb"):
                w = args.bce_weight if args.loss == "comb" else 1.0
                loss = loss + w * torch.nn.functional.binary_cross_entropy_with_logits(-o, err)
            losses.append(loss)
        loss = sum(losses) / len(losses)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                hard = (outs[-1] < 0).float()
                resid = (hard + err) % 2
                ok_syn = ((resid @ sampler.Hb.T) % 2 == 0).all(dim=1)
                ok_log = ((resid @ sampler.L) % 2 == 0).all(dim=1)
            print(f"  step {step:5d}  loss {loss.item():.4f}  converged {ok_syn.float().mean():.3f}  "
                  f"converged & logically correct {(ok_syn & ok_log).float().mean():.3f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    alpha, beta = net.full_weights()
    return dict(alpha=alpha.detach().numpy().astype(np.float64), beta=beta.detach().numpy().astype(np.float64),
                pairs=pairs, fingerprint=D.fingerprint(base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="data/train", help="folder with X_circuit.stim and Z_circuit.stim")
    ap.add_argument("--bases", default="XZ")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share", action="store_true", help="tie weights of time-translated mechanisms and edges")
    ap.add_argument("--redundant", type=int, default=0, help="redundant checks per detector (0 = none)")
    ap.add_argument("--loss", choices=["bce", "degen", "comb"], default="comb")
    ap.add_argument("--bce-weight", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(HERE, "nbp_weights.npz"))
    args = ap.parse_args()
    print(vars(args), flush=True)
    out = {}
    for b in args.bases:
        print(f"training {b} ...", flush=True)
        w = train_one(os.path.join(args.circuits, f"{b}_circuit.stim"), args)
        for k, v in w.items():
            out[f"{b}_{k}"] = v
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
