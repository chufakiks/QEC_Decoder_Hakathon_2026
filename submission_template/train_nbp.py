"""Train the weights of the neural (weighted) min-sum BP used by decoder.py.

The BP is unrolled for T iterations. Each iteration t has
  - one weight per Tanner-graph edge that scales the check-to-variable message  (alpha[t, e], starts at MS_SCALING),
  - one weight per error mechanism that scales its prior log-likelihood ratio   (beta[t, j],  starts at 1).
Training data is sampled straight from the detector error model, so every example comes with the true error vector:
each mechanism j fires independently with probability p_j, the syndrome is H e mod 2. The loss is the binary
cross-entropy between each iteration's posterior and the true errors, summed over iterations, so the network is
pushed to give good beliefs early as well as late.

Run from the repository root (writes submission_template/nbp_weights.npz):

    python submission_template/train_nbp.py [--steps 3000] [--iters 6] [--batch 512]
"""
import argparse
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


class NeuralMinSum(torch.nn.Module):
    def __init__(self, model, iters, alpha0):
        super().__init__()
        edge_chk = torch.as_tensor(model["edge_chk"], dtype=torch.long)
        edge_var = torch.as_tensor(model["edge_var"], dtype=torch.long)
        chk_ptr = model["chk_ptr"]
        n_chk = len(chk_ptr) - 1
        n_edge = len(edge_var)
        deg = np.diff(chk_ptr)
        max_deg = int(deg.max())
        # slot (c, s) -> edge chk_ptr[c] + s, padded with the dummy edge n_edge
        slot = np.full((n_chk, max_deg), n_edge, dtype=np.int64)
        for c in range(n_chk):
            slot[c, : deg[c]] = np.arange(chk_ptr[c], chk_ptr[c + 1])
        self.register_buffer("slot", torch.as_tensor(slot))
        self.register_buffer("flat_valid", torch.as_tensor(np.flatnonzero((slot != n_edge).ravel())))
        self.register_buffer("edge_chk", edge_chk)
        self.register_buffer("edge_var", edge_var)
        self.register_buffer("llr", torch.as_tensor(model["llr"], dtype=torch.float32))
        self.n_var = len(model["llr"])
        self.n_edge = n_edge
        self.iters = iters
        self.alpha = torch.nn.Parameter(torch.full((iters, n_edge), float(alpha0)))
        self.beta = torch.nn.Parameter(torch.ones(iters, self.n_var))

    def forward(self, syn):
        """syn: float (B, n_chk) of 0/1. Returns a list of posterior LLRs (B, n_var), one per iteration."""
        B = syn.shape[0]
        syn_sign = 1.0 - 2.0 * syn  # (B, C)
        v2c = self.llr[self.edge_var].expand(B, -1)
        outs = []
        big = torch.full((B, 1), 1e4, dtype=v2c.dtype)
        for t in range(self.iters):
            x = torch.cat([v2c, big], dim=1)[:, self.slot]  # (B, C, D)
            neg = x < 0
            sgn = 1.0 - 2.0 * neg.float()
            mag = x.abs()
            prod = (1.0 - 2.0 * (neg.sum(dim=2) % 2).float()) * syn_sign  # (B, C)
            m1, idx = mag.min(dim=2, keepdim=True)
            is_min = torch.zeros_like(neg).scatter_(2, idx, True)
            m2 = mag.masked_fill(is_min, 1e9).min(dim=2, keepdim=True).values
            mag_excl = torch.where(is_min, m2, m1)
            c2v_slot = prod.unsqueeze(2) * sgn * mag_excl  # (B, C, D)
            c2v = c2v_slot.reshape(B, -1)[:, self.flat_valid]  # (B, E) in check-major edge order
            c2v = c2v * self.alpha[t]
            tot = (self.beta[t] * self.llr).expand(B, -1).clone()
            tot = tot.index_add(1, self.edge_var, c2v)
            outs.append(tot)
            v2c = tot[:, self.edge_var] - c2v
        return outs


def sample(model, H_rows, obs, priors, batch, rng):
    err = rng.random((batch, len(priors))) < priors
    syn = np.zeros((batch, model["num_det"]), dtype=bool)
    for c, cols in enumerate(H_rows):
        syn[:, c] = np.bitwise_xor.reduce(err[:, cols], axis=1)
    return syn, err


def train_one(circuit_path, iters, steps, batch, lr, seed):
    circuit = stim.Circuit.from_file(circuit_path)
    model = D._model(circuit)
    priors = 1 / (1 + np.exp(model["llr"]))
    H_rows = [model["edge_var"][model["chk_ptr"][c]: model["chk_ptr"][c + 1]] for c in range(model["num_det"])]
    net = NeuralMinSum(model, iters, D.MS_SCALING)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        syn, err = sample(model, H_rows, None, priors, batch, rng)
        outs = net(torch.as_tensor(syn, dtype=torch.float32))
        target = torch.as_tensor(err, dtype=torch.float32)
        # P(error) = sigmoid(-LLR)
        loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(-o, target) for o in outs) / len(outs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                hard = (outs[-1] < 0).numpy()
                bit_err = (hard != err).any(axis=1).mean()
            print(f"  step {step:5d}  loss {loss.item():.5f}  shots with any wrong error bit {bit_err:.3f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return dict(alpha=net.alpha.detach().numpy().astype(np.float64),
                beta=net.beta.detach().numpy().astype(np.float64),
                fingerprint=D.fingerprint(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="data/train", help="folder with X_circuit.stim and Z_circuit.stim")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "nbp_weights.npz"))
    args = ap.parse_args()
    out = {}
    for b in "XZ":
        print(f"training {b} ...", flush=True)
        w = train_one(os.path.join(args.circuits, f"{b}_circuit.stim"), args.iters, args.steps, args.batch,
                      args.lr, args.seed)
        for k, v in w.items():
            out[f"{b}_{k}"] = v
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
