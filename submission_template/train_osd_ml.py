"""Train the machine-learning correction on top of BP+OSD (weights in osd_ml_weights.npz).

OSD gives an approximate probability for each of the 16 logical classes (it sums the prior probability of the
~13,600 candidate error sets it tries). Those estimates are systematically off: candidates are missing, and the
search is ordered by a few BP iterations. A small MLP learns to correct them:

    input   the 70 detector bits, OSD's 16 class log-probabilities, the best candidate's cost, # fired detectors
    output  per observable: OSD's log-odds + MLP residual   (the residual layer starts at zero, so the untrained
            network reproduces OSD exactly)
    loss    binary cross-entropy against the true observable flips, which is what the leaderboard scores

Training data is sampled fresh from each circuit with stim (seeds that none of the repository's data sets use), so
there is no limit on its size. data/train (100,000 labelled shots per basis) is only used to report the LER.

Run from the repository root:

    python submission_template/train_osd_ml.py [--shots 1000000] [--epochs 8]
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
os.environ.setdefault("SKIP_WARMUP", "1")
import decoder as D  # noqa: E402
from hackathon_2026 import load_kaggle_csv  # noqa: E402


class Net(torch.nn.Module):
    def __init__(self, n_in, hidden, mean, std, cols, dropout):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        self.register_buffer("cols", torch.as_tensor(cols))
        self.dropout = torch.nn.Dropout(dropout)
        layers, d = [], n_in
        for h in hidden:
            layers.append(torch.nn.Linear(d, h))
            d = h
        last = torch.nn.Linear(d, 4)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.layers = torch.nn.ModuleList(layers + [last])

    def forward(self, x, base_logits):
        h = (x[:, self.cols] - self.mean) / self.std
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.dropout(torch.relu(h))
        return base_logits + h

    def export(self, prefix):
        out = {f"{prefix}mean": self.mean.numpy(), f"{prefix}std": self.std.numpy(), f"{prefix}cols": self.cols.numpy()}
        for i, layer in enumerate(self.layers):
            out[f"{prefix}W{i}"] = layer.weight.detach().numpy().T.astype(np.float32)
            out[f"{prefix}b{i}"] = layer.bias.detach().numpy().astype(np.float32)
        return out


def ler(logits, y):
    return float(((logits > 0) != (y > 0.5)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="data/train")
    ap.add_argument("--valid", default="data/train/train.csv", help="labelled shots used only for reporting")
    ap.add_argument("--shots", type=int, default=1_000_000, help="generated training shots per basis")
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--features", choices=["all", "osd"], default="all",
                    help="all: detector bits + OSD outputs; osd: only OSD's class log-probabilities, cost, # fired")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(HERE, "osd_ml_weights.npz"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    valid = load_kaggle_csv(args.valid)
    hidden = [int(h) for h in args.hidden.split(",")]
    out = {}
    for bi, b in enumerate("XZ"):
        circuit = stim.Circuit.from_file(os.path.join(args.circuits, f"{b}_circuit.stim"))
        t0 = time.time()
        det, obs = circuit.compile_detector_sampler(seed=args.seed + bi).sample(args.shots, separate_observables=True)
        x, osd_pred, _ = D.osd_features(circuit, det)
        print(f"{b}: OSD features for {args.shots} generated shots in {time.time() - t0:.0f} s; "
              f"OSD LER on them {np.mean(osd_pred != obs):.5f}", flush=True)
        rows = valid["basis"] == b
        vx, vpred, _ = D.osd_features(circuit, valid["detector_events"][rows])
        vy = torch.as_tensor(valid["observable_flips"][rows], dtype=torch.float32)
        vx_t, vb_t = torch.as_tensor(vx), torch.as_tensor(D.class_logits(vx))
        print(f"{b}: OSD LER on {args.valid}: {np.mean(vpred != valid['observable_flips'][rows]):.5f}", flush=True)

        cols = np.arange(x.shape[1]) if args.features == "all" else np.arange(70, x.shape[1])
        mean = x[:, cols].mean(axis=0)
        std = x[:, cols].std(axis=0) + 1e-3
        net = Net(len(cols), hidden, mean, std, cols, args.dropout)
        # The last 10% of the generated shots pick the best epoch (early stopping).
        n_hold = len(x) // 10
        X = torch.as_tensor(x[:-n_hold])
        B = torch.as_tensor(D.class_logits(x[:-n_hold]))
        Y = torch.as_tensor(obs[:-n_hold], dtype=torch.float32)
        hX, hB = torch.as_tensor(x[-n_hold:]), torch.as_tensor(D.class_logits(x[-n_hold:]))
        hY = torch.as_tensor(obs[-n_hold:], dtype=torch.float32)
        best = (ler(hB, hY), None, 0)
        print(f"  held-out generated shots: OSD LER {best[0]:.5f}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * ((len(X) + args.batch - 1) // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
        for epoch in range(args.epochs):
            perm = torch.randperm(len(X))
            net.train()
            tot = 0.0
            for i in range(0, len(X), args.batch):
                idx = perm[i: i + args.batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[idx], B[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                h = ler(net(hX, hB), hY)
                v = ler(net(vx_t, vb_t), vy)
            print(f"  epoch {epoch + 1}: train loss {tot / len(X):.5f}  held-out LER {h:.5f}  "
                  f"validation LER {v:.5f}", flush=True)
            if h < best[0]:
                best = (h, {k: t.clone() for k, t in net.state_dict().items()}, epoch + 1)
        if best[1] is None:
            print("  no epoch beat plain OSD on the held-out shots: the correction is left at zero", flush=True)
            for layer in net.layers[-1:]:
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        else:
            net.load_state_dict(best[1])
            print(f"  keeping epoch {best[2]}", flush=True)
        net.eval()
        with torch.no_grad():
            print(f"{b}: validation LER  OSD {ler(vb_t, vy):.5f}  ->  OSD + ML {ler(net(vx_t, vb_t), vy):.5f}",
                  flush=True)
        out.update(net.export(f"{b}_"))
        out[f"{b}_fingerprint"] = D.fingerprint(D._model(circuit))
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
