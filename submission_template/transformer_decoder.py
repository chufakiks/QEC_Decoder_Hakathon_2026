"""Transformer decoder for the [[18,4,4]] memory experiments (train and decode).

Model
    Each of the 70 detectors is a token: embedding(fired or not) + a learned embedding of the detector (its position
    in space and time) + a learned embedding of the basis (X or Z experiment). Four learned query tokens, one per
    logical observable, are appended. A small pre-norm transformer encoder lets every token attend to every other;
    the four query tokens are read out as the log-odds that L0..L3 flipped.
    One model serves both bases.

Training
    Supervised, with binary cross-entropy against the true observable flips. Every batch is freshly sampled from the
    X and Z circuits with stim, so the model never sees the same shot twice and cannot overfit; data/train is only
    used to report the LER while training.

Run from the repository root:

    python submission_template/transformer_decoder.py --hours 2      # trains, writes transformer.pt
"""
import argparse
import math
import os
import time

import numpy as np
import stim
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "transformer.pt")
N_DET, N_OBS = 70, 4


class TransformerDecoder(torch.nn.Module):
    def __init__(self, d_model=64, n_layers=3, n_heads=4, d_ff=128, dropout=0.0):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
        self.value = torch.nn.Embedding(2, d_model)
        self.position = torch.nn.Parameter(torch.randn(2, N_DET, d_model) * 0.02)  # per basis and detector
        self.query = torch.nn.Parameter(torch.randn(2, N_OBS, d_model) * 0.02)     # per basis and observable
        layer = torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True,
                                                 norm_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, 1)

    def forward(self, det, basis):
        """det: (B, 70) 0/1 long; basis: (B,) 0 for X, 1 for Z. Returns (B, 4) log-odds."""
        x = self.value(det) + self.position[basis]
        x = torch.cat([x, self.query[basis]], dim=1)
        x = self.encoder(x)
        return self.head(self.norm(x[:, N_DET:])).squeeze(-1)


# ----------------------------------------------------------------------------------------------------------------
# Decoding (the competition interface)
# ----------------------------------------------------------------------------------------------------------------
_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        ck = torch.load(WEIGHTS, map_location="cpu")
        _MODEL = TransformerDecoder(**ck["config"])
        _MODEL.load_state_dict(ck["state"])
        _MODEL.eval()
        _MODEL.circuits = ck["circuits"]
    return _MODEL


def _basis_of(circuit, model):
    text = str(circuit)
    for i, b in enumerate("XZ"):
        if model.circuits[b] == text:
            return i
    raise ValueError("circuit does not match the X or Z circuit the transformer was trained on")


@torch.no_grad()
def predict_logits(circuit, detector_events, batch=8192):
    model = _load()
    basis = _basis_of(circuit, model)
    det = np.asarray(detector_events, dtype=np.int64)
    out = []
    for i in range(0, len(det), batch):
        d = torch.as_tensor(det[i: i + batch])
        out.append(model(d, torch.full((len(d),), basis, dtype=torch.long)))
    return torch.cat(out).numpy()


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    det = np.asarray(detector_events, dtype=bool)
    packed = np.packbits(det, axis=1)
    uniq, inverse = np.unique(packed, axis=0, return_inverse=True)
    syns = np.unpackbits(uniq, axis=1, count=det.shape[1])
    pred = (predict_logits(circuit, syns) > 0).astype(np.uint8)
    return pred[inverse.reshape(-1)]


# ----------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------
def main():
    from hackathon_2026 import load_kaggle_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="data/train")
    ap.add_argument("--valid", default="data/train/train.csv")
    ap.add_argument("--valid-shots", type=int, default=20000, help="per basis, for the LER reported while training")
    ap.add_argument("--hours", type=float, default=2.0, help="training time budget")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=WEIGHTS)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    circuits = {b: stim.Circuit.from_file(os.path.join(args.circuits, f"{b}_circuit.stim")) for b in "XZ"}
    samplers = {b: circuits[b].compile_detector_sampler(seed=args.seed + i) for i, b in enumerate("XZ")}
    valid = load_kaggle_csv(args.valid)
    vdet, vbasis, vobs = [], [], []
    for i, b in enumerate("XZ"):
        rows = np.flatnonzero(valid["basis"] == b)[: args.valid_shots]
        vdet.append(valid["detector_events"][rows])
        vobs.append(valid["observable_flips"][rows])
        vbasis.append(np.full(len(rows), i))
    vdet = torch.as_tensor(np.concatenate(vdet), dtype=torch.long)
    vbasis = torch.as_tensor(np.concatenate(vbasis))
    vobs = torch.as_tensor(np.concatenate(vobs), dtype=torch.bool)

    model = TransformerDecoder(args.d_model, args.layers, args.heads, args.d_ff)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def batch():
        half = args.batch // 2
        dets, obss, bases = [], [], []
        for i, b in enumerate("XZ"):
            d, o = samplers[b].sample(half, separate_observables=True)
            dets.append(d)
            obss.append(o)
            bases.append(np.full(half, i))
        return (torch.as_tensor(np.concatenate(dets), dtype=torch.long), torch.as_tensor(np.concatenate(bases)),
                torch.as_tensor(np.concatenate(obss), dtype=torch.float32))

    # Measure the speed, then fix the number of steps so that the cosine schedule ends at the time budget.
    t0 = time.time()
    for _ in range(5):
        d, bs, o = batch()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(d, bs), o)
        opt.zero_grad()
        loss.backward()
    per_step = (time.time() - t0) / 5
    steps = max(100, int(args.hours * 3600 / (per_step * 1.05)))
    warmup = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    print(f"{n_params} parameters, {per_step * 1000:.0f} ms per batch of {args.batch} -> {steps} steps "
          f"({steps * args.batch / 1e6:.1f}M shots) in {args.hours} h", flush=True)

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(vdet[i: i + 8192], vbasis[i: i + 8192]) for i in range(0, len(vdet), 8192)])
        model.train()
        wrong = (logits > 0) != vobs
        nx = int((vbasis == 0).sum())
        return wrong[:nx].float().mean().item(), wrong[nx:].float().mean().item(), wrong.float().mean().item()

    t0 = time.time()
    running = 0.0
    best = (1.0, None)
    for step in range(1, steps + 1):
        d, bs, o = batch()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(d, bs), o)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % args.eval_every == 0 or step == steps:
            lx, lz, l = evaluate()
            print(f"step {step:6d}/{steps}  shots {step * args.batch / 1e6:6.2f}M  loss {running:.4f}  "
                  f"LER X {lx:.4f} Z {lz:.4f} both {l:.4f}  lr {sched.get_last_lr()[0]:.1e}  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            if l < best[0]:
                best = (l, {k: v.clone() for k, v in model.state_dict().items()})
                torch.save(dict(config=model.config, state=best[1],
                                circuits={b: str(c) for b, c in circuits.items()}), args.out)
    print(f"best validation LER {best[0]:.4f}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
