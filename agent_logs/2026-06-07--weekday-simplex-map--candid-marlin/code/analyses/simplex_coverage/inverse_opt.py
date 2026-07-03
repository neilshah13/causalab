"""Gradient inverse optimiser — tighten the linear-inverse residual (Step 7 follow-up).

inverse_map.py inverts the linear forward map (subspace coords -> sqrt behaviour) and
refines with random local steps; interior/mixture targets plateau at Hellinger ~0.20,
leaving open whether that residual is the model's reachable-set boundary or just the
optimiser budget. This resolves it: optimise the subspace coords DIRECTLY by gradient
descent through the frozen model (the paper's pullback machinery, App. A.8, applied
per-target), so any remaining residual is a property of the model, not the inverse.

Per target q: z (k,) -> H = (c + z Q) * sd + mu patched at layer L -> p_hat ->
loss = squared Hellinger(p_hat, q). Adam on z, init = linear-inverse proposal.
Targets reuse inverse_map.make_targets (vertices / edges / uniform / Dirichlet / sparse).

Run from repo root (GPU + activations.safetensors).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "methods"))
from run_coverage import MODEL_HF  # noqa: E402
from concept_core import (  # noqa: E402
    build_concept_token_ids, concept_distributions, frame_uses_chat, get_concept, load_model, wrap,
)
from inverse_map import make_targets  # noqa: E402


def hell_np(p, q):
    return float(np.sqrt(((np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))) ** 2).sum()) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--carrier", default="A day of the week:")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--opt-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat); model = pipe.model; model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    variant_ids, canonical_ids, _ = build_concept_token_ids(pipe.tokenizer, tokens)
    dev = next(model.parameters()).device
    L = args.layer; holder = {"H": None}

    def hook(m, i, o):
        if holder["H"] is None:
            return o
        hs = (o[0] if isinstance(o, tuple) else o).clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
    handle = model.model.layers[L - 1].register_forward_hook(hook)

    from safetensors.torch import load_file
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    X = blob["activations"][:, L, :].float().numpy()
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    X = X[ridx]; P = dists0[ridx]
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xz = (X - mu) / sd
    Y = np.sqrt(np.clip(P, 0, None))
    apca = PCA(n_components=min(40, Xz.shape[0] - 1)).fit(Xz); Xp = apca.transform(Xz)
    kk = min(args.k, Xp.shape[1], Y.shape[1])
    cca = CCA(n_components=kk, max_iter=1000).fit(Xp, Y)
    Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T
    c = Xz.mean(0); coords = (Xz - c) @ Q.T
    Cb = np.concatenate([coords, np.ones((len(coords), 1))], 1)
    Wb, *_ = np.linalg.lstsq(Cb, Y, rcond=None); W = Wb[:-1]; b = Wb[-1]
    Wpinv = np.linalg.pinv(W)

    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])

    targets = make_targets(n, rng)
    sqrtT = np.stack([np.sqrt(t) for _, t in targets])
    coords0 = (sqrtT - b) @ Wpinv

    # torch constants for the differentiable chart
    tQ = torch.tensor(Q, dtype=torch.float32, device=dev)
    tc = torch.tensor(c, dtype=torch.float32, device=dev)
    tsd = torch.tensor(sd, dtype=torch.float32, device=dev)
    tmu = torch.tensor(mu, dtype=torch.float32, device=dev)
    # variant-id index tensors for differentiable concept aggregation
    vidx = [torch.tensor(v, dtype=torch.long, device=dev) for v in variant_ids]

    def concept_mass(probs):
        cols = [probs.index_select(1, vi).sum(1) if len(vi) else torch.zeros(probs.shape[0], device=dev)
                for vi in vidx]
        d = torch.stack(cols, 1)
        other = (1.0 - d.sum(1, keepdim=True)).clamp(min=0.0)
        return torch.cat([d, other], 1)                         # (B, n+1)

    best_h = np.full(len(targets), np.inf)
    best_d = np.zeros((len(targets), n + 1))
    trace = []
    for s0 in range(0, len(targets), args.opt_batch):
        sl = slice(s0, min(s0 + args.opt_batch, len(targets)))
        B = sl.stop - sl.start
        z = torch.tensor(coords0[sl], dtype=torch.float32, device=dev, requires_grad=True)
        tgt_sqrt = torch.tensor(sqrtT[sl], dtype=torch.float32, device=dev)
        ids = carrier_enc["input_ids"].repeat(B, 1)
        mask = carrier_enc["attention_mask"].repeat(B, 1)
        opt = torch.optim.Adam([z], lr=args.lr)
        for it in range(args.steps):
            opt.zero_grad(set_to_none=True)
            holder["H"] = (tc[None] + z @ tQ) * tsd[None] + tmu[None]
            out = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
            holder["H"] = None
            probs = F.softmax(out.logits[:, -1, :].float(), -1)
            d = concept_mass(probs)
            hl2 = 0.5 * ((torch.sqrt(d.clamp(min=0) + 1e-12) - tgt_sqrt) ** 2).sum(1)   # sq Hellinger
            loss = hl2.sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                h_now = torch.sqrt(hl2.clamp(min=0)).detach().cpu().numpy()
                dn = d.detach().cpu().numpy()
                for j in range(B):
                    gi = s0 + j
                    if h_now[j] < best_h[gi]:
                        best_h[gi] = h_now[j]; best_d[gi] = dn[j]
            if it % 25 == 0:
                trace.append({"chunk": s0 // args.opt_batch, "iter": it,
                              "mean_h": round(float(np.sqrt(hl2.detach().cpu().numpy()).mean()), 4)})
                print(f"chunk {s0//args.opt_batch} it {it}: mean Hellinger "
                      f"{np.sqrt(hl2.detach().cpu().numpy()).mean():.4f}", flush=True)
    handle.remove()

    types = sorted(set(t for t, _ in targets))
    summary = {"concept": args.concept, "layer": L, "k": int(kk), "steps": args.steps,
               "lr": args.lr, "n_targets": len(targets), "optimiser": "adam_through_model"}
    for ty in types:
        hs = best_h[[i for i, (t, _) in enumerate(targets) if t == ty]]
        summary[ty] = {"n": int(len(hs)), "median_hellinger": round(float(np.median(hs)), 3),
                       "p90_hellinger": round(float(np.percentile(hs, 90)), 3),
                       "frac_reached_0.10": round(float((hs <= 0.10).mean()), 3),
                       "frac_reached_0.20": round(float((hs <= 0.20).mean()), 3)}
    summary["overall_median_hellinger"] = round(float(np.median(best_h)), 3)
    summary["overall_frac_reached_0.20"] = round(float((best_h <= 0.20).mean()), 3)
    summary["trace"] = trace
    with open(os.path.join(args.experiment_root, f"inverse_opt_L{L}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))
    print(f"saved -> inverse_opt_L{L}.json")


if __name__ == "__main__":
    main()
