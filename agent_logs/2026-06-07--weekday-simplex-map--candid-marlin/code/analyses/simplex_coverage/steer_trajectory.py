"""Steering trajectory demo — the payoff: continuously steer behaviour via the subspace map.

Define a target PATH through behaviour space (e.g. walk the weekday cycle Mon->Tue->...,
the month/season cycle, or a colour rainbow Red->Orange->...->Purple). For each waypoint
distribution we invert the behaviour<->activation map to an in-subspace activation, patch it
in, and read the ACHIEVED distribution. If the achieved trajectory tracks the target smoothly,
we can steer the model continuously across the gamut by walking activation space — including
through interpolated (unprompted) belief states.

Reports per-waypoint fidelity Hellinger(achieved,target), trajectory smoothness
(Hellinger between consecutive achieved points), validity, and whether the achieved arg-max
sweeps through the intended token order. Saves a 2-D Hellinger-PCA trajectory figure
(target path vs achieved path).

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


def hell(p, q):
    return float(np.sqrt(((np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))) ** 2).sum()) / math.sqrt(2))


def build_path(n, order, steps_per_edge):
    """Sequence of target distributions (n+1, other=0) interpolating vertex order in sqrt space."""
    verts = []
    for d in order:
        v = np.zeros(n + 1); v[d] = 1.0; verts.append(v)
    path, labels = [], []
    for a in range(len(verts) - 1):
        sa, sb = np.sqrt(verts[a]), np.sqrt(verts[a + 1])
        for s in range(steps_per_edge):
            al = s / steps_per_edge
            sq = (1 - al) * sa + al * sb
            sq = sq / (np.linalg.norm(sq) + 1e-9)
            path.append(sq ** 2); labels.append((order[a], order[a + 1], round(al, 2)))
    path.append(verts[-1]); labels.append((order[-1], order[-1], 1.0))
    return np.array(path), labels


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
    ap.add_argument("--order", default="", help="comma abbrs for the path; default = cyclic 0..n-1,0")
    ap.add_argument("--steps-per-edge", type=int, default=8)
    ap.add_argument("--refine-steps", type=int, default=40)
    ap.add_argument("--refine-scale", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    order = [ABBR.index(x) for x in args.order.split(",")] if args.order else list(range(n)) + [0]
    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat); model = pipe.model; model.eval()
    variant_ids, canonical_ids, _ = build_concept_token_ids(pipe.tokenizer, tokens)
    dev = next(model.parameters()).device
    L = args.layer; holder = {"H": None}

    def hook(m, i, o):
        hs = (o[0] if isinstance(o, tuple) else o).clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
    handle = model.model.layers[L - 1].register_forward_hook(hook)

    from safetensors.torch import load_file
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    A = blob["activations"][:, L, :].float().numpy()
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    X = A[ridx]; P = dists0[ridx]
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xz = (X - mu) / sd
    Y = np.sqrt(np.clip(P, 0, None))
    apca = PCA(n_components=min(40, Xz.shape[0] - 1)).fit(Xz); Xp = apca.transform(Xz)
    kk = min(args.k, Xp.shape[1], Y.shape[1])
    cca = CCA(n_components=kk, max_iter=1000).fit(Xp, Y)
    Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T
    c = Xz.mean(0); coords = (Xz - c) @ Q.T
    Cb = np.concatenate([coords, np.ones((len(coords), 1))], 1)
    Wb, *_ = np.linalg.lstsq(Cb, Y, rcond=None); W = Wb[:-1]; b = Wb[-1]
    Wpinv = np.linalg.pinv(W); span = coords.max(0) - coords.min(0)

    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])

    @torch.no_grad()
    def decode(coords_arr):
        Xraw = ((c[None] + coords_arr @ Q) * sd + mu).astype(np.float32)
        res = []
        for s in range(0, len(Xraw), args.batch):
            hb = Xraw[s:s + args.batch]; B = hb.shape[0]
            holder["H"] = torch.from_numpy(hb).to(dev)
            o = model(input_ids=carrier_enc["input_ids"].repeat(B, 1),
                      attention_mask=carrier_enc["attention_mask"].repeat(B, 1), use_cache=False, logits_to_keep=1)
            dd, _ = concept_distributions(F.softmax(o.logits[:, -1, :].float(), -1).cpu(), variant_ids, canonical_ids)
            res.append(dd.numpy())
        return np.concatenate(res, 0)

    targets, labels = build_path(n, order, args.steps_per_edge)
    coords0 = (np.sqrt(targets) - b) @ Wpinv
    achieved = decode(coords0)
    best_h = np.array([hell(achieved[i], targets[i]) for i in range(len(targets))])
    best_c = coords0.copy(); best_a = achieved.copy()
    for _ in range(args.refine_steps // 8):
        cand = best_c + rng.normal(0, args.refine_scale, size=best_c.shape) * span[None, :]
        ach = decode(cand); h = np.array([hell(ach[i], targets[i]) for i in range(len(targets))])
        imp = h < best_h
        best_h[imp] = h[imp]; best_c[imp] = cand[imp]; best_a[imp] = ach[imp]
    handle.remove()

    valid = (best_a[:, :n].sum(1) >= 0.90) & (best_a[:, n] <= 0.10)
    amax_ach = best_a[:, :n].argmax(1)
    smooth = [hell(best_a[i], best_a[i + 1]) for i in range(len(best_a) - 1)]
    # does the achieved argmax sweep through the intended vertex order?
    seq = [int(amax_ach[i]) for i in range(len(amax_ach))]
    intended_seq = [order[min(range(len(order)), key=lambda j: abs(j*args.steps_per_edge - i))] for i in range(len(seq))]
    out = {
        "concept": args.concept, "layer": L, "k": int(kk),
        "order": [ABBR[d] for d in order], "n_waypoints": len(targets),
        "fidelity_median_hellinger": round(float(np.median(best_h)), 3),
        "fidelity_p90_hellinger": round(float(np.percentile(best_h, 90)), 3),
        "frac_valid": round(float(valid.mean()), 3),
        "smoothness_median_step": round(float(np.median(smooth)), 3),
        "smoothness_max_step": round(float(np.max(smooth)), 3),
        "achieved_argmax_seq": [ABBR[s] for s in seq],
    }
    tag = args.tag or f"L{L}k{kk}"
    with open(os.path.join(args.experiment_root, f"steer_{args.concept}_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "achieved_argmax_seq"}, indent=2))
    print("argmax sweep:", " -> ".join(out["achieved_argmax_seq"]))

    # figure: 2-D Hellinger-PCA, target path vs achieved
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sA = np.sqrt(np.clip(P, 0, None)); mA = sA.mean(0)
    _, _, Vt = np.linalg.svd(sA - mA, full_matrices=False); comps = Vt[:2]
    def pl(d): return (np.sqrt(np.clip(d, 0, None)) - mA) @ comps.T
    TT = pl(targets); AA = pl(best_a); ANC = pl(P)
    fig_dir = os.path.join(args.experiment_root, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(ANC[:, 0], ANC[:, 1], s=14, c="lightgray", label="anchors")
    ax.plot(TT[:, 0], TT[:, 1], "-", color="k", alpha=0.5, label="target path")
    sc = ax.scatter(AA[:, 0], AA[:, 1], s=26, c=np.arange(len(AA)), cmap="rainbow", label="achieved (steered)")
    ax.scatter(TT[::args.steps_per_edge, 0], TT[::args.steps_per_edge, 1], s=70, facecolors="none", edgecolors="k")
    ax.set_title(f"Steering {args.concept} along {'->'.join(out['order'])} (L{L}, k={kk})\n"
                 f"fidelity {out['fidelity_median_hellinger']}, smoothness {out['smoothness_median_step']}, valid {out['frac_valid']:.0%}")
    ax.set_xlabel("Hellinger-PC1"); ax.set_ylabel("Hellinger-PC2"); ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="position along path")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"steer_{args.concept}_{tag}.{e}"))
    print(f"saved -> steer_{args.concept}_{tag}.json")


if __name__ == "__main__":
    main()
