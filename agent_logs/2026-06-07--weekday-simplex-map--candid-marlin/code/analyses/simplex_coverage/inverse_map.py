"""Inverse-map / sharp onto-test.

For a set of TARGET simplex points (vertices, 2-token edges, interior mixtures), find an
in-subspace activation that produces each, and measure the achievable behavioural error
(Hellinger to target). Turns "the map is onto" into an explicit behaviour->activation
inverse + a precise completeness number (which targets are reachable, how well).

Method: fit a linear forward map (subspace coords -> sqrt behaviour) on the anchors, invert
it (pseudo-inverse) to propose coords for each target, reconstruct + patch to read the true
achieved distribution, then refine with a few random local steps (keep best). Reports
achieved Hellinger per target, grouped by target type.

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


def make_targets(n, rng):
    """List of (type, dist over n+1 with other=0)."""
    T = []
    for d in range(n):  # vertices
        v = np.zeros(n + 1); v[d] = 1.0; T.append(("vertex", v))
    for i in range(n):  # 2-token edges (50/50)
        for j in range(i + 1, n):
            v = np.zeros(n + 1); v[i] = v[j] = 0.5; T.append(("edge", v))
    v = np.zeros(n + 1); v[:n] = 1.0 / n; T.append(("uniform", v))  # centroid
    for _ in range(60):  # interior Dirichlet mixtures
        d = rng.dirichlet(np.ones(n)); v = np.zeros(n + 1); v[:n] = d; T.append(("interior", v))
    for _ in range(40):  # sparser mixtures (2-3 tokens)
        d = rng.dirichlet(0.3 * np.ones(n)); v = np.zeros(n + 1); v[:n] = d; T.append(("sparse", v))
    return T


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
    ap.add_argument("--refine-steps", type=int, default=48)
    ap.add_argument("--refine-scale", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
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
    Y = np.sqrt(np.clip(P, 0, None))                     # (M, n+1) sqrt behaviour

    # behaviour-relevant subspace (CCA), orthonormal Q (k,H) std space
    apca = PCA(n_components=min(40, Xz.shape[0] - 1)).fit(Xz); Xp = apca.transform(Xz)
    kk = min(args.k, Xp.shape[1], Y.shape[1])
    cca = CCA(n_components=kk, max_iter=1000).fit(Xp, Y)
    Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T
    c = Xz.mean(0); coords = (Xz - c) @ Q.T                # (M, k)

    # linear forward map: coords -> sqrt behaviour ;  invert via pinv
    Cb = np.concatenate([coords, np.ones((len(coords), 1))], 1)  # (M, k+1)
    Wb, *_ = np.linalg.lstsq(Cb, Y, rcond=None)            # (k+1, n+1)
    W = Wb[:-1]; b = Wb[-1]
    Wpinv = np.linalg.pinv(W)                              # (n+1, k)

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

    targets = make_targets(n, rng)
    sqrtT = np.stack([np.sqrt(t) for _, t in targets])     # (T, n+1)
    coords0 = (sqrtT - b) @ Wpinv                          # (T, k) linear-inverse proposals
    span = coords.max(0) - coords.min(0)

    achieved = decode(coords0)
    best_h = np.array([hell(achieved[i], targets[i][1]) for i in range(len(targets))])
    best_c = coords0.copy()
    # light random local refinement (keep best per target)
    for _ in range(args.refine_steps // 8):
        cand = best_c + rng.normal(0, args.refine_scale, size=best_c.shape) * span[None, :]
        ach = decode(cand)
        h = np.array([hell(ach[i], targets[i][1]) for i in range(len(targets))])
        imp = h < best_h
        best_h[imp] = h[imp]; best_c[imp] = cand[imp]
    handle.remove()

    types = sorted(set(t for t, _ in targets))
    summary = {"concept": args.concept, "layer": L, "k": int(kk), "n_targets": len(targets)}
    for ty in types:
        hs = best_h[[i for i, (t, _) in enumerate(targets) if t == ty]]
        summary[ty] = {"n": int(len(hs)), "median_hellinger": round(float(np.median(hs)), 3),
                       "p90_hellinger": round(float(np.percentile(hs, 90)), 3),
                       "frac_reached_0.10": round(float((hs <= 0.10).mean()), 3),
                       "frac_reached_0.20": round(float((hs <= 0.20).mean()), 3)}
    summary["overall_median_hellinger"] = round(float(np.median(best_h)), 3)
    summary["overall_frac_reached_0.20"] = round(float((best_h <= 0.20).mean()), 3)
    with open(os.path.join(args.experiment_root, f"inverse_map_L{L}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # figure: achieved Hellinger by target type
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig_dir = os.path.join(args.experiment_root, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [best_h[[i for i, (t, _) in enumerate(targets) if t == ty]] for ty in types]
    ax.boxplot(data, labels=types, showmeans=True)
    ax.axhline(0.1, color="g", ls="--", alpha=0.6, label="Hellinger 0.1"); ax.axhline(0.2, color="orange", ls="--", alpha=0.6, label="0.2")
    ax.set_title(f"Inverse map: achievable error to target simplex points (L{L}, k={kk})")
    ax.set_ylabel("achieved Hellinger to target"); ax.legend()
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"inverse_map_L{L}.{e}"))
    print(f"saved -> inverse_map_L{L}.json")


if __name__ == "__main__":
    main()
