"""Per-sample geometry dump of the subspace map — the data the geometry analysis needs.

map_subspace.py samples the behaviour-relevant k-D box and decodes each point, but only
saves aggregate metrics; the per-sample (subspace coords -> achieved distribution) pairs
are discarded, which makes every metric-level geometry question (isometry, foliation,
Jacobian field, density overlay) unanswerable offline. This script re-runs the box
sampling at several layers and DUMPS the pairs.

Per layer: geometry_L{L}.npz with
  samp_coords (S, k)   sampled subspace coordinates (uniform box, margin*span around anchors)
  samp_dists  (S, n+1) achieved concept distribution per sample
  valid       (S,)     mass-filter validity
  anchor_coords (M, k) anchors in the same coordinates
  anchor_dists  (M, n+1)
  Q (k, H), mu (H), sd (H), c (H)  the subspace chart (standardised space) for reproducibility
  cca_corr (k,)        canonical correlations of the chart

Run from repo root (GPU + activations.safetensors).
"""

from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layers", default="19,23,27,31")
    ap.add_argument("--carrier", default="A day of the week:")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=30000)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--mode", default="box", choices=["box", "gauss"],
                    help="box = uniform over the margin box (far-field); gauss = Gaussian around "
                         "random anchors with sigma = sigma-nn * median anchor NN spacing (core)")
    ap.add_argument("--sigma-nn", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="extra filename suffix (e.g. seed1)")
    ap.add_argument("--keep", default="",
                    help="comma abbrs: build the chart from THESE values' anchors only (sparse "
                         "chart); all anchors still saved for reference")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, _ = get_concept(args.concept); n = len(tokens)
    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat); model = pipe.model; model.eval()
    variant_ids, canonical_ids, _ = build_concept_token_ids(pipe.tokenizer, tokens)
    dev = next(model.parameters()).device
    holder = {"H": None}

    def hook(m, i, o):
        hs = (o[0] if isinstance(o, tuple) else o).clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    from safetensors.torch import load_file
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    P = dists0[ridx]
    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])

    for L in [int(x) for x in args.layers.split(",")]:
        hook_handle = model.model.layers[L - 1].register_forward_hook(hook)
        X_all_anchors = blob["activations"][:, L, :].float().numpy()[ridx]
        if args.keep:
            _, ABBR2 = get_concept(args.concept)
            keep_ids = [ABBR2.index(x) for x in args.keep.split(",")]
            kmask = np.isin(P[:, :n].argmax(1), keep_ids)
            X = X_all_anchors[kmask]; Pc = P[kmask]
            print(f"L{L}: sparse chart from {kmask.sum()} anchors (keep={args.keep})")
        else:
            kmask = np.ones(len(X_all_anchors), bool)
            X = X_all_anchors; Pc = P
        mu = X.mean(0); sd = X.std(0) + 1e-6; Xz = (X - mu) / sd
        Y = np.sqrt(np.clip(Pc, 0, None))
        npc = min(40, Xz.shape[0] - 1)
        apca = PCA(n_components=npc).fit(Xz); Xp = apca.transform(Xz)
        k = min(args.k, npc, Y.shape[1])
        cca = CCA(n_components=k, max_iter=1000).fit(Xp, Y)
        Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T
        Xc, Yc = cca.transform(Xp, Y)
        ccorr = np.array([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(k)], dtype=np.float32)
        c = Xz.mean(0); coords = (Xz - c) @ Q.T
        if args.mode == "gauss":
            pw = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            nn = float(np.median(np.sort(pw, axis=1)[:, 1]))
            base = coords[rng.integers(0, len(coords), size=args.n_samples)]
            samp = base + rng.normal(0, args.sigma_nn * nn / np.sqrt(k), size=(args.n_samples, k))
        else:
            lo = coords.min(0); hi = coords.max(0); span = hi - lo
            samp = rng.uniform(lo - args.margin * span, hi + args.margin * span, size=(args.n_samples, k))
        Xraw = ((c[None] + samp @ Q) * sd + mu).astype(np.float32)

        res = []
        with torch.no_grad():
            for s in range(0, len(Xraw), args.batch):
                hb = Xraw[s:s + args.batch]; B = hb.shape[0]
                holder["H"] = torch.from_numpy(hb).to(dev)
                o = model(input_ids=carrier_enc["input_ids"].repeat(B, 1),
                          attention_mask=carrier_enc["attention_mask"].repeat(B, 1),
                          use_cache=False, logits_to_keep=1)
                dd, _ = concept_distributions(F.softmax(o.logits[:, -1, :].float(), -1).cpu(),
                                              variant_ids, canonical_ids)
                res.append(dd.numpy())
        dd = np.concatenate(res, 0)
        valid = (dd[:, :n].sum(1) >= 0.90) & (dd[:, n] <= 0.10)
        hook_handle.remove()

        sfx = ("_core" if args.mode == "gauss" else "") + (("_" + args.tag) if args.tag else "")
        out = os.path.join(args.experiment_root, f"geometry{sfx}_L{L}.npz")
        all_coords = (((X_all_anchors - mu) / sd) - c) @ Q.T   # ALL anchors in this chart
        np.savez_compressed(out,
                            samp_coords=samp.astype(np.float32), samp_dists=dd.astype(np.float32),
                            valid=valid, anchor_coords=all_coords.astype(np.float32),
                            anchor_dists=P.astype(np.float32), kept_mask=kmask,
                            Q=Q.astype(np.float32),
                            mu=mu.astype(np.float32), sd=sd.astype(np.float32), c=c.astype(np.float32),
                            cca_corr=ccorr, layer=np.array([L]), k=np.array([k]),
                            margin=np.array([args.margin]))
        print(f"L{L}: dumped {len(samp)} samples (valid {valid.mean():.2f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
