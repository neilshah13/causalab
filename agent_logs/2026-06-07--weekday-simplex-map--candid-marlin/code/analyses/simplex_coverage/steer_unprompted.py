"""Steer to UNPROMPTED concept values — the sparse-anchor steering test, done properly.

Build the subspace chart from the anchors of a sparse KEEP set only (e.g. {Red,Green,Blue}),
then steer to each HELD-OUT value's vertex target two ways:
  (1) linear-inverse proposal (the old sparse_recovery could only random-sample the box);
  (2) gradient refinement through the frozen model (Adam on the kept-chart coords).
Records, per held-out token: best achieved P(token), Hellinger to its vertex, and (for colours)
the comparison against the token's natural ceiling happens offline (color_ceilings.json).

This is the direct test of "steer the model to produce concept values that were never in the
anchor set" — distinct from sparse_recovery's blind box sampling, which conflates 'the box
doesn't contain it' with 'the map cannot reach it'.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="colors")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--carrier", default="A color:")
    ap.add_argument("--keep", required=True, help="comma abbrs whose anchors are KEPT (chart source)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--method", default="pca", choices=["pca", "cca"],
                    help="chart from kept anchors: PCA of activations (uncapped) or CCA")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--opt-batch", type=int, default=48)
    ap.add_argument("--init", default="both", choices=["linear", "box", "both"],
                    help="gradient init: linear-inverse proposal, best-of-box exploration, or the "
                         "better of the two per target (gradient alone stalls in flat softmax "
                         "regions; box exploration finds the basin, gradient sharpens it)")
    ap.add_argument("--box-samples", type=int, default=6000)
    ap.add_argument("--box-margin", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    keep_names = args.keep.split(",")
    keep_ids = [ABBR.index(x) for x in keep_names]
    held_ids = [d for d in range(n) if d not in keep_ids]
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
    A = blob["activations"][:, L, :].float().numpy()
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    X_all = A[ridx]; P_all = dists0[ridx]
    amax = P_all[:, :n].argmax(1)
    kept = np.isin(amax, keep_ids)
    X = X_all[kept]; P = P_all[kept]
    Mk = len(X)
    print(f"chart from {Mk} kept anchors ({args.keep}); steering to {len(held_ids)} held-out values")

    mu = X.mean(0); sd = X.std(0) + 1e-6; Xz = (X - mu) / sd
    Y = np.sqrt(np.clip(P, 0, None))
    k = min(args.k, Mk - 1)
    if args.method == "pca":
        pca = PCA(n_components=k).fit(Xz)
        Q = pca.components_                                   # (k, H)
    else:
        npc = min(40, Mk - 1)
        apca = PCA(n_components=npc).fit(Xz)
        kk = min(k, npc, Y.shape[1])
        cca = CCA(n_components=kk, max_iter=1000).fit(apca.transform(Xz), Y)
        Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T
        k = Q.shape[0]
    c = Xz.mean(0); coords = (Xz - c) @ Q.T
    Cb = np.concatenate([coords, np.ones((len(coords), 1))], 1)
    Wb, *_ = np.linalg.lstsq(Cb, Y, rcond=None); W = Wb[:-1]; b = Wb[-1]
    Wpinv = np.linalg.pinv(W)

    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])
    vidx = [torch.tensor(v, dtype=torch.long, device=dev) for v in variant_ids]

    def concept_mass(probs):
        cols = [probs.index_select(1, vi).sum(1) if len(vi) else torch.zeros(probs.shape[0], device=dev)
                for vi in vidx]
        d = torch.stack(cols, 1)
        other = (1.0 - d.sum(1, keepdim=True)).clamp(min=0.0)
        return torch.cat([d, other], 1)

    # vertex targets for every held-out value
    T = np.zeros((len(held_ids), n + 1)); T[np.arange(len(held_ids)), held_ids] = 1.0
    sqrtT = np.sqrt(T)
    coords0 = (sqrtT - b) @ Wpinv
    box_best_p = np.zeros(len(held_ids))

    if args.init in ("box", "both"):
        # global exploration of the kept chart: best box sample per held target as gradient init
        lo = coords.min(0); hi = coords.max(0); span = hi - lo
        samp = rng.uniform(lo - args.box_margin * span, hi + args.box_margin * span,
                           size=(args.box_samples, Q.shape[0]))
        Xs = ((c[None] + samp @ Q) * sd + mu).astype(np.float32)
        dd_parts = []
        with torch.no_grad():
            for s in range(0, len(Xs), 256):
                hb = Xs[s:s + 256]; B = hb.shape[0]
                holder["H"] = torch.from_numpy(hb).to(dev)
                o = model(input_ids=carrier_enc["input_ids"].repeat(B, 1),
                          attention_mask=carrier_enc["attention_mask"].repeat(B, 1),
                          use_cache=False, logits_to_keep=1)
                holder["H"] = None
                dd, _ = concept_distributions(F.softmax(o.logits[:, -1, :].float(), -1).cpu(),
                                              variant_ids, canonical_ids)
                dd_parts.append(dd.numpy())
        DD = np.concatenate(dd_parts, 0)
        for i, hd in enumerate(held_ids):
            j = int(DD[:, hd].argmax())
            box_best_p[i] = float(DD[j, hd])
            if args.init == "box" or DD[j, hd] > 0.05:   # 'both': prefer box basin when it found mass
                coords0[i] = samp[j]
        print(f"box exploration: held with best-sample P>=0.5: {(box_best_p >= 0.5).sum()}/{len(held_ids)}")

    tQ = torch.tensor(Q, dtype=torch.float32, device=dev)
    tc = torch.tensor(c, dtype=torch.float32, device=dev)
    tsd = torch.tensor(sd, dtype=torch.float32, device=dev)
    tmu = torch.tensor(mu, dtype=torch.float32, device=dev)

    best_p = np.zeros(len(held_ids)); best_h = np.full(len(held_ids), np.inf)
    init_p = np.zeros(len(held_ids))
    for s0 in range(0, len(held_ids), args.opt_batch):
        sl = slice(s0, min(s0 + args.opt_batch, len(held_ids)))
        B = sl.stop - sl.start
        z = torch.tensor(coords0[sl], dtype=torch.float32, device=dev, requires_grad=True)
        tgt = torch.tensor(sqrtT[sl], dtype=torch.float32, device=dev)
        tok_idx = torch.tensor([held_ids[i] for i in range(s0, sl.stop)], dtype=torch.long, device=dev)
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
            hl2 = 0.5 * ((torch.sqrt(d.clamp(min=0) + 1e-12) - tgt) ** 2).sum(1)
            hl2.sum().backward()
            opt.step()
            with torch.no_grad():
                pt = d.gather(1, tok_idx[:, None]).squeeze(1).cpu().numpy()
                hn = torch.sqrt(hl2.clamp(min=0)).detach().cpu().numpy()
                if it == 0:
                    init_p[s0:sl.stop] = pt
                upd = pt > best_p[s0:sl.stop]
                best_p[s0:sl.stop][upd] = pt[upd]
                best_h[s0:sl.stop] = np.minimum(best_h[s0:sl.stop], hn)
            if it % 50 == 0:
                print(f"chunk {s0//args.opt_batch} it {it}: mean best P(held) "
                      f"{best_p[s0:sl.stop].mean():.3f}", flush=True)
    handle.remove()

    per = {ABBR[held_ids[i]]: {"P_linear_init": round(float(init_p[i]), 3),
                               "P_gradient": round(float(best_p[i]), 3),
                               "hellinger_to_vertex": round(float(best_h[i]), 3)}
           for i in range(len(held_ids))}
    for i, hd in enumerate(held_ids):
        per[ABBR[hd]]["P_box_best"] = round(float(box_best_p[i]), 3)
    out = {"concept": args.concept, "layer": L, "keep": args.keep, "method": args.method,
           "k": int(k), "n_kept_anchors": int(Mk), "steps": args.steps, "init": args.init,
           "n_held": len(held_ids),
           "held_reached_P50_gradient": int((best_p >= 0.5).sum()),
           "held_reached_P50_box": int((box_best_p >= 0.5).sum()),
           "held_reached_P50_linear": int((init_p >= 0.5).sum()),
           "per_held": per}
    tag = ("_" + args.tag) if args.tag else f"_{''.join(keep_names)[:24]}_L{L}{args.method}"
    path = os.path.join(args.experiment_root, f"steer_unprompted{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({kk: vv for kk, vv in out.items() if kk != "per_held"}, indent=2))
    print(f"saved -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
