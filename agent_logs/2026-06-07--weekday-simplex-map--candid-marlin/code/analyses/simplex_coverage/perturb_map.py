"""Map the weekday activation subspace by perturbing/walking around anchors.

Step 3 of weekday-simplex-map. The 194 retained prompts are ANCHORS: each is a
point in the activation region that decodes to a valid weekday distribution. Here
we explore that region directly — perturb the last-token residual at a chosen layer
around each anchor, patch it back through the rest of the model, and test whether
the output is still a VALID weekday distribution (Σweekday >= 0.90 AND other <= 0.10).

Two exploration modes:
  radial : from each anchor, step along random directions at increasing radii;
           record the validity boundary and how far behaviour moves while valid.
  walk   : cumulative random walk; record how many steps stay valid.

Direction subspaces (to address "is PCA necessary?"):
  subspace : random directions inside the anchor-spanned subspace (PCA of anchors,
             top-d) — the region the behaviour states actually occupy.
  full     : random directions in raw R^4096 — CONTROL; expected to break validity
             almost immediately if the subspace restriction matters.

Scale: radii/steps are in units of the median anchor-to-anchor L2 distance at the
layer (so "1" ~ moving toward a neighbouring belief state), not absolute L2.

Run from repo root (needs the GPU + activations.safetensors from capture step):
  uv run python <...>/perturb_map.py --layer 31 --mode radial \
     --prompts <...>/prompts.json --experiment-root <...>/simplex_coverage
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
    build_concept_token_ids,
    concept_distributions,
    frame_uses_chat,
    get_concept,
    load_model,
    wrap,
)


def hellinger(p, q):
    return float(np.sqrt(np.clip(((np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))) ** 2).sum(), 0, None)) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layer", type=int, default=31, help="CAPTURED hidden-state index (1..32); hooks decoder layer L-1")
    ap.add_argument("--mode", choices=["radial", "walk", "shape"], default="radial")
    ap.add_argument("--n-anchors", type=int, default=32, help="-1 = all retained")
    ap.add_argument("--n-dirs", type=int, default=16, help="random directions per anchor per dir-type")
    ap.add_argument("--dir-types", default="both", choices=["both", "subspace", "full"])
    ap.add_argument("--subspace-dims", type=int, default=20)
    ap.add_argument("--radii", default="0,0.5,1,2,4,8", help="radial: radii in NN-distance units")
    ap.add_argument("--walk-steps", type=int, default=40)
    ap.add_argument("--step-size", type=float, default=0.5, help="walk: per-step size in NN-distance units")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    with open(args.prompts) as f:
        prompts = json.load(f)["prompts"]

    tokens, ABBR = get_concept(args.concept)
    n = len(tokens)
    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat)
    tok = pipe.tokenizer
    model = pipe.model
    model.eval()
    variant_ids, canonical_ids, _ = build_concept_token_ids(tok, tokens)

    # ---- load captured anchors at this layer ----
    from safetensors.torch import load_file

    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    acts = blob["activations"]                      # (N, L+1, H) bf16
    dists0 = blob["dists"].float().numpy()          # (N, 8)
    retained = blob["retained"].numpy().astype(bool)
    L = args.layer
    H = acts.shape[2]
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    anchor_acts = acts[:, L, :].float().numpy()     # (N, H)
    ridx = np.where(retained)[0]
    A = anchor_acts[ridx]                           # (M, H)
    M = A.shape[0]

    # anchor-spanned subspace basis (PCA of anchors), and NN-distance scale
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist

    mean = A.mean(0)
    pca = PCA(n_components=min(args.subspace_dims, M - 1)).fit(A - mean)
    V = pca.components_                              # (d, H) orthonormal rows
    D = cdist(A, A); np.fill_diagonal(D, np.inf)
    nn = float(np.median(D.min(1)))                 # median nearest-neighbour L2
    print(f"layer L{L}: M={M} anchors, subspace dims={V.shape[0]} "
          f"(EVR {pca.explained_variance_ratio_.sum():.2f}), NN-dist={nn:.2f}")

    # decoder module to hook: captured index L -> layers[L-1]
    layers = model.model.layers
    dl = L - 1
    holder = {"H": None}

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        if isinstance(out, tuple):
            return (hs,) + tuple(out[1:])
        return hs

    handle = layers[dl].register_forward_hook(hook)

    @torch.no_grad()
    def patched_dists(prompt_core, Hprime_np):
        # Hprime_np: (B, H). Replicate the anchor's own prompt B times, patch last token.
        enc = pipe.load([{"raw_input": wrap(prompt_core, args.frame)}])
        B = Hprime_np.shape[0]
        ids = enc["input_ids"].repeat(B, 1)
        am = enc["attention_mask"].repeat(B, 1)
        holder["H"] = torch.from_numpy(Hprime_np).to(dev)
        out = model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1)
        last = out.logits[:, -1, :].float()
        probs = F.softmax(last, dim=-1).cpu()
        dd, _ = concept_distributions(probs, variant_ids, canonical_ids)
        return dd.numpy()                            # (B, 8)

    def rand_dirs(n, kind):
        if kind == "subspace":
            g = rng.standard_normal((n, V.shape[0]))
            d = g @ V                                # (n, H)
        else:
            d = rng.standard_normal((n, H))
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        return d

    sel = ridx if args.n_anchors < 0 else ridx[rng.choice(M, size=min(args.n_anchors, M), replace=False)]
    dir_types = ["subspace", "full"] if args.dir_types == "both" else [args.dir_types]

    results = {"model": args.model, "layer": L, "mode": args.mode, "nn_dist": nn,
               "subspace_dims": int(V.shape[0]), "n_anchors": len(sel), "records": []}

    # sanity: zero-perturbation reproduces the anchor distribution
    a0 = ridx[0]
    z = patched_dists(prompts[a0]["text_core"], anchor_acts[a0:a0 + 1])
    sane = hellinger(z[0], dists0[a0])
    results["zero_perturbation_hellinger_to_anchor"] = round(sane, 4)
    print(f"sanity: zero-perturbation Hellinger to stored anchor dist = {sane:.4f} (should be ~0)")

    if args.mode == "radial":
        radii = [float(x) for x in args.radii.split(",")]
        # agg[(kind, radius)] = [n_valid, n_total, sum_hellinger_valid]
        agg = {(k, r): [0, 0, 0.0] for k in dir_types for r in radii}
        for ai in sel:
            core = prompts[ai]["text_core"]
            h = anchor_acts[ai]                      # (H,)
            p_anchor = dists0[ai]
            for kind in dir_types:
                dirs = rand_dirs(args.n_dirs, kind)  # (n_dirs, H)
                # build batch over (dir, radius)
                Hp = []
                key = []
                for r in radii:
                    Hp.append(h[None, :] + (r * nn) * dirs)
                    key += [(kind, r)] * args.n_dirs
                Hp = np.concatenate(Hp, 0).astype(np.float32)
                dd = patched_dists(core, Hp)
                for j, (k, r) in enumerate(key):
                    wm = dd[j, :n].sum(); other = dd[j, n]
                    valid = (wm >= 0.90) and (other <= 0.10)
                    a = agg[(k, r)]
                    a[1] += 1
                    if valid:
                        a[0] += 1
                        a[2] += hellinger(dd[j], p_anchor)
        for (k, r), (nv, nt, sh) in agg.items():
            results["records"].append({
                "dir_type": k, "radius_nn": r,
                "validity_rate": round(nv / max(nt, 1), 3),
                "mean_hellinger_to_anchor_valid": round(sh / max(nv, 1), 3),
                "n": nt,
            })
    elif args.mode == "walk":
        steps = args.walk_steps
        for kind in dir_types:
            # frac of walks still valid at each step
            still = np.zeros(steps + 1)
            tot = 0
            for ai in sel:
                core = prompts[ai]["text_core"]
                for _w in range(args.n_dirs):       # n_dirs independent walks per anchor
                    h = anchor_acts[ai].copy()
                    traj = [h.copy()]
                    for _s in range(steps):
                        d = rand_dirs(1, kind)[0]
                        h = h + (args.step_size * nn) * d
                        traj.append(h.copy())
                    dd = patched_dists(core, np.stack(traj).astype(np.float32))
                    valid = (dd[:, :n].sum(1) >= 0.90) & (dd[:, n] <= 0.10)
                    # first invalid step truncates the "still valid" run
                    run = np.cumprod(valid.astype(int))
                    still += run
                    tot += 1
            results["records"].append({
                "dir_type": kind,
                "frac_valid_by_step": [round(float(s / max(tot, 1)), 3) for s in still],
                "n_walks": tot,
            })

    else:  # shape — record simplex point + naturalness for valid in-subspace perturbations
        radii = [float(x) for x in args.radii.split(",") if float(x) > 0]
        A_sub = (A - mean) @ V.T                       # anchor subspace coords (M, d)
        var = np.clip(pca.explained_variance_, 1e-6, None)
        rec_p, rec_anchor, rec_rad, rec_hell, rec_nat_nn, rec_maha = [], [], [], [], [], []
        for ai in sel:
            core = prompts[ai]["text_core"]
            h = anchor_acts[ai]; p_anchor = dists0[ai]
            dirs = rand_dirs(args.n_dirs, "subspace")  # in-subspace only
            Hp, rads = [], []
            for r in radii:
                Hp.append(h[None, :] + (r * nn) * dirs); rads += [r] * args.n_dirs
            Hp = np.concatenate(Hp, 0).astype(np.float32)
            dd = patched_dists(core, Hp)               # (B, 8)
            Hp_sub = (Hp - mean) @ V.T                 # (B, d)
            nn_dist = cdist(Hp_sub, A_sub).min(1) / nn  # nearest-anchor (NN-units)
            maha = np.sqrt((Hp_sub ** 2 / var).sum(1))  # Mahalanobis in subspace
            for j in range(Hp.shape[0]):
                if (dd[j, :n].sum() >= 0.90) and (dd[j, n] <= 0.10):
                    rec_p.append(dd[j]); rec_anchor.append(int(ai)); rec_rad.append(rads[j])
                    rec_hell.append(hellinger(dd[j], p_anchor))
                    rec_nat_nn.append(float(nn_dist[j])); rec_maha.append(float(maha[j]))
        anchor_idx = np.array(sel)
        np.savez(
            os.path.join(args.experiment_root, f"region_shape_L{L}.npz"),
            pprime=np.array(rec_p, dtype=np.float32),
            anchor=np.array(rec_anchor),
            radius=np.array(rec_rad, dtype=np.float32),
            hellinger=np.array(rec_hell, dtype=np.float32),
            nat_nn=np.array(rec_nat_nn, dtype=np.float32),
            nat_maha=np.array(rec_maha, dtype=np.float32),
            anchor_pprime=dists0[anchor_idx].astype(np.float32),
            anchor_argmax=np.array([int(dists0[i, :n].argmax()) for i in anchor_idx]),
            nn_dist=np.array([nn], dtype=np.float32),
        )
        results["records"].append({
            "n_valid_perturbations": len(rec_p),
            "n_total": len(sel) * args.n_dirs * len(radii),
            "radii": radii,
            "mean_naturalness_nn": round(float(np.mean(rec_nat_nn)), 3) if rec_p else None,
        })
        print(f"shape: kept {len(rec_p)} valid in-subspace perturbations -> region_shape_L{L}.npz")

    handle.remove()
    out_path = os.path.join(args.experiment_root, f"perturb_{args.mode}_L{L}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["records"], indent=2)[:2000])
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
