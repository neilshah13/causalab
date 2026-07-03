"""Sparse-anchor recovery — can the map reach UNPROMPTED regions? (steering completeness)

Steering use case: anchor a SPARSE subset of the concept (e.g. a few days / 10 colors),
build the activation subspace from only those, and ask whether walking that subspace
reaches VALID activations for the HELD-OUT (unprompted) regions (the other days / the
other 10 colors). If yes, we can prompt sparsely and steer the full continuum.

Sweeps:
  - subspace dim k (does giving more dimensions let us reach unprompted regions?)
  - subspace method: 'cca' (behaviour-supervised) vs 'pca' (pure activation variance) vs
    'diff' (span of anchor-difference vectors) — "different ways of doing PCA".

The subspace + sampling box are built ONLY from the kept anchors (deletion is pre-PCA);
the full anchor set is used solely for behaviour-plane bookkeeping (where held-out days live).

Recovery per held-out token d: max P(d) reached by any valid sample, # valid samples whose
argmax == d, and cell-coverage of d's region. "Recovered" = max P(d) >= --recover-thresh.

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


def subspace(method, Xz, Y, k, npc=40):
    """Return orthonormal basis (k,H) in standardised activation space."""
    from sklearn.decomposition import PCA
    npc = min(npc, Xz.shape[0] - 1)
    if method == "pca":
        return PCA(n_components=min(k, npc)).fit(Xz).components_
    if method == "diff":
        # random anchor-difference vectors -> PCA to k dims (spans the affine hull directions)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, Xz.shape[0], size=(min(2000, Xz.shape[0] * 4), 2))
        diffs = Xz[idx[:, 0]] - Xz[idx[:, 1]]
        return PCA(n_components=min(k, npc)).fit(diffs).components_
    # cca (behaviour-supervised)
    from sklearn.cross_decomposition import CCA
    apca = PCA(n_components=npc).fit(Xz)
    Xp = apca.transform(Xz)
    kk = min(k, npc, Y.shape[1])
    cca = CCA(n_components=kk, max_iter=1000).fit(Xp, Y)
    Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T)
    return Q.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--carrier", default="A day of the week:")
    ap.add_argument("--keep", required=True, help="comma abbrs to KEEP, e.g. Mon,Thu")
    ap.add_argument("--k-list", default="4,6,8,12,20,40")
    ap.add_argument("--methods", default="cca,pca,diff")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--recover-thresh", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    keep = [ABBR.index(x) for x in args.keep.split(",")]
    held = [d for d in range(n) if d not in keep]
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
    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    A = blob["activations"][:, L, :].float().numpy()
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    Aa = A[ridx]; P = dists0[ridx]; M = Aa.shape[0]
    amax = P[:, :n].argmax(1)
    keep_mask = np.isin(amax, keep)
    Xk = Aa[keep_mask]; Yk = np.sqrt(np.clip(P[keep_mask], 0, None))
    print(f"keep={args.keep} ({keep_mask.sum()} anchors), held-out days={[ABBR[d] for d in held]}")

    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])

    @torch.no_grad()
    def decode(Hraw):
        res = []
        for s in range(0, len(Hraw), args.batch):
            hb = Hraw[s:s + args.batch]; B = hb.shape[0]
            holder["H"] = torch.from_numpy(hb).to(dev)
            o = model(input_ids=carrier_enc["input_ids"].repeat(B, 1),
                      attention_mask=carrier_enc["attention_mask"].repeat(B, 1), use_cache=False, logits_to_keep=1)
            dd, _ = concept_distributions(F.softmax(o.logits[:, -1, :].float(), -1).cpu(), variant_ids, canonical_ids)
            res.append(dd.numpy())
        return np.concatenate(res, 0)

    mu = Xk.mean(0); sd = Xk.std(0) + 1e-6
    Xkz = (Xk - mu) / sd
    results = []
    for method in args.methods.split(","):
        for k in [int(x) for x in args.k_list.split(",")]:
            Q = subspace(method, Xkz, Yk, k)
            c = Xkz.mean(0); coords = (Xkz - c) @ Q.T
            lob = coords.min(0) - args.margin * (coords.max(0) - coords.min(0))
            hib = coords.max(0) + args.margin * (coords.max(0) - coords.min(0))
            samp = rng.uniform(lob, hib, size=(args.n_samples, Q.shape[0]))
            dd = decode(((c[None] + samp @ Q) * sd + mu).astype(np.float32))
            valid = (dd[:, :n].sum(1) >= 0.90) & (dd[:, n] <= 0.10)
            v = dd[valid]
            per_day = {}
            for d in held:
                maxp = float(v[:, d].max()) if len(v) else 0.0
                nam = int((v[:, :n].argmax(1) == d).sum()) if len(v) else 0
                per_day[ABBR[d]] = {"max_P": round(maxp, 3), "n_argmax": nam, "recovered": bool(maxp >= args.recover_thresh)}
            nrec = sum(1 for d in held if per_day[ABBR[d]]["recovered"])
            results.append({"method": method, "k": int(Q.shape[0]), "n_valid": int(valid.sum()),
                            "held_days_recovered": nrec, "n_held": len(held),
                            "frac_held_recovered": round(nrec / max(len(held), 1), 3), "per_day": per_day})
            print(f"  {method:4s} k={Q.shape[0]:2d}: valid={valid.sum():5d} recovered {nrec}/{len(held)} held-out days "
                  f"({', '.join(ABBR[d] for d in held if per_day[ABBR[d]]['recovered'])})")
    handle.remove()

    out = {"concept": args.concept, "layer": L, "keep": args.keep, "held": [ABBR[d] for d in held],
           "margin": args.margin, "recover_thresh": args.recover_thresh, "results": results}
    tag = args.keep.replace(",", "")
    with open(os.path.join(args.experiment_root, f"sparse_recovery_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # figure: frac held-out days recovered vs k, per method
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig_dir = os.path.join(args.experiment_root, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in args.methods.split(","):
        rs = [r for r in results if r["method"] == method]
        ax.plot([r["k"] for r in rs], [r["frac_held_recovered"] for r in rs], "o-", label=method)
    ax.set_title(f"Recovering UNPROMPTED days vs subspace dim k\nkeep={args.keep}, held={[ABBR[d] for d in held]} (L{L})")
    ax.set_xlabel("subspace dim k"); ax.set_ylabel("frac held-out days recovered (max P>=%.1f)" % args.recover_thresh)
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"sparse_recovery_{tag}.{e}"))
    print(f"saved -> sparse_recovery_{tag}.json")


if __name__ == "__main__":
    main()
