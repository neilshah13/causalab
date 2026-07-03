"""Robust routine: map the activation-space preimage of a behavioural region.

Goal: for a concept that defines a probability simplex (weekdays now; any token-set
later), map — as completely as possible — the region of activation space at a late
layer that decodes to the VALID behavioural region (concept mass >= 0.90, other <= 0.10).

Lever for completeness: the behaviour-relevant activation variation is low-dimensional
(~|Z|-1), so we (1) isolate that low-D subspace at the layer via the linear
activation->behaviour map (CCA between standardised activation PCs and Hellinger
behaviour coords), then (2) densely/quasi-uniformly SAMPLE that low-D subspace (with a
margin beyond the anchors) and patch each sample through the model to read validity +
behaviour. A low-D region can be near-exhaustively mapped; the raw 4096-D cannot.

Patching uses a FIXED carrier context (so decode is a function of the patched
last-token activation). A sanity check verifies that patching each anchor's own
activation into the carrier reproduces its behaviour.

Concept-general: pass --concept to swap the token-set/anchor source; only
``concept_token_ids`` and the anchors change.

Run from repo root (GPU + activations.safetensors from capture step):
  uv run python <...>/map_subspace.py --layers 31 --prompts <...> --experiment-root <...>
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
    return float(np.sqrt(((np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))) ** 2).sum()) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layers", default="31", help="comma list of CAPTURED hidden-state indices")
    ap.add_argument("--carrier", default="A day of the week:", help="fixed carrier stem")
    ap.add_argument("--k", type=int, default=8, help="behaviour-relevant subspace dim to map")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--margin", type=float, default=0.6, help="extend sampling box beyond anchors by this frac")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for output files so parallel sweeps don't clobber")
    args = ap.parse_args()
    sfx = ("_" + args.tag) if args.tag else ""

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    with open(args.prompts) as f:
        prompts = json.load(f)["prompts"]

    tokens, ABBR = get_concept(args.concept)
    n = len(tokens)
    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat)
    model = pipe.model; model.eval()
    variant_ids, canonical_ids, _ = build_concept_token_ids(pipe.tokenizer, tokens)
    dev = next(model.parameters()).device

    from safetensors.torch import load_file
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA

    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    acts = blob["activations"]                       # (N, L+1, H)
    dists0 = blob["dists"].float().numpy()
    retained = blob["retained"].numpy().astype(bool)
    ridx = np.where(retained)[0]

    # fixed carrier
    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])
    holder = {"H": None}

    layers = [int(x) for x in args.layers.split(",")]
    out = {"model": args.model, "carrier": args.carrier, "k": args.k,
           "n_samples": args.n_samples, "margin": args.margin, "layers": {}}

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    for L in layers:
        dl = L - 1
        handle = model.model.layers[dl].register_forward_hook(hook)

        @torch.no_grad()
        def decode(Hraw_np):
            res = []
            for s in range(0, len(Hraw_np), args.batch):
                hb = Hraw_np[s:s + args.batch]
                B = hb.shape[0]
                ids = carrier_enc["input_ids"].repeat(B, 1)
                am = carrier_enc["attention_mask"].repeat(B, 1)
                holder["H"] = torch.from_numpy(hb).to(dev)
                o = model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1)
                pr = F.softmax(o.logits[:, -1, :].float(), dim=-1).cpu()
                dd, _ = concept_distributions(pr, variant_ids, canonical_ids)
                res.append(dd.numpy())
            return np.concatenate(res, 0)

        X = acts[ridx, L, :].float().numpy()          # (M, H) anchors at layer
        mu = X.mean(0); sd = X.std(0) + 1e-6
        Xz = (X - mu) / sd
        M, H = X.shape

        # carrier sanity: patch each anchor's own raw activation into the carrier
        san = decode(X.astype(np.float32))
        san_h = np.array([hellinger(san[i], dists0[ridx[i]]) for i in range(M)])
        carrier_ok = float(np.median(san_h))

        # behaviour-relevant subspace via CCA(std-activation PCs, Hellinger behaviour)
        Y = np.sqrt(np.clip(dists0[ridx], 0, None))    # (M, 8) — behaviour, NOT pca'd
        npc = min(40, M - 1)
        apca = PCA(n_components=npc).fit(Xz)
        Xp = apca.transform(Xz)                        # (M, npc)
        k = min(args.k, npc, Y.shape[1])
        cca = CCA(n_components=k, max_iter=1000).fit(Xp, Y)
        Dact = cca.x_rotations_.T @ apca.components_    # (k, H) std-space directions
        Q, _ = np.linalg.qr(Dact.T)                    # (H, k) orthonormal
        Q = Q.T                                        # (k, H)
        Xc_cca, Yc_cca = cca.transform(Xp, Y)
        ccorr = [round(float(np.corrcoef(Xc_cca[:, i], Yc_cca[:, i])[0, 1]), 3) for i in range(k)]

        c = Xz.mean(0)                                 # centroid (std space)
        coords = (Xz - c) @ Q.T                        # (M, k) anchor coords in subspace
        lo = coords.min(0); hi = coords.max(0)
        span = hi - lo
        lo2 = lo - args.margin * span; hi2 = hi + args.margin * span

        # quasi-uniform sample of the k-D box (+ the anchors themselves)
        S = args.n_samples
        samp = rng.uniform(lo2, hi2, size=(S, k))      # (S, k)
        Xz_s = c[None, :] + samp @ Q                   # (S, H) std space
        Xraw_s = (Xz_s * sd + mu).astype(np.float32)   # un-standardise
        dd = decode(Xraw_s)                            # (S, n+1)
        wm = dd[:, :n].sum(1); other = dd[:, n]
        valid = (wm >= 0.90) & (other <= 0.10)

        # ---- completeness / shape metrics ----
        # behaviour coverage: 2D Hellinger plane (fit on anchors), grid 40x40
        sA = np.sqrt(np.clip(dists0[ridx], 0, None)); mA = sA.mean(0)
        _, _, Vt = np.linalg.svd(sA - mA, full_matrices=False); comps = Vt[:2]
        CA = (sA - mA) @ comps.T
        CV = (np.sqrt(np.clip(dd[valid], 0, None)) - mA) @ comps.T
        G = 40
        plo = np.minimum(CA.min(0), CV.min(0) if len(CV) else CA.min(0))
        phi = np.maximum(CA.max(0), CV.max(0) if len(CV) else CA.max(0))
        def cellset(C):
            ix = np.clip(((C[:, 0]-plo[0])/(phi[0]-plo[0]+1e-9)*G).astype(int), 0, G-1)
            iy = np.clip(((C[:, 1]-plo[1])/(phi[1]-plo[1]+1e-9)*G).astype(int), 0, G-1)
            return set(zip(ix.tolist(), iy.tolist()))
        ca = cellset(CA); cv = cellset(CV) if len(CV) else set()

        layer_res = {
            "carrier_sanity_median_hellinger": round(carrier_ok, 4),
            "cca_canonical_corr": ccorr,
            "valid_fraction_of_box": round(float(valid.mean()), 3),
            "n_valid": int(valid.sum()), "n_samples": int(S),
            "behaviour_cells_anchors": len(ca), "behaviour_cells_valid_samples": len(cv),
            "behaviour_cells_new": len(cv - ca),
            "coverage_gain_ratio": round(len(ca | cv) / max(len(ca), 1), 2),
            "argmax_day_counts_valid": {ABBR[d]: int((dd[valid][:, :n].argmax(1) == d).sum()) for d in range(n)},
        }
        out["layers"][str(L)] = layer_res
        print(f"L{L}: carrier_sanity(med Hellinger)={carrier_ok:.3f}  "
              f"valid_frac_of_box={valid.mean():.3f}  cov_gain={layer_res['coverage_gain_ratio']}  "
              f"new_cells={layer_res['behaviour_cells_new']}")

        # per-layer plot: subspace coords PC1-PC2 colored valid; + behaviour coverage
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig_dir = os.path.join(args.experiment_root, "figures"); os.makedirs(fig_dir, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        ax[0].scatter(samp[~valid, 0], samp[~valid, 1], s=3, c="lightgray", alpha=0.3, label="invalid")
        ax[0].scatter(samp[valid, 0], samp[valid, 1], s=3, c="tab:green", alpha=0.4, label="valid")
        ax[0].scatter(coords[:, 0], coords[:, 1], s=25, facecolors="none", edgecolors="k", label="anchors")
        ax[0].set_title(f"L{L}: valid region in behaviour-relevant subspace (dims 1-2)")
        ax[0].set_xlabel("subspace dim 1"); ax[0].set_ylabel("subspace dim 2"); ax[0].legend(fontsize=8)
        if len(CV):
            ax[1].scatter(CV[:, 0], CV[:, 1], s=4, c="tab:green", alpha=0.25, label="valid samples")
        ax[1].scatter(CA[:, 0], CA[:, 1], s=25, facecolors="none", edgecolors="k", label="anchors")
        ax[1].set_title(f"L{L}: behaviour reached (Hellinger plane)")
        ax[1].set_xlabel("Hellinger-PC1"); ax[1].set_ylabel("Hellinger-PC2"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"map_subspace{sfx}_L{L}.png"))
        fig.savefig(os.path.join(fig_dir, f"map_subspace{sfx}_L{L}.pdf")); plt.close(fig)

        handle.remove()

    with open(os.path.join(args.experiment_root, f"map_subspace_metrics{sfx}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.experiment_root}/map_subspace_metrics.json")


if __name__ == "__main__":
    main()
