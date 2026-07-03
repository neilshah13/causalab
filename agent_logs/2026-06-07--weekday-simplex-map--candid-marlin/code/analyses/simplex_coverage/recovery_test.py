"""Leave-a-region-out RECOVERY test — validation of the mapping method.

Idea (user): delete the anchors in one part of behavioural space (e.g. all anchors
whose distribution peaks on Wednesday), rebuild the behaviour-relevant subspace +
sampling box from the REMAINING anchors only, then walk/sample that subspace and ask:
do we RECOVER the deleted region (reach valid distributions back in the Wednesday
corner)? If yes, the method genuinely generalises beyond its anchor sample; if no,
our activation-space walking needs improvement.

Two recovery probes per held-out region:
  (1) SAMPLING recovery — fraction of the held-out region's behaviour cells that valid
      samples from the TRAINING-only map reach (and max P(target) recovered).
  (2) PROJECTION reconstruction — project each held-out anchor's activation onto the
      TRAINING subspace and decode it; if behaviour is preserved, the training subspace
      already SPANS the deleted direction (so it is recoverable).

A matched RANDOM hold-out (same count, scattered) is run as a control: random deletion
should recover trivially (interpolation); a whole-region deletion is the hard test
(extrapolation). Recovery(region) >> Recovery(random-is-easy baseline=~1) is the failure
signature; Recovery(region) ~ Recovery(random) means the method recovers regions too.

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


def build_subspace(Xz, Y, k, npc=40):
    """CCA behaviour-relevant orthonormal basis (k,H) in standardised space."""
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    npc = min(npc, Xz.shape[0] - 1)
    apca = PCA(n_components=npc).fit(Xz)
    Xp = apca.transform(Xz)
    kk = min(k, npc, Y.shape[1])
    cca = CCA(n_components=kk, max_iter=1000).fit(Xp, Y)
    Dact = cca.x_rotations_.T @ apca.components_
    Q, _ = np.linalg.qr(Dact.T)
    return Q.T  # (k, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--carrier", default="A day of the week:")
    ap.add_argument("--holdout", required=True,
                    help="token abbr(s) to delete by argmax — comma list for multi-region (e.g. Sat,Sun)")
    ap.add_argument("--holdout-frac", type=float, default=1.0,
                    help="delete only this fraction of the region's anchors (titration)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=30000)
    ap.add_argument("--margin", type=float, default=1.5, help="generous, to allow extrapolation toward deleted region")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    auto = args.holdout.replace(",", "") + (f"_f{args.holdout_frac}" if args.holdout_frac < 1.0 else "")
    sfx = ("_" + args.tag) if args.tag else f"_{auto}"

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    tds = [ABBR.index(x) for x in args.holdout.split(",")]
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
    acts = blob["activations"][:, L, :].float().numpy()
    dists0 = blob["dists"].float().numpy()
    retained = blob["retained"].numpy().astype(bool)
    ridx = np.where(retained)[0]
    A = acts[ridx]; P = dists0[ridx]; M = A.shape[0]
    carrier_enc = pipe.load([{"raw_input": wrap(args.carrier, args.frame)}])

    @torch.no_grad()
    def decode(Hraw):
        res = []
        for s in range(0, len(Hraw), args.batch):
            hb = Hraw[s:s + args.batch]; B = hb.shape[0]
            holder["H"] = torch.from_numpy(hb).to(dev)
            o = model(input_ids=carrier_enc["input_ids"].repeat(B, 1),
                      attention_mask=carrier_enc["attention_mask"].repeat(B, 1), use_cache=False, logits_to_keep=1)
            pr = F.softmax(o.logits[:, -1, :].float(), -1).cpu()
            dd, _ = concept_distributions(pr, variant_ids, canonical_ids)
            res.append(dd.numpy())
        return np.concatenate(res, 0)

    amaxP = P[:, :n].argmax(1)
    region = np.where(np.isin(amaxP, tds))[0]          # held-out (deleted) region indices into anchors
    if args.holdout_frac < 1.0 and len(region) > 1:
        nkeep = max(1, int(round(args.holdout_frac * len(region))))
        region = rng.choice(region, size=nkeep, replace=False)
    nreg = len(region)
    print(f"holdout={args.holdout} (tds={tds}, frac={args.holdout_frac}): {nreg}/{M} anchors deleted")

    # 2D Hellinger plane (fit on ALL anchors) for coverage bookkeeping
    sA = np.sqrt(np.clip(P, 0, None)); mA = sA.mean(0)
    _, _, Vt = np.linalg.svd(sA - mA, full_matrices=False); comps = Vt[:2]
    def plane(d):
        return (np.sqrt(np.clip(d, 0, None)) - mA) @ comps.T
    G = 40
    CA = plane(P)
    lo = CA.min(0) - 0.3 * np.abs(CA).max(0); hi = CA.max(0) + 0.3 * np.abs(CA).max(0)
    def cells(C):
        ix = np.clip(((C[:, 0] - lo[0]) / (hi[0] - lo[0] + 1e-9) * G).astype(int), 0, G - 1)
        iy = np.clip(((C[:, 1] - lo[1]) / (hi[1] - lo[1] + 1e-9) * G).astype(int), 0, G - 1)
        return set(zip(ix.tolist(), iy.tolist()))

    def run_holdout(hold_idx, name):
        train = np.setdiff1d(np.arange(M), hold_idx)
        Xtr = A[train]; mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xtr_z = (Xtr - mu) / sd
        Ytr = np.sqrt(np.clip(P[train], 0, None))
        Q = build_subspace(Xtr_z, Ytr, args.k)
        c = Xtr_z.mean(0); coords = (Xtr_z - c) @ Q.T
        lob = coords.min(0); hib = coords.max(0); span = hib - lob
        lob2 = lob - args.margin * span; hib2 = hib + args.margin * span
        samp = rng.uniform(lob2, hib2, size=(args.n_samples, Q.shape[0]))
        Xraw = ((c[None] + samp @ Q) * sd + mu).astype(np.float32)
        dd = decode(Xraw)
        valid = (dd[:, :n].sum(1) >= 0.90) & (dd[:, n] <= 0.10)
        vdd = dd[valid]
        in_region = np.isin(vdd[:, :n].argmax(1), tds)
        # cell-coverage recovery of the deleted region
        held_cells = cells(plane(P[hold_idx]))
        valid_cells = cells(plane(vdd)) if len(vdd) else set()
        recovered_cells = held_cells & valid_cells
        # projection reconstruction of held-out anchors onto TRAINING subspace
        Xho_z = (A[hold_idx] - mu) / sd
        proj = c[None] + ((Xho_z - c) @ Q.T) @ Q
        recon = decode((proj * sd + mu).astype(np.float32))
        recon_h = np.array([hell(recon[i], P[hold_idx][i]) for i in range(len(hold_idx))])
        recon_keeps_argmax = float(np.isin(recon[:, :n].argmax(1), tds).mean())
        return {
            "name": name, "n_held": int(len(hold_idx)), "n_train": int(len(train)),
            "n_valid_samples": int(valid.sum()),
            "frac_valid_in_deleted_region": round(float(in_region.mean()) if len(vdd) else 0.0, 3),
            "n_valid_in_deleted_region": int(in_region.sum()),
            "max_P_target_recovered": round(float(vdd[:, tds].max()) if len(vdd) else 0.0, 3),
            "max_P_per_deleted": {ABBR[d]: round(float(vdd[:, d].max()) if len(vdd) else 0.0, 3) for d in tds},
            "held_region_cells": len(held_cells), "recovered_cells": len(recovered_cells),
            "cell_recovery_rate": round(len(recovered_cells) / max(len(held_cells), 1), 3),
            "projection_recon_median_hellinger": round(float(np.median(recon_h)), 3),
            "projection_recon_keeps_argmax_frac": round(recon_keeps_argmax, 3),
        }, dict(valid_plane=plane(vdd) if len(vdd) else np.zeros((0, 2)),
                held_plane=plane(P[hold_idx]), train_plane=plane(P[train]), in_region=in_region)

    res_region, viz = run_holdout(region, f"region:{args.holdout}")
    rand_idx = rng.choice(M, size=nreg, replace=False)
    res_random, _ = run_holdout(rand_idx, "random_matched")
    handle.remove()

    out = {"concept": args.concept, "layer": L, "holdout": args.holdout, "k": args.k,
           "holdout_frac": args.holdout_frac, "margin": args.margin, "n_samples": args.n_samples,
           "region_holdout": res_region, "random_control": res_random}
    with open(os.path.join(args.experiment_root, f"recovery_test{sfx}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig_dir = os.path.join(args.experiment_root, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(viz["train_plane"][:, 0], viz["train_plane"][:, 1], s=18, c="lightgray", label="training anchors")
    if len(viz["valid_plane"]):
        ax.scatter(viz["valid_plane"][~viz["in_region"], 0], viz["valid_plane"][~viz["in_region"], 1],
                   s=4, c="tab:green", alpha=0.2, label="valid samples (training-only map)")
        ax.scatter(viz["valid_plane"][viz["in_region"], 0], viz["valid_plane"][viz["in_region"], 1],
                   s=6, c="tab:purple", alpha=0.5, label=f"recovered (argmax={args.holdout})")
    ax.scatter(viz["held_plane"][:, 0], viz["held_plane"][:, 1], s=45, facecolors="none",
               edgecolors="red", linewidths=1.4, label=f"DELETED anchors ({args.holdout})")
    ax.set_title(f"Recovery of deleted '{args.holdout}' region — L{L}\n"
                 f"cell-recovery {res_region['cell_recovery_rate']:.0%}, "
                 f"proj-recon keeps-argmax {res_region['projection_recon_keeps_argmax_frac']:.0%}")
    ax.set_xlabel("Hellinger-PC1"); ax.set_ylabel("Hellinger-PC2"); ax.legend(fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"recovery{sfx}.{e}"))
    print(json.dumps(out, indent=2))
    print(f"saved -> recovery_test{sfx}.json")


if __name__ == "__main__":
    main()
