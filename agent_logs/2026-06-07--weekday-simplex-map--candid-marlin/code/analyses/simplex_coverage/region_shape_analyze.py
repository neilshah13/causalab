"""Analyse the SHAPE of the valid weekday activation region in behaviour space.

Loads region_shape_L{L}.npz (valid in-subspace perturbations + their simplex points
+ naturalness) and asks: do perturbations FILL the behavioural simplex / reach
NEW regions beyond the anchors (map onto?), or just smear locally? Plus how natural
are the reached points. Pure numpy + matplotlib (runs locally; no torch).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hellinger_pca(sqrtP, mean, comps):
    return (sqrtP - mean) @ comps.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layer", type=int, default=31)
    args = ap.parse_args()
    npz = np.load(os.path.join(args.root, f"region_shape_L{args.layer}.npz"))
    P = npz["pprime"]                      # (Np, 8) valid perturbation distributions
    A = npz["anchor_pprime"]               # (M, 8) anchor distributions
    nat_nn = npz["nat_nn"]; hell = npz["hellinger"]; rad = npz["radius"]
    print(f"valid perturbations Np={len(P)}, anchors M={len(A)}")

    # behaviour Hellinger-PCA fit on ANCHORS (we never PCA to reduce behaviour; this
    # is only a 2-D projection for visualising/quantifying coverage)
    sA = np.sqrt(np.clip(A, 0, None))
    mean = sA.mean(0)
    _, _, Vt = np.linalg.svd(sA - mean, full_matrices=False)
    comps = Vt[:3]
    CA = hellinger_pca(sA, mean, comps)              # (M,3)
    CP = hellinger_pca(np.sqrt(np.clip(P, 0, None)), mean, comps)  # (Np,3)

    amax_A = A[:, :7].argmax(1)
    amax_P = P[:, :7].argmax(1)

    # ---- coverage via 2-D grid (no scipy) ----
    lo = np.minimum(CA[:, :2].min(0), CP[:, :2].min(0))
    hi = np.maximum(CA[:, :2].max(0), CP[:, :2].max(0))
    G = 40
    def cell_idx(C):
        ix = np.clip(((C[:, 0] - lo[0]) / (hi[0] - lo[0] + 1e-9) * G).astype(int), 0, G - 1)
        iy = np.clip(((C[:, 1] - lo[1]) / (hi[1] - lo[1] + 1e-9) * G).astype(int), 0, G - 1)
        return ix, iy
    aix, aiy = cell_idx(CA); pix, piy = cell_idx(CP)
    ca = set(zip(aix.tolist(), aiy.tolist()))
    cp = set(zip(pix.tolist(), piy.tolist()))
    new_cells = cp - ca
    frac_new = float(np.mean([(pix[i], piy[i]) in new_cells for i in range(len(CP))])) if len(CP) else 0.0
    metrics = {
        "layer": args.layer, "n_valid_perturbations": int(len(P)), "n_anchors": int(len(A)),
        "grid": G,
        "cells_anchors": len(ca), "cells_perturbations": len(cp),
        "cells_new_from_perturbations": len(new_cells),
        "coverage_gain_ratio": round(len(ca | cp) / max(len(ca), 1), 2),
        "frac_perturbations_in_new_cells": round(frac_new, 3),
        "anchor_2d_std": [round(float(x), 3) for x in CA[:, :2].std(0)],
        "perturb_2d_std": [round(float(x), 3) for x in CP[:, :2].std(0)],
        "naturalness_nn": {"mean": round(float(nat_nn.mean()), 3),
                            "p50": round(float(np.median(nat_nn)), 3),
                            "p90": round(float(np.percentile(nat_nn, 90)), 3),
                            "max": round(float(nat_nn.max()), 3)},
        "argmax_day_counts_perturb": {ABBR[d]: int((amax_P == d).sum()) for d in range(7)},
    }
    with open(os.path.join(args.root, "region_shape_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    fig_dir = os.path.join(args.root, "figures"); os.makedirs(fig_dir, exist_ok=True)
    cmap = plt.cm.rainbow(np.linspace(0, 1, 7))

    # (1) overlay: anchors (black rings) over perturbations (coloured by day)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for d in range(7):
        m = amax_P == d
        if m.any():
            ax[0].scatter(CP[m, 0], CP[m, 1], s=8, color=cmap[d], alpha=0.35)
    ax[0].scatter(CA[:, 0], CA[:, 1], s=40, facecolors="none", edgecolors="k", linewidths=1.0, label="anchors")
    ax[0].set_title(f"Valid perturbations (filled, by day) vs anchors (rings) — L{args.layer}")
    ax[0].set_xlabel("Hellinger-PC1"); ax[0].set_ylabel("Hellinger-PC2"); ax[0].legend(fontsize=8)
    # (2) coloured by naturalness
    sc = ax[1].scatter(CP[:, 0], CP[:, 1], s=8, c=nat_nn, cmap="magma", alpha=0.6, vmax=np.percentile(nat_nn, 95))
    ax[1].scatter(CA[:, 0], CA[:, 1], s=30, facecolors="none", edgecolors="cyan", linewidths=0.8)
    ax[1].set_title("Perturbations coloured by naturalness (nearest-anchor dist, NN units)")
    ax[1].set_xlabel("Hellinger-PC1"); ax[1].set_ylabel("Hellinger-PC2")
    fig.colorbar(sc, ax=ax[1], fraction=0.046, label="nearest-anchor dist")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "region_shape_overlay.pdf"))
    fig.savefig(os.path.join(fig_dir, "region_shape_overlay.png")); plt.close(fig)

    # (3) naturalness vs behaviour moved
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(nat_nn, bins=40, color="teal", alpha=0.8); ax[0].axvline(1.0, color="r", ls="--", label="1 NN-dist")
    ax[0].set_title("Naturalness of valid perturbations"); ax[0].set_xlabel("nearest-anchor dist (NN units)"); ax[0].legend()
    ax[1].scatter(hell, nat_nn, s=6, alpha=0.3, color="purple")
    ax[1].set_title("Naturalness vs behaviour moved"); ax[1].set_xlabel("Hellinger to source anchor"); ax[1].set_ylabel("nearest-anchor dist (NN)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "region_shape_naturalness.pdf"))
    fig.savefig(os.path.join(fig_dir, "region_shape_naturalness.png")); plt.close(fig)
    print(f"figures + metrics -> {args.root}")


if __name__ == "__main__":
    main()
