"""Compare the behavioural subspace of days to the activation subspace, per layer.

Step 2 analysis for weekday-simplex-map. LINEAR subspace exploration — NOT manifold
fitting. Given paired (behaviour p(x), activation h_ℓ(x)) for the retained prompts:

  behaviour subspace : span of Hellinger coords √p(x) (the ~6-D weekday simplex region)
  activation subspace: span of residual-stream h_ℓ(x) at layer ℓ (top-k PCA directions)

For each layer we report:
  - activation effective dimension (participation ratio of PCA spectrum);
  - CCA canonical correlations between the activation top-k PCs and the behaviour
    coords = cosines of principal angles between the two subspaces -> how many
    behavioural dimensions are LINEARLY present in the activation subspace, and how
    strongly aligned;
  - cross-validated R² of a linear decode activation -> behaviour coords.

Outputs metrics + figures. Pure CPU (loads the saved tensors); run on the login
node or inside the GPU job after capture.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def participation_ratio(eigvals):
    eigvals = np.clip(np.asarray(eigvals, float), 0, None)
    return float((eigvals.sum() ** 2) / np.clip((eigvals**2).sum(), 1e-12, None))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from safetensors.torch import load_file  # bf16-safe; convert to numpy below

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", required=True, help="dir with activations.safetensors")
    ap.add_argument("--act-pcs", type=int, default=40, help="top activation PCs to keep per layer")
    ap.add_argument("--beh-dims", type=int, default=6, help="behaviour PCA dims (subspace size)")
    ap.add_argument("--cca-comps", type=int, default=6)
    args = ap.parse_args()

    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    meta = json.load(open(os.path.join(args.experiment_root, "activations_meta.json")))
    A = blob["activations"].float().numpy()             # (N, L+1, H)  (bf16 -> fp32)
    dists = blob["dists"].float().numpy()               # (N, 8)
    retained = blob["retained"].numpy().astype(bool)    # (N,)
    fams = np.array(meta["families"])
    amax = np.array(meta["argmax_day"])
    ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    R = retained
    A = A[R]                                            # (M, L+1, H)
    P = dists[R]                                        # (M, 8)
    fams_r = fams[R]
    amax_r = amax[R]
    M, Lp1, H = A.shape
    print(f"retained M={M}, layers(+embed)={Lp1}, hidden={H}")

    # ---- behaviour subspace: Hellinger coords, PCA to beh_dims ----
    sqrtP = np.sqrt(np.clip(P, 0, None))               # (M, 8)
    beh_pca = PCA(n_components=min(args.beh_dims, 7))
    B = beh_pca.fit_transform(sqrtP)                    # (M, db)
    db = B.shape[1]
    print(f"behaviour subspace: {db} dims, EVR {np.round(beh_pca.explained_variance_ratio_,3)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    per_layer = []
    for L in range(Lp1):
        X = A[:, L, :]                                  # (M, H)
        # standardize per-dim then PCA to top-k
        mu = X.mean(0); sd = X.std(0) + 1e-6
        Xz = (X - mu) / sd
        k = min(args.act_pcs, M - 1)
        apca = PCA(n_components=k)
        Xp = apca.fit_transform(Xz)                    # (M, k)
        act_pr = participation_ratio(apca.explained_variance_)

        # CCA canonical correlations (cosines of principal angles between subspaces)
        nc = min(args.cca_comps, k, db)
        cca = CCA(n_components=nc, max_iter=1000)
        try:
            Xc, Yc = cca.fit_transform(Xp, B)
            ccorr = [float(np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]) for i in range(nc)]
        except Exception as e:
            ccorr = [float("nan")] * nc
            Xc = Yc = None
            print(f"  layer {L}: CCA failed {e}")

        # cross-validated linear decode activation(top-k PCs) -> behaviour coords
        r2s = []
        for tr, te in kf.split(Xp):
            reg = LinearRegression().fit(Xp[tr], B[tr])
            r2s.append(r2_score(B[te], reg.predict(Xp[te]), multioutput="variance_weighted"))
        r2_cv = float(np.mean(r2s))

        per_layer.append(
            {
                "layer": L,
                "act_participation_ratio": round(act_pr, 2),
                "canonical_correlations": [round(c, 3) for c in ccorr],
                "mean_top_canon": round(float(np.nanmean(ccorr)), 3),
                "n_canon_above_0.9": int(np.sum(np.array(ccorr) > 0.9)),
                "linear_decode_r2_cv": round(r2_cv, 3),
            }
        )
        print(
            f"  L{L:2d} actPR={act_pr:5.1f} meanCanon={np.nanmean(ccorr):.3f} "
            f"nCanon>0.9={int(np.sum(np.array(ccorr)>0.9))} R2cv={r2_cv:.3f}"
        )

    best_decode = max(per_layer, key=lambda d: d["linear_decode_r2_cv"])
    best_align = max(per_layer, key=lambda d: d["mean_top_canon"])
    metrics = {
        "model": meta["model"],
        "frame": meta["frame"],
        "M_retained": M,
        "behaviour_subspace_dims": db,
        "behaviour_evr": [round(float(x), 3) for x in beh_pca.explained_variance_ratio_],
        "act_pcs_used": int(min(args.act_pcs, M - 1)),
        "best_layer_linear_decode": {"layer": best_decode["layer"], "r2_cv": best_decode["linear_decode_r2_cv"]},
        "best_layer_subspace_align": {"layer": best_align["layer"], "mean_canon": best_align["mean_top_canon"]},
        "per_layer": per_layer,
    }
    with open(os.path.join(args.experiment_root, "subspace_compare.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- figures ----
    fig_dir = os.path.join(args.experiment_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    layers = [d["layer"] for d in per_layer]
    r2 = [d["linear_decode_r2_cv"] for d in per_layer]
    meancanon = [d["mean_top_canon"] for d in per_layer]
    actpr = [d["act_participation_ratio"] for d in per_layer]

    # 1) alignment & decode vs layer
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, r2, "o-", label="linear-decode R² (CV)")
    ax.plot(layers, meancanon, "s-", label=f"mean top-{args.cca_comps} canonical corr")
    ax.axvline(best_decode["layer"], color="r", ls="--", alpha=0.5, label=f"best decode L{best_decode['layer']}")
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Behaviour vs activation subspace alignment by layer ({meta['model']})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "subspace_alignment_by_layer.pdf")); plt.close(fig)

    # 2) canonical-correlation heatmap (layer x component)
    cc = np.array([[d["canonical_correlations"][i] if i < len(d["canonical_correlations"]) else np.nan
                    for i in range(args.cca_comps)] for d in per_layer])
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(cc, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xlabel("canonical component"); ax.set_ylabel("layer")
    ax.set_xticks(range(args.cca_comps)); ax.set_xticklabels([f"c{i+1}" for i in range(args.cca_comps)])
    ax.set_title("Canonical correlations (principal-angle cosines)")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "canonical_correlations_heatmap.pdf")); plt.close(fig)

    # 3) activation effective dim vs layer
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, actpr, "o-", color="purple")
    ax.set_xlabel("layer"); ax.set_ylabel("participation ratio")
    ax.set_title(f"Activation effective dimension by layer ({meta['model']})")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "activation_effdim_by_layer.pdf")); plt.close(fig)

    # 4) best-layer CCA: 1st canonical variate pair, coloured by argmax day
    L = best_align["layer"]
    X = A[:, L, :]; mu = X.mean(0); sd = X.std(0) + 1e-6
    Xp = PCA(n_components=min(args.act_pcs, M - 1)).fit_transform((X - mu) / sd)
    cca = CCA(n_components=2, max_iter=1000); Xc, Yc = cca.fit_transform(Xp, B)
    day_cmap = plt.cm.rainbow(np.linspace(0, 1, 7))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (xx, yy, ttl) in zip(
        axes,
        [(Xc[:, 0], Yc[:, 0], "canonical pair 1"), (Xc[:, 1], Yc[:, 1], "canonical pair 2")],
    ):
        for di, d in enumerate(ABBR):
            m = amax_r == d
            if m.any():
                ax.scatter(xx[m], yy[m], s=22, color=day_cmap[di], label=d, alpha=0.8)
        ax.set_xlabel("activation canonical variate"); ax.set_ylabel("behaviour canonical variate")
        ax.set_title(ttl)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Best-aligned layer L{L}: activation vs behaviour canonical variates")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "best_layer_cca_scatter.pdf")); plt.close(fig)

    print(
        f"\nbest linear-decode layer L{best_decode['layer']} R2cv={best_decode['linear_decode_r2_cv']}; "
        f"best subspace-align layer L{best_align['layer']} meanCanon={best_align['mean_top_canon']}"
    )
    print(f"metrics + figures -> {args.experiment_root}")


if __name__ == "__main__":
    main()
