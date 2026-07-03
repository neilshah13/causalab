"""Offline geometry analysis of the subspace map (no GPU — consumes geometry_L*.npz).

The map figures so far colour samples valid/invalid only, which is why the region looks
structureless ("after PCA everything maps to valid behaviour"). This measures the
geometry that validity can't see, per layer:

  1. ISOMETRY — d_Euclid in subspace coords vs d_Hellinger in behaviour over sample
     pairs (the paper's central quantitative object, applied to the subspace map):
     global Pearson r, local r (short pairs), and a binned distance-distortion curve.
  2. FOLIATION — the activation box coloured by achieved argmax / entropy: are the
     level sets of the map straight parallel slabs (affine readout) or curved?
     Quantified via local-Jacobian variation (3).
  3. JACOBIAN FIELD — local linear fits coords->sqrt(p) on kNN neighbourhoods at many
     base points: singular-value spread and principal-angle drift vs the global linear
     map. Constant J = linear map; drifting J = curvature.
  4. DENSITY — where natural anchors concentrate inside the valid blob
     (distance-to-nearest-anchor distribution; anchor-occupied fraction of valid cells).
  5. RING SKELETON — argmax-conditional centroids in the top-2 subspace dims, linked in
     cyclic order: the paper's 1-D ring as the skeleton of the filled region; circular-
     order test (do centroid angles follow the concept cycle?).
  6. GLOBAL LINEARITY — R² of the anchor-fitted linear forward map evaluated on the
     box samples (does the affine map fit the whole region, not just the anchors?).

Run from session root:  python3 code/analyses/simplex_coverage/analyze_geometry.py \
    --concept weekdays --layers 19,23,27,31
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

SESSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def hellinger_pairs(sq, idx_a, idx_b):
    d = sq[idx_a] - sq[idx_b]
    return np.sqrt((d ** 2).sum(1)) / math.sqrt(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--layers", default="19,23,27,31")
    ap.add_argument("--n-pair-sample", type=int, default=4000)
    ap.add_argument("--n-jac-points", type=int, default=200)
    ap.add_argument("--knn", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    root = os.path.join(SESSION, "artifacts", f"{'weekday' if args.concept=='weekdays' else args.concept}_simplex",
                        "llama31_8b", "simplex_coverage")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {"concept": args.concept, "layers": {}}
    layers = [int(x) for x in args.layers.split(",")]
    iso_curves = {}

    for L in layers:
        p = os.path.join(root, f"geometry_L{L}.npz")
        if not os.path.exists(p):
            print(f"L{L}: no dump, skipping"); continue
        z = np.load(p)
        C, D = z["samp_coords"].astype(np.float64), z["samp_dists"].astype(np.float64)
        V = z["valid"].astype(bool)
        AC, AD = z["anchor_coords"].astype(np.float64), z["anchor_dists"].astype(np.float64)
        pc = os.path.join(root, f"geometry_core_L{L}.npz")
        if os.path.exists(pc):
            zc = np.load(pc)   # same chart (deterministic fit on the same anchors) — safe to merge
            C = np.concatenate([C, zc["samp_coords"].astype(np.float64)], 0)
            D = np.concatenate([D, zc["samp_dists"].astype(np.float64)], 0)
            V = np.concatenate([V, zc["valid"].astype(bool)], 0)
            print(f"L{L}: merged core dump ({zc['valid'].sum()} valid core samples)")
        n = AD.shape[1] - 1
        k = C.shape[1]
        Cv, Dv = C[V], D[V]
        sqv = np.sqrt(np.clip(Dv, 0, None))
        sqa = np.sqrt(np.clip(AD, 0, None))

        # ---- 1. isometry over valid sample pairs ----
        m = min(args.n_pair_sample, len(Cv))
        sub = rng.choice(len(Cv), size=m, replace=False)
        ia = rng.integers(0, m, size=60000); ib = rng.integers(0, m, size=60000)
        keep = ia != ib; ia, ib = ia[keep], ib[keep]
        da = np.sqrt(((Cv[sub][ia] - Cv[sub][ib]) ** 2).sum(1))
        db = hellinger_pairs(sqv[sub], ia, ib)
        r_global = float(np.corrcoef(da, db)[0, 1])
        q25 = np.quantile(da, 0.25)
        loc = da < q25
        r_local = float(np.corrcoef(da[loc], db[loc])[0, 1])
        # binned distortion curve (does dH saturate? is the map metric-affine?)
        bins = np.quantile(da, np.linspace(0, 1, 25))
        bi = np.clip(np.digitize(da, bins) - 1, 0, 23)
        curve_x = [float(da[bi == i].mean()) for i in range(24) if (bi == i).sum() > 50]
        curve_y = [float(db[bi == i].mean()) for i in range(24) if (bi == i).sum() > 50]
        iso_curves[L] = (curve_x, curve_y, r_global)
        # anchors-only isometry (natural states)
        Ma = len(AC)
        iaa = rng.integers(0, Ma, size=20000); ibb = rng.integers(0, Ma, size=20000)
        keep = iaa != ibb; iaa, ibb = iaa[keep], ibb[keep]
        daa = np.sqrt(((AC[iaa] - AC[ibb]) ** 2).sum(1))
        dba = hellinger_pairs(sqa, iaa, ibb)
        r_anchor = float(np.corrcoef(daa, dba)[0, 1])

        # ---- 3. Jacobian field ----
        nb = min(args.n_jac_points, len(Cv))
        base_idx = rng.choice(len(Cv), size=nb, replace=False)
        from scipy.spatial import cKDTree
        tree = cKDTree(Cv)
        smax, smin, fro, ang = [], [], [], []
        # global linear forward map fit on anchors
        Ab = np.concatenate([AC, np.ones((len(AC), 1))], 1)
        Wg, *_ = np.linalg.lstsq(Ab, sqa, rcond=None)
        Wg_row = Wg[:-1]
        Ug, _, _ = np.linalg.svd(Wg_row, full_matrices=False)
        for bi_ in base_idx:
            _, nn = tree.query(Cv[bi_], k=args.knn)
            Xn = Cv[nn]; Yn = sqv[nn]
            Xb = np.concatenate([Xn - Xn.mean(0), np.ones((len(Xn), 1))], 1)
            Wl, *_ = np.linalg.lstsq(Xb, Yn, rcond=None)
            Jl = Wl[:-1]                                   # (k, n+1) local Jacobian
            sv = np.linalg.svd(Jl, compute_uv=False)
            smax.append(float(sv[0])); smin.append(float(sv[min(len(sv), n) - 1]))
            fro.append(float(np.sqrt((sv ** 2).sum())))
            Ul, _, _ = np.linalg.svd(Jl, full_matrices=False)
            r_ = min(3, Ul.shape[1], Ug.shape[1])
            cosang = np.linalg.svd(Ul[:, :r_].T @ Ug[:, :r_], compute_uv=False)
            ang.append(float(np.degrees(np.arccos(np.clip(cosang[-1], -1, 1)))))
        jac = {"sigma_max_cv": round(float(np.std(smax) / np.mean(smax)), 3),
               "frobenius_cv": round(float(np.std(fro) / np.mean(fro)), 3),
               "principal_angle_to_global_deg_med": round(float(np.median(ang)), 1),
               "principal_angle_p90": round(float(np.percentile(ang, 90)), 1)}

        # ---- 6. global linearity: anchor-fit map evaluated on box samples ----
        pred = np.concatenate([Cv, np.ones((len(Cv), 1))], 1) @ Wg
        ss_res = ((sqv - pred) ** 2).sum()
        ss_tot = ((sqv - sqv.mean(0)) ** 2).sum()
        r2_samples = float(1 - ss_res / ss_tot)
        pred_a = Ab @ Wg
        r2_anchors = float(1 - ((sqa - pred_a) ** 2).sum() / ((sqa - sqa.mean(0)) ** 2).sum())
        # per-sample residual Hellinger of the linear model
        lin_resid = np.sqrt(((sqv - pred) ** 2).sum(1)) / math.sqrt(2)

        # ---- 4. density + TWO-REGIME curve: chart residual vs distance from the natural set ----
        atree = cKDTree(AC)
        dnn, _ = atree.query(Cv, k=1)
        pw = np.sqrt(((AC[:, None, :] - AC[None, :, :]) ** 2).sum(-1))
        nn_anchor = float(np.median(np.sort(pw, axis=1)[:, 1]))   # median nearest-neighbour spacing
        dens = {"median_dist_to_anchor": round(float(np.median(dnn)), 2),
                "p90_dist_to_anchor": round(float(np.percentile(dnn, 90)), 2),
                "median_anchor_NN_spacing": round(nn_anchor, 2)}
        ent_v = -(Dv[:, :n] * np.log(np.clip(Dv[:, :n], 1e-12, None))).sum(1)
        regime_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 99)]
        regime = []
        for lo, hi in regime_bins:
            sel = (dnn >= lo * nn_anchor) & (dnn < hi * nn_anchor)
            if sel.sum() < 30:
                continue
            regime.append({"nn_range": f"{lo}-{hi}", "n": int(sel.sum()),
                           "chart_resid_med": round(float(np.median(lin_resid[sel])), 3),
                           "entropy_med": round(float(np.median(ent_v[sel])), 2),
                           "frac_near_vertex": round(float((ent_v[sel] < 0.3).mean()), 2)})

        # ---- 5. ring skeleton ----
        amax_a = AD[:, :n].argmax(1)
        cents = np.stack([AC[amax_a == d].mean(0) if (amax_a == d).any() else np.full(k, np.nan)
                          for d in range(n)], 0)
        # angle in the centroids' OWN principal plane (top-2 of centroid variance) — fairer than
        # global dims 1-2 when n is large relative to 2-D
        ok = ~np.isnan(cents[:, 0])
        cc = cents[ok] - cents[ok].mean(0)
        _, _, Vt_c = np.linalg.svd(cc, full_matrices=False)
        c2_ok = cc @ Vt_c[:2].T
        c2 = np.full((n, 2), np.nan)
        c2[ok] = c2_ok
        theta = np.arctan2(c2[:, 1], c2[:, 0])
        pop = np.where(ok)[0]                                  # concept values with anchors
        np_ = len(pop)
        order_by_angle = pop[np.argsort(theta[pop])]           # values sorted by centroid angle
        # circular alignment: best cyclic rotation (either direction) matching the concept order
        best_match = 0
        for direction in (1, -1):
            seq = list(order_by_angle[::direction])
            for s in range(np_):
                rot = seq[s:] + seq[:s]
                best_match = max(best_match, int(sum(int(rot[i] == pop[i]) for i in range(np_))))
        ring = {"circular_order_match": f"{best_match}/{np_}",
                "centroid_angles_deg": [None if np.isnan(t) else round(float(np.degrees(t)), 1)
                                        for t in theta]}

        out["layers"][str(L)] = {
            "n_valid": int(V.sum()), "k": int(k),
            "isometry_r_global": round(r_global, 3), "isometry_r_local": round(r_local, 3),
            "isometry_r_anchors": round(r_anchor, 3),
            "jacobian": jac, "linear_forward_R2_anchors": round(r2_anchors, 3),
            "linear_forward_R2_box_samples": round(r2_samples, 3),
            "linear_residual_hellinger_med": round(float(np.median(lin_resid)), 3),
            "density": dens, "ring": ring, "chart_breakdown_by_distance": regime,
        }
        print(f"L{L}: iso r={r_global:.3f} (local {r_local:.3f}, anchors {r_anchor:.3f}) | "
              f"linW R2 box={r2_samples:.3f} | J angle med {jac['principal_angle_to_global_deg_med']}° | "
              f"ring {ring['circular_order_match']}", flush=True)

        # ---- figures: foliation + density + isometry (2x2) ----
        fig, ax = plt.subplots(2, 2, figsize=(13, 11))
        amax_v = Dv[:, :n].argmax(1)
        cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
        for d in range(n):
            sel = amax_v == d
            if sel.any():
                ax[0, 0].scatter(Cv[sel, 0], Cv[sel, 1], s=3, color=cmap(d % 20), alpha=0.4)
        ring_xy = np.concatenate([cents[:, :2], cents[:1, :2]], 0)
        ax[0, 0].plot(ring_xy[:, 0], ring_xy[:, 1], "k-o", lw=1.5, ms=5)
        ax[0, 0].set_title(f"L{L} foliation: box coloured by ACHIEVED argmax + centroid ring")
        ax[0, 0].set_xlabel("subspace dim 1"); ax[0, 0].set_ylabel("dim 2")
        ent = -(Dv[:, :n] * np.log(np.clip(Dv[:, :n], 1e-12, None))).sum(1)
        sc = ax[0, 1].scatter(Cv[:, 0], Cv[:, 1], s=3, c=ent, cmap="viridis", alpha=0.5)
        ax[0, 1].scatter(AC[:, 0], AC[:, 1], s=18, facecolors="none", edgecolors="r", lw=0.6)
        plt.colorbar(sc, ax=ax[0, 1], label="achieved entropy (nats)")
        ax[0, 1].set_title("entropy level sets (anchors red)")
        sc2 = ax[1, 0].scatter(Cv[:, 0], Cv[:, 1], s=3, c=dnn, cmap="magma", alpha=0.5)
        plt.colorbar(sc2, ax=ax[1, 0], label="dist to nearest anchor (coords)")
        ax[1, 0].set_title("naturalness: distance to anchor cloud")
        ax[1, 1].plot(curve_x, curve_y, "o-", ms=3)
        ax[1, 1].set_xlabel("activation distance (subspace coords)")
        ax[1, 1].set_ylabel("behaviour distance (Hellinger)")
        ax[1, 1].set_title(f"isometry curve r={r_global:.3f} (anchors r={r_anchor:.3f})")
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(fig_dir, f"geometry_L{L}.{e}"))
        plt.close(fig)

    # cross-layer isometry overlay
    if iso_curves:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for L, (cx, cy, r) in sorted(iso_curves.items()):
            ax.plot(cx, cy, "o-", ms=3, label=f"L{L} (r={r:.3f})")
        ax.set_xlabel("activation distance (subspace coords)")
        ax.set_ylabel("behaviour distance (Hellinger)")
        ax.set_title(f"{args.concept}: metric distortion of the map by layer")
        ax.legend()
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(fig_dir, f"geometry_isometry_layers.{e}"))

    with open(os.path.join(root, "geometry_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> geometry_analysis.json + figures/geometry_L*.png")


if __name__ == "__main__":
    main()
