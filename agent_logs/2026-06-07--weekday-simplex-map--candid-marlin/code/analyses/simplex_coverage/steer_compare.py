"""Steering head-to-head: linear vs spline-manifold vs subspace steering + generation check.

The missing comparison vs the paper. On IDENTICAL endpoint pairs (concept-value centroids)
and IDENTICAL carrier prompts, run five steering strategies at a layer:

  linear_full        paper Eq.1 — straight line between raw activation centroids,
                     ENTIRE last-token residual replaced.
  manifold           paper Eq.2 — periodic cubic spline through argmax-conditional
                     activation centroids in PCA-64 (raw space); waypoints replace the
                     top-64 PCA components, carrier's orthogonal complement preserved.
  subspace_inv_full  ours (steer_trajectory) — target the M_y spline path, linear-inverse
                     to k-D CCA-subspace coords, FULL replacement (anchor-mean complement).
  subspace_inv_resid same coords, applied as an in-subspace delta — carrier's off-subspace
                     residual preserved (apples-to-apples with `manifold`).
  linear_subspace    straight line between endpoint-centroid COORDS in the k-D subspace,
                     residual-preserving. Isolates "right subspace" from "right path".

Metrics per (pair, method), averaged over carriers pointwise (paper protocol):
  - E_BC cumulative energy to the behaviour manifold M_y (paper Eq.3; M_y = periodic
    tangent-plane spline through sqrt behaviour centroids, paper A.4)
  - off-concept mass along the path; mass-filter validity
  - smoothness (consecutive Hellinger); teleportation (argmax off the shorter cyclic
    arc between endpoints; max consecutive cyclic jump)
Plus a GENERATION check on a subsample: patch the prefill last position only, generate
24 greedy tokens, score continuation fluency = perplexity of tokens 2..24 under the
UNSTEERED model (conditioned on prompt+token1), plus repetition stats. This is the
"does the model break after steering" test the first-token metrics cannot see.

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

METHODS = ["linear_full", "manifold", "subspace_inv_full", "subspace_inv_resid", "linear_subspace"]


def hell(p, q):
    return float(np.sqrt(((np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))) ** 2).sum()) / math.sqrt(2))


def sphere_log(b, s):
    """Log map on the unit sphere at base b: returns tangent vector with |t| = geodesic dist."""
    ct = float(np.clip(np.dot(b, s), -1.0, 1.0))
    th = math.acos(ct)
    if th < 1e-8:
        return np.zeros_like(b)
    return th / math.sin(th) * (s - ct * b)


def sphere_exp(b, t):
    nt = float(np.linalg.norm(t))
    if nt < 1e-8:
        return b.copy()
    return math.cos(nt) * b + math.sin(nt) * t / nt


def fit_my_spline(bcent_sqrt):
    """Paper A.4: periodic cubic spline through sqrt behaviour centroids in the tangent
    plane at b* (normalized mean), lifted back via exp map. Returns dense samples (U, n+1)."""
    from scipy.interpolate import CubicSpline
    n = bcent_sqrt.shape[0]
    bstar = bcent_sqrt.mean(0)
    bstar = bstar / (np.linalg.norm(bstar) + 1e-12)
    T = np.stack([sphere_log(bstar, s) for s in bcent_sqrt], 0)        # (n, n+1)
    x = np.arange(n + 1, dtype=float)
    Tp = np.concatenate([T, T[:1]], 0)                                  # close the loop
    cs = CubicSpline(x, Tp, axis=0, bc_type="periodic")
    return cs, bstar


def my_samples(cs, bstar, n, U=700):
    us = np.linspace(0, n, U, endpoint=False)
    S = np.stack([sphere_exp(bstar, cs(u)) for u in us], 0)
    return np.clip(S, 0.0, None)                                        # sqrt-space points on M_y


def ebc_to_my(p, MyS):
    """Bhattacharyya distance from distribution p to nearest M_y sample (sqrt-space)."""
    bc = MyS @ np.sqrt(np.clip(p, 0, None))
    return float(-math.log(max(float(bc.max()), 1e-12)))


def cyc_arc(ua, ub, n, t):
    """Position along the SHORTER cyclic arc from ua to ub at fraction t (mod n)."""
    fwd = (ub - ua) % n
    bwd = (ua - ub) % n
    return (ua + t * fwd) % n if fwd <= bwd else (ua - t * bwd) % n


def cyc_dist(a, b, n):
    d = abs(int(a) - int(b)) % n
    return min(d, n - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layers", default="19,23,27,31")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all ordered pairs")
    ap.add_argument("--n-carriers", type=int, default=16)
    ap.add_argument("--waypoints", type=int, default=20)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--gen-pairs", type=int, default=7)
    ap.add_argument("--gen-carriers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat); model = pipe.model; model.eval()
    tok = pipe.tokenizer
    variant_ids, canonical_ids, _ = build_concept_token_ids(tok, tokens)
    dev = next(model.parameters()).device

    from safetensors.torch import load_file
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    from scipy.interpolate import CubicSpline
    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    P = dists0[ridx]
    with open(args.prompts) as f:
        cores_all = [p["text_core"] for p in json.load(f)["prompts"]]
    cores = [cores_all[i] for i in ridx]
    amax = P[:, :n].argmax(1)

    # restrict the cycle to concept values that have >=1 anchor (centroids undefined otherwise);
    # keeps the original cyclic order on the populated subset
    populated = [d for d in range(n) if (amax == d).any()]
    if len(populated) < n:
        print(f"WARNING: {n - len(populated)} concept values have no anchors "
              f"({[ABBR[d] for d in range(n) if d not in populated]}) — cycle restricted to {len(populated)}")
    remap = {d: i for i, d in enumerate(populated)}
    n_eff = len(populated)

    # endpoint pairs (ordered) in POPULATED-CYCLE index space (0..n_eff-1) — all centroid arrays,
    # arcs and splines are indexed in this space; raw token ids only via populated[i]
    all_pairs = [(i, j) for i in range(n_eff) for j in range(n_eff) if i != j]
    if args.n_pairs and args.n_pairs < len(all_pairs):
        sel = rng.choice(len(all_pairs), size=args.n_pairs, replace=False)
        pairs = [all_pairs[i] for i in sel]
    else:
        pairs = all_pairs
    car_idx = rng.choice(len(cores), size=min(args.n_carriers, len(cores)), replace=False)
    carriers = [cores[i] for i in car_idx]
    K = args.waypoints
    ts = np.linspace(0.0, 1.0, K)

    # behaviour manifold M_y (concept-level, layer-independent); pairs/arcs live in the
    # POPULATED-cycle index space (identical to token space when every value has anchors)
    bcent = np.stack([P[amax == populated[i]].mean(0) for i in range(n_eff)], 0)  # (n_eff, n+1)
    cs_y, bstar = fit_my_spline(np.sqrt(np.clip(bcent, 0, None)))
    MyS = my_samples(cs_y, bstar, n_eff)                                # (U, n+1) sqrt-space
    q_path = {}                                                         # M_y target distributions per pair
    for (a, b) in pairs:
        qs = np.clip(np.stack([sphere_exp(bstar, cs_y(cyc_arc(a, b, n_eff, t))) for t in ts], 0), 0, None) ** 2
        q_path[(a, b)] = qs / qs.sum(1, keepdims=True)

    holder = {"H": None}

    def hook(m, i, o):
        if holder["H"] is None:
            return o
        hs = (o[0] if isinstance(o, tuple) else o).clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        holder["H"] = None if holder.pop("once", False) else holder["H"]
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    out_all = {"concept": args.concept, "k": args.k, "n_pairs": len(pairs), "K": K,
               "n_carriers": len(carriers), "carrier_idx": car_idx.tolist(),
               "methods": METHODS, "seed": args.seed, "layers": {}}

    for L in [int(x) for x in args.layers.split(",")]:
        X = blob["activations"][:, L, :].float().numpy()[ridx]          # (M, H) raw
        mu = X.mean(0); sd = X.std(0) + 1e-6; Xz = (X - mu) / sd
        Y = np.sqrt(np.clip(P, 0, None))

        # --- k-D CCA subspace chart + linear forward map (ours) ---
        npc = min(40, Xz.shape[0] - 1)
        apca = PCA(n_components=npc).fit(Xz)
        k = min(args.k, npc, Y.shape[1])
        cca = CCA(n_components=k, max_iter=1000).fit(apca.transform(Xz), Y)
        Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T   # (k, H) std space
        c = Xz.mean(0); coords = (Xz - c) @ Q.T
        Cb = np.concatenate([coords, np.ones((len(coords), 1))], 1)
        Wb, *_ = np.linalg.lstsq(Cb, Y, rcond=None); W = Wb[:-1]; boff = Wb[-1]
        Wpinv = np.linalg.pinv(W)

        # --- PCA-64 + periodic spline through activation centroids (paper M_h) ---
        p64 = PCA(n_components=min(64, X.shape[0] - 1)).fit(X)
        Z64 = p64.transform(X)
        acent64 = np.stack([Z64[amax == populated[i]].mean(0) for i in range(n_eff)], 0)   # (n_eff, 64)
        cs_h = CubicSpline(np.arange(n_eff + 1, dtype=float),
                           np.concatenate([acent64, acent64[:1]], 0), axis=0, bc_type="periodic")
        acent_raw = np.stack([X[amax == populated[i]].mean(0) for i in range(n_eff)], 0)   # (n_eff, H)
        ccent_k = np.stack([coords[amax == populated[i]].mean(0) for i in range(n_eff)], 0)  # (n_eff, k)

        # carrier encodings + raw last-token activations at this layer (for residual-preserving)
        car_enc = [pipe.load([{"raw_input": wrap(cstr, args.frame)}]) for cstr in carriers]
        car_h = X[car_idx]                                                        # (C, H) raw
        car_hz = (car_h - mu) / sd
        car_z64 = p64.transform(car_h)
        car_coords = (car_hz - c) @ Q.T

        @torch.no_grad()
        def decode_on_carrier(Hraw, ci):
            """Decode rows of Hraw (B, H) patched into carrier ci. Returns (B, n+1)."""
            enc = car_enc[ci]; res = []
            for s in range(0, len(Hraw), args.batch):
                hb = Hraw[s:s + args.batch]; B = hb.shape[0]
                holder["H"] = torch.from_numpy(hb).to(dev)
                o = model(input_ids=enc["input_ids"].repeat(B, 1),
                          attention_mask=enc["attention_mask"].repeat(B, 1),
                          use_cache=False, logits_to_keep=1)
                holder["H"] = None
                dd, _ = concept_distributions(F.softmax(o.logits[:, -1, :].float(), -1).cpu(),
                                              variant_ids, canonical_ids)
                res.append(dd.numpy())
            return np.concatenate(res, 0)

        def waypoint_acts(method, a, b, ci):
            """(K, H) raw activations to patch for this method/pair/carrier."""
            if method == "linear_full":
                return np.stack([(1 - t) * acent_raw[a] + t * acent_raw[b] for t in ts], 0)
            if method == "manifold":
                z = np.stack([cs_h(cyc_arc(a, b, n_eff, t)) for t in ts], 0)      # (K, 64)
                base = car_z64[ci]
                return car_h[ci][None] + (z - base[None]) @ p64.components_
            if method in ("subspace_inv_full", "subspace_inv_resid"):
                tgt = q_path[(a, b)]
                z = (np.sqrt(np.clip(tgt, 0, None)) - boff) @ Wpinv               # (K, k)
                if method == "subspace_inv_full":
                    return ((c[None] + z @ Q) * sd + mu)
                base = car_coords[ci]
                return ((car_hz[ci][None] + (z - base[None]) @ Q) * sd + mu)
            if method == "linear_subspace":
                z = np.stack([(1 - t) * ccent_k[a] + t * ccent_k[b] for t in ts], 0)
                base = car_coords[ci]
                return ((car_hz[ci][None] + (z - base[None]) @ Q) * sd + mu)
            raise ValueError(method)

        handle = model.model.layers[L - 1].register_forward_hook(hook)
        layer_res = {m: [] for m in METHODS}
        for (a, b) in pairs:
            for m in METHODS:
                # batch: all carriers × K waypoints in one decode stream
                Hs, owner = [], []
                for ci in range(len(carriers)):
                    Hs.append(waypoint_acts(m, a, b, ci).astype(np.float32)); owner.append(ci)
                acc = np.zeros((K, n + 1))
                for ci, Hk in zip(owner, Hs):
                    acc += decode_on_carrier(Hk, ci)
                traj = acc / len(carriers)                                        # pointwise mean (paper)
                ebc = float(np.sum([ebc_to_my(traj[i], MyS) for i in range(K)]))
                amax_t = traj[:, :n].argmax(1)                                    # raw token ids
                arc = {populated[int(round(cyc_arc(a, b, n_eff, t))) % n_eff] for t in ts}
                offpath = float(np.mean([d not in arc for d in amax_t]))
                pos = [remap.get(int(d)) for d in amax_t]                         # populated-cycle positions
                jumps = [cyc_dist(pos[i], pos[i + 1], n_eff) for i in range(K - 1)
                         if pos[i] is not None and pos[i + 1] is not None]
                smooth = [hell(traj[i], traj[i + 1]) for i in range(K - 1)]
                valid = (traj[:, :n].sum(1) >= 0.90) & (traj[:, n] <= 0.10)
                layer_res[m].append({
                    "pair": [ABBR[populated[a]], ABBR[populated[b]]], "ebc_sum": round(ebc, 4),
                    "off_mass_mean": round(float(traj[:, n].mean()), 4),
                    "off_mass_max": round(float(traj[:, n].max()), 4),
                    "valid_frac": round(float(valid.mean()), 3),
                    "offpath_argmax_frac": round(offpath, 3),
                    "max_cyc_jump": int(max(jumps)) if jumps else 0,
                    "smooth_med": round(float(np.median(smooth)), 4),
                    "smooth_max": round(float(np.max(smooth)), 4),
                })
            print(f"L{L} pair {ABBR[populated[a]]}->{ABBR[populated[b]]} done", flush=True)

        # ---------- generation / fluency check ----------
        gsel = rng.choice(len(pairs), size=min(args.gen_pairs, len(pairs)), replace=False)
        gts = [K // 4, K // 2, (3 * K) // 4]
        gen_stats = {m: {"ppl": [], "uniq": [], "maxrep": [], "samples": []} for m in METHODS}
        for ci in range(min(args.gen_carriers, len(carriers))):
            enc = car_enc[ci]
            rows, meta = [], []
            for pi in gsel:
                a, b = pairs[pi]
                for m in METHODS:
                    Hk = waypoint_acts(m, a, b, ci)
                    for ti in gts:
                        rows.append(Hk[ti].astype(np.float32)); meta.append((m, a, b, ti))
            rows = np.stack(rows, 0)
            ids = enc["input_ids"].repeat(len(rows), 1)
            mask = enc["attention_mask"].repeat(len(rows), 1)
            with torch.no_grad():
                holder["H"] = torch.from_numpy(rows).to(dev); holder["once"] = True
                o = model(input_ids=ids, attention_mask=mask, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen = [nxt]
                for _ in range(args.gen_tokens - 1):
                    o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(nxt)
                gen_ids = torch.cat(gen, 1)                                       # (B, T)
                # fluency: ppl of tokens 2..T under the UNSTEERED model given prompt+tok1
                full = torch.cat([ids, gen_ids], 1)
                fmask = torch.cat([mask, torch.ones_like(gen_ids)], 1)
                o2 = model(input_ids=full, attention_mask=fmask, use_cache=False)
                lp = F.log_softmax(o2.logits.float(), -1)
                Tg = gen_ids.shape[1]; Lp = ids.shape[1]
                tgt = full[:, Lp + 1:]                                            # tokens 2..T
                pred = lp[:, Lp:-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
                nll = (-pred).mean(-1)
                ppl = torch.exp(nll).cpu().numpy()
            g = gen_ids.cpu().numpy()
            for r, (m, a, b, ti) in enumerate(meta):
                row = g[r]
                uniq = len(set(row.tolist())) / len(row)
                bg = list(zip(row[:-1].tolist(), row[1:].tolist()))
                maxrep = max(bg.count(x) for x in set(bg))
                gen_stats[m]["ppl"].append(float(ppl[r]))
                gen_stats[m]["uniq"].append(uniq)
                gen_stats[m]["maxrep"].append(maxrep)
                if ci == 0 and ti == gts[1] and len(gen_stats[m]["samples"]) < 3:
                    gen_stats[m]["samples"].append(
                        {"pair": f"{ABBR[populated[a]]}->{ABBR[populated[b]]}", "text": tok.decode(row)})
        handle.remove()

        def agg(vals):
            v = np.array(vals, dtype=float)
            return {"mean": round(float(v.mean()), 4), "se": round(float(v.std(ddof=1) / max(np.sqrt(len(v)), 1)), 4),
                    "median": round(float(np.median(v)), 4)}

        Lsum = {}
        for m in METHODS:
            rows = layer_res[m]
            Lsum[m] = {
                "ebc_sum": agg([r["ebc_sum"] for r in rows]),
                "off_mass_mean": agg([r["off_mass_mean"] for r in rows]),
                "valid_frac": agg([r["valid_frac"] for r in rows]),
                "offpath_argmax_frac": agg([r["offpath_argmax_frac"] for r in rows]),
                "max_cyc_jump": agg([r["max_cyc_jump"] for r in rows]),
                "smooth_max": agg([r["smooth_max"] for r in rows]),
                "gen_ppl": agg(gen_stats[m]["ppl"]) if gen_stats[m]["ppl"] else None,
                "gen_uniq_frac": agg(gen_stats[m]["uniq"]) if gen_stats[m]["uniq"] else None,
                "gen_max_bigram_rep": agg(gen_stats[m]["maxrep"]) if gen_stats[m]["maxrep"] else None,
                "gen_samples": gen_stats[m]["samples"],
                "per_pair": rows,
            }
        out_all["layers"][str(L)] = Lsum
        print(f"== L{L} E_BC: " + "  ".join(
            f"{m}={Lsum[m]['ebc_sum']['mean']:.2f}±{Lsum[m]['ebc_sum']['se']:.2f}" for m in METHODS), flush=True)

    tag = ("_" + args.tag) if args.tag else ""
    with open(os.path.join(args.experiment_root, f"steer_compare_{args.concept}{tag}.json"), "w") as f:
        json.dump(out_all, f, indent=2)
    print(f"saved -> steer_compare_{args.concept}{tag}.json")

    # summary figure: E_BC by method across layers (+ fluency ppl)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig_dir = os.path.join(args.experiment_root, "figures"); os.makedirs(fig_dir, exist_ok=True)
    layers = sorted(out_all["layers"].keys(), key=int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    w = 0.15
    xs = np.arange(len(layers))
    for j, m in enumerate(METHODS):
        mv = [out_all["layers"][L][m]["ebc_sum"]["mean"] for L in layers]
        se = [out_all["layers"][L][m]["ebc_sum"]["se"] for L in layers]
        axes[0].bar(xs + (j - 2) * w, mv, w, yerr=se, label=m, capsize=2)
        if out_all["layers"][layers[0]][m]["gen_ppl"]:
            pv = [out_all["layers"][L][m]["gen_ppl"]["median"] for L in layers]
            axes[1].plot(xs, pv, "o-", label=m)
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"L{L}" for L in layers])
    axes[0].set_ylabel("E_BC cumulative energy (lower = more natural)")
    axes[0].set_title(f"{args.concept}: naturalness of steering paths vs M_y")
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"L{L}" for L in layers])
    axes[1].set_ylabel("continuation ppl under unsteered model (median)")
    axes[1].set_title("post-steering fluency (24 greedy tokens)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"steer_compare_{args.concept}{tag}.{e}"))
    print("figure saved")


if __name__ == "__main__":
    main()
