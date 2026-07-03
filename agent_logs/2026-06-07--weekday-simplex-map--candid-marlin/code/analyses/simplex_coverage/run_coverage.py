"""Simplex-coverage analysis (session-local, step 1 of weekday-simplex-map).

Runs the weekday prompt set through a Llama model, reads the *first generated
token* distribution restricted to the 7 weekday tokens (+ an "other" class),
applies the mass filter (weekday mass >= 0.90 AND other <= 0.10), and evaluates
how well the retained distributions cover the behavioural simplex.

Two modes:
  --mode probe : run a ~26-prompt representative subset, write format_probe.json
                 (used to pick base vs instruct).
  --mode full  : run all prompts, write distributions + coverage metrics + plots.

Model wrappers:
  base  (use_chat_template=False): prompt = "Q: {core}\\nA:"  -> first token " Monday"
  instruct (use_chat_template=True): user turn = {core}, chat template applied.

Robust weekday-mass: per day we sum probability over single-token surface
variants {" Monday", "Monday", " monday"} (handles base vs chat capitalisation);
canonical (" Monday") mass is reported separately as a diagnostic.

Run from the repo root so `causalab` imports resolve:
  uv run python <session>/code/analyses/simplex_coverage/run_coverage.py \\
      --mode full --model llama31_8b \\
      --prompts <session>/code/methods/weekday_prompts/prompts.json \\
      --experiment-root <session>/artifacts/weekday_simplex/llama31_8b/simplex_coverage
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ABBR = [d[:3] for d in DAYS]

MODEL_HF = {
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "llama31_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

# ----------------------------------------------------------------------------
# Tokenisation: weekday token ids
# ----------------------------------------------------------------------------


def build_day_token_ids(tokenizer):
    """Return (variant_ids, canonical_ids, report).

    variant_ids[d]   : list of single-token ids for day d across surface variants.
    canonical_ids[d] : the single token id for " <Day>" (leading space), or None.
    report           : per-day tokenisation detail for the metrics file.
    """
    variant_ids = []
    canonical_ids = []
    report = {}
    for day in DAYS:
        variants = [f" {day}", day, f" {day.lower()}", day.lower()]
        ids = []
        detail = {}
        for v in variants:
            enc = tokenizer.encode(v, add_special_tokens=False)
            detail[v] = enc
            if len(enc) == 1:
                ids.append(enc[0])
        ids = sorted(set(ids))
        variant_ids.append(ids)
        canon = tokenizer.encode(f" {day}", add_special_tokens=False)
        canonical_ids.append(canon[0] if len(canon) == 1 else None)
        report[day] = {
            "variant_token_ids": ids,
            "canonical_single_token": len(canon) == 1,
            "encodings": detail,
        }
    return variant_ids, canonical_ids, report


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------


# LEAKING few-shot (reference only): 7 worked exemplars, one per answer-day so
# the day-prior stays uniform, 7 distinct constraint shapes so the format
# transfers. Stems are phrased to NOT collide verbatim with any test stem. This
# DEMONSTRATES ONLY-DAY ANSWERS, so it prescribes the answer space (leaks=True);
# kept as an upper-bound reference, not a candidate for the real run.
FEWSHOT_PREFIX = (
    "A day of the week, for instance: Wednesday\n"
    "Four days after Monday is Friday\n"
    "A day whose name begins with S: Sunday\n"
    "A weekday other than Friday: Tuesday\n"
    "A day on the weekend: Saturday\n"
    "Six days after Tuesday is Monday\n"
    "Three days before Sunday is Thursday\n"
)

# CLEAN few-shot for the BASE model: answer-first colon completions on NON-weekday
# tasks (color/fruit/letter/category/relational/ordinal) mirroring the test
# families' structure. Demonstrates "lead the completion with the answer" WITHOUT
# ever showing a day (no answer-space leak, leaks=False). Kept TERSE: empirically
# (run 12420 vs 12421) adding continuations ("..., for example") raises off-concept
# 'other' mass ~0.05 -> ~0.19 and pushes the spread families over the 10% line,
# while terse exemplars keep 'other' low AND do NOT collapse the spread (the base
# model still yields ~74% interior points — unlike an RLHF "one word" instruction).
NEUTRAL_FEWSHOT_PREFIX = (
    "A color: blue\n"
    "A fruit that is not an apple: banana\n"
    "A word that starts with the letter K: kite\n"
    "The letter after A: B\n"
    "The third letter of the alphabet: C\n"
    "The opposite of up: down\n"
    "Something you can drink: water\n"
)

# ---------------------------------------------------------------------------
# Prompt frames.
#
# METHODOLOGY: a good frame strips the *prose scaffolding* the model wraps around
# its answer (" is called Wednesday", " a day that ...") WITHOUT prescribing the
# answer space. Instructions that name the answer ("reply with a day") inflate
# weekday-mass artificially and stop measuring the model's genuine belief — so
# brevity/length instructions ("answer in one word", "answer briefly") are the
# right tool: they force concision without telling the model the answer must be a
# day. Frames flagged ``leaks=True`` DO prescribe a day (or demonstrate only-day
# answers) and are kept only as REFERENCE upper bounds to quantify the inflation;
# they are not candidates for the real run.
# ---------------------------------------------------------------------------
FRAMES = {
    # base / completion frames (no chat template)
    "plain":           {"prefix": "",                      "chat": False, "leaks": False},
    "qa":              {"prefix": None,                    "chat": False, "leaks": False},  # special "Q:\nA:"
    "fewshot_neutral": {"prefix": NEUTRAL_FEWSHOT_PREFIX,  "chat": False, "leaks": False},  # CLEAN: non-day exemplars
    "fewshot":         {"prefix": FEWSHOT_PREFIX,          "chat": False, "leaks": True},   # REF: demonstrates only-day answers
    # instruct / chat frames — brevity (clean) ...
    "i_oneword": {"prefix": "Answer with a single word.\n",        "chat": True, "leaks": False},
    "i_brief":   {"prefix": "Answer briefly.\n",                   "chat": True, "leaks": False},
    "i_concise": {"prefix": "Answer concisely.\n",                 "chat": True, "leaks": False},
    "i_short":   {"prefix": "Give a short answer.\n",              "chat": True, "leaks": False},
    "i_word3":   {"prefix": "Answer in no more than three words.\n", "chat": True, "leaks": False},
    # ... vs answer-space-leaking REFERENCE (not for the real run)
    "i_leak":    {"prefix": "Reply with exactly one day of the week and nothing else.\n", "chat": True, "leaks": True},
}


def frame_uses_chat(frame: str) -> bool:
    return FRAMES[frame]["chat"]


def load_model(model_key: str, use_chat: bool):
    from causalab.neural.pipeline import LMPipeline

    hf_name = MODEL_HF[model_key]
    pipe = LMPipeline(
        hf_name,
        max_new_tokens=1,
        use_chat_template=use_chat,
        device="cuda",
        dtype=torch.bfloat16,
        eager_attn=True,
    )
    return pipe


def wrap(core: str, frame: str) -> str:
    """Apply the prompt frame. ``core`` is a completion stem.

    For chat frames the prefix is prepended to the user turn; LMPipeline.load()
    then applies the chat template. For completion frames the result is fed to
    the base model verbatim.
    """
    spec = FRAMES[frame]
    if spec["prefix"] is None:  # qa
        return f"Q: {core}\nA:"
    return spec["prefix"] + core


@torch.no_grad()
def score_prompts(pipe, frame, cores, batch_size=32):
    """Return first-token full-vocab probabilities, (N, vocab) on CPU."""
    all_probs = []
    n = len(cores)
    for s in range(0, n, batch_size):
        batch = cores[s : s + batch_size]
        inputs = [{"raw_input": wrap(c, frame)} for c in batch]
        out = pipe.generate(inputs)
        logits = out["scores"][0]  # (B, vocab)
        probs = F.softmax(logits.float(), dim=-1)
        all_probs.append(probs)
    return torch.cat(all_probs, dim=0)


def day_distributions(full_probs, variant_ids, canonical_ids):
    """Map (N, vocab) -> per-day variant mass and canonical mass.

    Returns dists (N, 8) = [day_variant_mass..., other] and
    canon (N, 8) = [day_canonical_mass..., other].
    """
    N = full_probs.shape[0]
    day_var = torch.zeros(N, 7)
    day_can = torch.zeros(N, 7)
    for d in range(7):
        if variant_ids[d]:
            day_var[:, d] = full_probs[:, variant_ids[d]].sum(dim=-1)
        if canonical_ids[d] is not None:
            day_can[:, d] = full_probs[:, canonical_ids[d]]
    other_var = (1.0 - day_var.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    other_can = (1.0 - day_can.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    dists = torch.cat([day_var, other_var], dim=-1)
    canon = torch.cat([day_can, other_can], dim=-1)
    return dists, canon


def topk_tokens(full_probs, tokenizer, k=8):
    vals, ids = torch.topk(full_probs, k, dim=-1)
    out = []
    for i in range(full_probs.shape[0]):
        out.append(
            [
                {"token": tokenizer.decode([int(ids[i, j])]), "prob": round(float(vals[i, j]), 5)}
                for j in range(k)
            ]
        )
    return out


# ----------------------------------------------------------------------------
# Probe mode
# ----------------------------------------------------------------------------


def select_probe(prompts, per_family=2):
    by_fam = defaultdict(list)
    for p in prompts:
        by_fam[p["family"]].append(p)
    sel = []
    for fam in sorted(by_fam):
        sel.extend(by_fam[fam][:per_family])
    # ensure a few hard constraint prompts are present
    hard_keys = [
        "starts with the letter t",
        "neither",
        "weekend day that is not",
        "third day of the week",
        "contains the letter r",
    ]
    have = {p["text_core"].lower() for p in sel}
    for p in prompts:
        if any(k in p["text_core"].lower() for k in hard_keys) and p["text_core"].lower() not in have:
            sel.append(p)
            have.add(p["text_core"].lower())
    return sel


def _probe_one_frame(pipe, tok, variant_ids, canonical_ids, sel, frame, batch_size):
    cores = [p["text_core"] for p in sel]
    full = score_prompts(pipe, frame, cores, batch_size=batch_size)
    dists, canon = day_distributions(full, variant_ids, canonical_ids)
    tops = topk_tokens(full, tok, k=6)

    rows = []
    for i, p in enumerate(sel):
        wm = float(dists[i, :7].sum())
        other = float(dists[i, 7])
        amax = ABBR[int(dists[i, :7].argmax())]
        rows.append(
            {
                "family": p["family"],
                "text_core": p["text_core"],
                "intended_days": p["intended_days"],
                "weekday_mass": round(wm, 4),
                "weekday_mass_canonical": round(float(canon[i, :7].sum()), 4),
                "other": round(other, 4),
                "argmax_day": amax,
                "argmax_in_intended": amax in p["intended_days"],
                "passed_filter": (wm >= 0.90) and (other <= 0.10),
                "top_tokens": tops[i],
            }
        )
    n_pass = sum(r["passed_filter"] for r in rows)
    return {
        "model": None,
        "frame": frame,
        "leaks_answer_space": FRAMES[frame]["leaks"],
        "n_probe": len(sel),
        "n_pass_filter": n_pass,
        "frac_pass_filter": round(n_pass / max(1, len(sel)), 3),
        "frac_argmax_in_intended": round(
            sum(r["argmax_in_intended"] for r in rows) / max(1, len(rows)), 3
        ),
        "mean_weekday_mass": round(sum(r["weekday_mass"] for r in rows) / max(1, len(rows)), 3),
        "rows": rows,
    }


def run_probe(args, prompts):
    frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    chats = {frame_uses_chat(f) for f in frames}
    if len(chats) > 1:
        raise SystemExit(
            "All --frames in one probe invocation must share chat-mode "
            "(group base frames {plain,qa,fewshot} separately from chat frames i_*)."
        )
    use_chat = chats.pop()
    pipe = load_model(args.model, use_chat)
    tok = pipe.tokenizer
    variant_ids, canonical_ids, tok_report = build_day_token_ids(tok)
    sel = select_probe(prompts, per_family=args.probe_per_family)

    os.makedirs(args.experiment_root, exist_ok=True)
    comparison = []
    per_frame = {}
    for frame in frames:
        res = _probe_one_frame(pipe, tok, variant_ids, canonical_ids, sel, frame, args.batch_size)
        res["model"] = args.model
        res["tokenisation"] = tok_report
        per_frame[frame] = res
        with open(os.path.join(args.experiment_root, f"format_probe_{frame}.json"), "w") as f:
            json.dump(res, f, indent=2)
        comparison.append(res)

    with open(os.path.join(args.experiment_root, "format_probe.json"), "w") as f:
        json.dump({"model": args.model, "frames": per_frame}, f, indent=2)

    print(f"\n=== PROBE {args.model} — frame comparison (n_probe={len(sel)}) ===")
    print(f"{'frame':12s} {'leaks':5s} {'pass%':>6s} {'meanWmass':>9s} {'argmax_in_intended':>18s}")
    for r in sorted(comparison, key=lambda x: -x["frac_pass_filter"]):
        print(
            f"{r['frame']:12s} {str(r['leaks_answer_space']):5s} "
            f"{r['frac_pass_filter']*100:5.0f}% {r['mean_weekday_mass']:9.2f} "
            f"{r['frac_argmax_in_intended']*100:17.0f}%"
        )
    # per-family pass for the best non-leaking frame
    clean = [r for r in comparison if not r["leaks_answer_space"]]
    best = max(clean or comparison, key=lambda x: x["frac_pass_filter"])
    from collections import defaultdict as _dd
    fp = _dd(lambda: [0, 0])
    for row in best["rows"]:
        fp[row["family"]][0] += int(row["passed_filter"]); fp[row["family"]][1] += 1
    print(f"\nbest clean frame = {best['frame']} — per-family pass:")
    print("  " + "  ".join(f"{f}:{fp[f][0]}/{fp[f][1]}" for f in sorted(fp)))
    print("tokenisation single-token check:")
    for day, d in tok_report.items():
        print(f"  {day:10s} canonical_single={d['canonical_single_token']} ids={d['variant_token_ids']}")


# ----------------------------------------------------------------------------
# Full mode
# ----------------------------------------------------------------------------


def hellinger_pairwise(sqrt_dists: np.ndarray) -> np.ndarray:
    # H(p,q) = (1/sqrt2) ||sqrt(p)-sqrt(q)||_2
    from scipy.spatial.distance import pdist

    return pdist(sqrt_dists, metric="euclidean") / math.sqrt(2.0)


def run_full(args, prompts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from safetensors.torch import save_file

    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat)
    tok = pipe.tokenizer
    variant_ids, canonical_ids, tok_report = build_day_token_ids(tok)

    cores = [p["text_core"] for p in prompts]
    full = score_prompts(pipe, args.frame, cores, batch_size=args.batch_size)
    dists, canon = day_distributions(full, variant_ids, canonical_ids)
    tops = topk_tokens(full, tok, k=10)

    wm = dists[:, :7].sum(dim=-1).numpy()
    other = dists[:, 7].numpy()
    retained = (wm >= 0.90) & (other <= 0.10)

    # per-prompt scored records
    scored = []
    for i, p in enumerate(prompts):
        amax = ABBR[int(dists[i, :7].argmax())]
        scored.append(
            {
                "id": p["id"],
                "family": p["family"],
                "text_core": p["text_core"],
                "intended_days": p["intended_days"],
                "weekday_mass": round(float(wm[i]), 4),
                "weekday_mass_canonical": round(float(canon[i, :7].sum()), 4),
                "other": round(float(other[i]), 4),
                "day_probs": {ABBR[d]: round(float(dists[i, d]), 4) for d in range(7)},
                "argmax_day": amax,
                "argmax_in_intended": amax in p["intended_days"],
                "retained": bool(retained[i]),
                "top_tokens": tops[i],
            }
        )

    os.makedirs(args.experiment_root, exist_ok=True)
    fig_dir = os.path.join(args.experiment_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    save_file(
        {"dists": dists.contiguous(), "canonical": canon.contiguous(),
         "retained": torch.tensor(retained, dtype=torch.uint8)},
        os.path.join(args.experiment_root, "distributions.safetensors"),
    )
    with open(os.path.join(args.experiment_root, "prompts_scored.json"), "w") as f:
        json.dump({"model": args.model, "n": len(scored), "prompts": scored}, f, indent=2)

    # ---- coverage metrics on retained set ----
    R = dists[retained].numpy()  # (M, 8)
    M = R.shape[0]
    metrics = {
        "model": args.model,
        "frame": args.frame,
        "use_chat_template": use_chat,
        "n_total": len(prompts),
        "n_retained": int(M),
        "frac_retained": round(M / len(prompts), 3),
        "tokenisation_all_single_token": all(
            d["canonical_single_token"] for d in tok_report.values()
        ),
    }
    # per-family yield
    fam_total = Counter(p["family"] for p in prompts)
    fam_keep = Counter(p["family"] for i, p in enumerate(prompts) if retained[i])
    metrics["per_family_yield"] = {
        k: {"kept": fam_keep.get(k, 0), "total": fam_total[k]} for k in sorted(fam_total)
    }

    if M >= 5:
        sqrtR = np.sqrt(np.clip(R, 0, None))
        # Hellinger PCA (full rank to read EVR curve)
        pca = PCA(n_components=min(8, M))
        coords = pca.fit_transform(sqrtR)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        n_for_90 = int(np.searchsorted(cum, 0.90) + 1)
        lam = pca.explained_variance_
        participation_ratio = float((lam.sum() ** 2) / np.clip((lam ** 2).sum(), 1e-12, None))
        metrics["hellinger_pca"] = {
            "explained_variance_ratio": [round(float(x), 4) for x in evr],
            "cumulative": [round(float(x), 4) for x in cum],
            "n_dims_for_90pct_var": n_for_90,
            "participation_ratio": round(participation_ratio, 3),
        }

        # entropy / interiority (over 7 days, renormalised)
        q = R[:, :7] / np.clip(R[:, :7].sum(axis=1, keepdims=True), 1e-12, None)
        ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(axis=1)
        logK = math.log(7)
        frac_interior = float((ent >= 0.5 * logK).mean())
        metrics["entropy"] = {
            "mean": round(float(ent.mean()), 3),
            "max_possible": round(logK, 3),
            "frac_interior_ge_half_logK": round(frac_interior, 3),
            "min": round(float(ent.min()), 3),
            "max": round(float(ent.max()), 3),
        }

        # per-day argmax coverage
        amax = R[:, :7].argmax(axis=1)
        day_counts = {ABBR[d]: int((amax == d).sum()) for d in range(7)}
        metrics["argmax_day_counts"] = day_counts
        metrics["all_days_reachable"] = all(v > 0 for v in day_counts.values())

        # pairwise Hellinger spread
        pw = hellinger_pairwise(sqrtR)
        metrics["pairwise_hellinger"] = {
            "min": round(float(pw.min()), 4),
            "p10": round(float(np.percentile(pw, 10)), 4),
            "median": round(float(np.median(pw)), 4),
            "p90": round(float(np.percentile(pw, 90)), 4),
            "max": round(float(pw.max()), 4),
        }

        # convex hull volume in 3D Hellinger-PCA, vs hull of 7 one-hot vertices
        try:
            from scipy.spatial import ConvexHull

            hull3 = ConvexHull(coords[:, :3])
            # reference: 7 vertices (one-hot days) projected via same PCA
            vtx = np.zeros((7, 8))
            for d in range(7):
                vtx[d, d] = 1.0
            vtx_coords = pca.transform(np.sqrt(vtx))[:, :3]
            ref = ConvexHull(vtx_coords)
            metrics["hull_volume_3d"] = {
                "retained": round(float(hull3.volume), 5),
                "seven_vertices_ref": round(float(ref.volume), 5),
                "ratio": round(float(hull3.volume / max(ref.volume, 1e-9)), 3),
            }
        except Exception as e:
            metrics["hull_volume_3d"] = {"error": str(e)}

        # ---- plots ----
        fams = sorted(set(p["family"] for p in prompts))
        fam_color = {f: plt.cm.tab20(i % 20) for i, f in enumerate(fams)}
        ret_prompts = [p for i, p in enumerate(prompts) if retained[i]]

        # 1) PCA scatter by family
        fig, ax = plt.subplots(figsize=(8, 7))
        for f in fams:
            idx = [j for j, p in enumerate(ret_prompts) if p["family"] == f]
            if idx:
                ax.scatter(coords[idx, 0], coords[idx, 1], s=22, color=fam_color[f], label=f, alpha=0.8)
        ax.set_title(f"Hellinger-PCA of retained dists by family ({args.model}, M={M})")
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "pca_scatter_by_family.pdf"))
        plt.close(fig)

        # 2) PCA scatter by argmax day
        fig, ax = plt.subplots(figsize=(8, 7))
        day_cmap = plt.cm.rainbow(np.linspace(0, 1, 7))
        for d in range(7):
            idx = np.where(amax == d)[0]
            if len(idx):
                ax.scatter(coords[idx, 0], coords[idx, 1], s=22, color=day_cmap[d], label=ABBR[d], alpha=0.8)
        ax.set_title(f"Hellinger-PCA by argmax day ({args.model}, M={M})")
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "pca_scatter_by_argmax_day.pdf"))
        plt.close(fig)

        # 3) entropy histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(ent, bins=20, color="steelblue", edgecolor="k", alpha=0.8)
        ax.axvline(0.5 * logK, color="r", ls="--", label="0.5 log 7 (interior threshold)")
        ax.axvline(logK, color="g", ls=":", label="log 7 (uniform)")
        ax.set_title(f"Entropy over 7 days, retained ({args.model})")
        ax.set_xlabel("entropy (nats)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "entropy_hist.pdf"))
        plt.close(fig)

        # 4) pairwise Hellinger histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(pw, bins=30, color="darkorange", edgecolor="k", alpha=0.8)
        ax.set_title(f"Pairwise Hellinger distances, retained ({args.model})")
        ax.set_xlabel("Hellinger distance")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "pairwise_hellinger_hist.pdf"))
        plt.close(fig)

        # 5) per-family centroid heatmap (mean 7-day dist per family, retained)
        fam_centroids = []
        fam_labels = []
        for f in fams:
            idx = [j for j, p in enumerate(ret_prompts) if p["family"] == f]
            if idx:
                c = R[idx, :7].mean(axis=0)
                c = c / max(c.sum(), 1e-9)
                fam_centroids.append(c)
                fam_labels.append(f"{f} (n={len(idx)})")
        if fam_centroids:
            fig, ax = plt.subplots(figsize=(7, 0.5 * len(fam_centroids) + 2))
            im = ax.imshow(np.array(fam_centroids), aspect="auto", cmap="viridis")
            ax.set_xticks(range(7))
            ax.set_xticklabels(ABBR)
            ax.set_yticks(range(len(fam_labels)))
            ax.set_yticklabels(fam_labels, fontsize=8)
            ax.set_title(f"Per-family mean day distribution ({args.model})")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "family_centroid_heatmap.pdf"))
            plt.close(fig)

        # 6) optional interactive 3D
        try:
            import plotly.graph_objects as go

            fig3 = go.Figure()
            for d in range(7):
                idx = np.where(amax == d)[0]
                if len(idx):
                    fig3.add_trace(
                        go.Scatter3d(
                            x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
                            mode="markers", name=ABBR[d],
                            marker=dict(size=4),
                            text=[ret_prompts[j]["text_core"] for j in idx],
                        )
                    )
            fig3.update_layout(title=f"Hellinger-PCA 3D ({args.model}, M={M})")
            fig3.write_html(os.path.join(fig_dir, "simplex_3d.html"))
        except Exception as e:
            metrics["plotly_3d_skipped"] = str(e)
    else:
        metrics["error"] = f"only {M} retained (<5); coverage metrics skipped"

    with open(os.path.join(args.experiment_root, "coverage_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== FULL {args.model} ===")
    print(f"retained {M}/{len(prompts)} = {metrics['frac_retained']:.0%}")
    if M >= 5:
        hp = metrics["hellinger_pca"]
        print(f"Hellinger-PCA: {hp['n_dims_for_90pct_var']} dims for 90% var, "
              f"participation ratio {hp['participation_ratio']}")
        print(f"interior frac {metrics['entropy']['frac_interior_ge_half_logK']}, "
              f"all days reachable: {metrics['all_days_reachable']}")
        print(f"argmax day counts: {metrics['argmax_day_counts']}")
        print(f"pairwise Hellinger median {metrics['pairwise_hellinger']['median']} "
              f"(min {metrics['pairwise_hellinger']['min']}, max {metrics['pairwise_hellinger']['max']})")
    print(f"artifacts -> {args.experiment_root}")


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "full"], required=True)
    ap.add_argument("--model", choices=list(MODEL_HF), required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--probe-per-family", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    # probe sweeps several frames in one model load (must share chat-mode);
    # full uses exactly one frame.
    ap.add_argument("--frames", default="plain", help="probe: comma-separated frame list")
    ap.add_argument("--frame", default="plain", choices=list(FRAMES), help="full: single frame")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with open(args.prompts) as f:
        prompts = json.load(f)["prompts"]

    if args.mode == "probe":
        run_probe(args, prompts)
    else:
        run_full(args, prompts)


if __name__ == "__main__":
    main()
