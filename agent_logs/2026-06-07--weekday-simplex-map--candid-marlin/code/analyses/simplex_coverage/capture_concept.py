"""Concept-general: run a concept's prompts, capture activations + distributions,
filter to the valid behavioural region, and summarise simplex coverage — in one pass.

Combines coverage (run_coverage full-mode metrics) and activation capture
(capture_activations) generalised to ANY concept (weekdays, months, ...). Saves the
same activations.safetensors layout map_subspace.py / perturb_map.py consume:
  activations (N, L+1, H) bf16, dists (N, n+1) fp32, retained (N,) uint8.

Run from repo root:
  uv run python <...>/capture_concept.py --concept months --frame fewshot_neutral \
     --prompts <...>/month_prompts/prompts.json --experiment-root <...>/months/.../simplex_coverage
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "methods"))
from concept_core import (  # noqa: E402
    build_concept_token_ids,
    concept_distributions,
    frame_uses_chat,
    get_concept,
    load_model,
    wrap,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    tokens, abbr = get_concept(args.concept)
    n = len(tokens)
    with open(args.prompts) as f:
        prompts = json.load(f)["prompts"]
    cores = [p["text_core"] for p in prompts]

    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat)
    tok = pipe.tokenizer
    model = pipe.model; model.eval()
    variant_ids, canonical_ids, tok_report = build_concept_token_ids(tok, tokens)
    n_layers = int(model.config.num_hidden_layers)
    print(f"concept={args.concept} n={n} layers={n_layers} n_prompts={len(prompts)}")

    logits_all, acts = [], []
    for s in range(0, len(cores), args.batch_size):
        batch = cores[s:s + args.batch_size]
        enc = pipe.load([{"raw_input": wrap(c, args.frame)} for c in batch])
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True, use_cache=False, logits_to_keep=1)
        logits_all.append(out.logits[:, -1, :].float().cpu())
        acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1).to(torch.bfloat16).cpu())
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {s+len(batch)}/{len(cores)}", flush=True)

    probs = F.softmax(torch.cat(logits_all, 0), dim=-1)
    activations = torch.cat(acts, 0)                       # (N, L+1, H)
    dists, canon = concept_distributions(probs, variant_ids, canonical_ids)  # (N, n+1)
    wm = dists[:, :n].sum(dim=-1); other = dists[:, n]
    retained = (wm >= 0.90) & (other <= 0.10)

    os.makedirs(args.experiment_root, exist_ok=True)
    from safetensors.torch import save_file
    save_file({"activations": activations.contiguous(), "dists": dists.float().contiguous(),
               "retained": retained.to(torch.uint8).contiguous()},
              os.path.join(args.experiment_root, "activations.safetensors"))

    # ---- coverage summary ----
    from collections import Counter
    from sklearn.decomposition import PCA
    D = dists.numpy(); R = D[retained.numpy().astype(bool)]
    M = R.shape[0]
    summary = {"concept": args.concept, "model": args.model, "frame": args.frame, "n_tokens": n,
               "n_total": len(prompts), "n_retained": int(M), "frac_retained": round(M / len(prompts), 3),
               "all_tokens_single": all(d["canonical_single_token"] for d in tok_report.values())}
    famT = Counter(p["family"] for p in prompts)
    famK = Counter(p["family"] for i, p in enumerate(prompts) if retained[i])
    summary["per_family_yield"] = {k: {"kept": famK.get(k, 0), "total": famT[k]} for k in sorted(famT)}
    if M >= n:
        sq = np.sqrt(np.clip(R, 0, None))
        pca = PCA(n_components=min(n + 1, M)).fit(sq)
        evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
        lam = pca.explained_variance_
        q = R[:, :n] / np.clip(R[:, :n].sum(1, keepdims=True), 1e-12, None)
        ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(1); logK = math.log(n)
        amax = R[:, :n].argmax(1)
        summary["hellinger_pca"] = {"n_dims_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
                                    "participation_ratio": round(float(lam.sum() ** 2 / np.clip((lam ** 2).sum(), 1e-12, None)), 2),
                                    "evr_head": [round(float(x), 3) for x in evr[:min(12, len(evr))]]}
        summary["entropy"] = {"mean": round(float(ent.mean()), 3), "max_logK": round(logK, 3),
                              "frac_interior_ge_half_logK": round(float((ent >= 0.5 * logK).mean()), 3)}
        summary["argmax_counts"] = {abbr[d]: int((amax == d).sum()) for d in range(n)}
        summary["all_tokens_reachable"] = all(v > 0 for v in summary["argmax_counts"].values())
    with open(os.path.join(args.experiment_root, "coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    meta = {"concept": args.concept, "tokens": tokens, "abbr": abbr, "n_tokens": n,
            "frame": args.frame, "n_prompts": len(prompts), "n_retained": int(retained.sum()),
            "n_layers_plus_embed": int(activations.shape[1]), "hidden_size": int(activations.shape[2]),
            "families": [p["family"] for p in prompts], "retained_mask": retained.tolist()}
    with open(os.path.join(args.experiment_root, "activations_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"retained {int(retained.sum())}/{len(prompts)}; "
          f"PCA dims90={summary.get('hellinger_pca', {}).get('n_dims_for_90pct')}; "
          f"interior={summary.get('entropy', {}).get('frac_interior_ge_half_logK')}; "
          f"all_reachable={summary.get('all_tokens_reachable')}")
    print(f"saved -> {args.experiment_root}")


if __name__ == "__main__":
    main()
