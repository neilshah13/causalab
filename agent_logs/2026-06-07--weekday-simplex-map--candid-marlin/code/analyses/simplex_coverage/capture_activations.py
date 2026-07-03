"""Capture last-token residual-stream activations for the weekday prompt set.

Step 2 of weekday-simplex-map. For each prompt we do ONE forward pass (no
generation needed: max_new_tokens=1 means the first-token logits are just the
logits at the last position) and store, at the last token position:

  - the next-token distribution over [7 days, other]  (the behaviour point p(x))
  - the residual-stream hidden state at EVERY layer    (the activations h_ℓ(x))

Paired (p(x), h_ℓ(x)) per prompt is exactly what the behaviour-vs-activation
SUBSPACE comparison (subspace_compare.py) consumes. We do NOT fit manifolds here.

Uses the WINNING frame: base llama31_8b + neutral few-shot + answer-first stems.

Run from repo root:
  uv run python <session>/code/analyses/simplex_coverage/capture_activations.py \
     --model llama31_8b --frame fewshot_neutral \
     --prompts <...>/prompts.json --experiment-root <...>/simplex_coverage
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

# import shared helpers from the sibling coverage script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_coverage import (  # noqa: E402
    MODEL_HF,
    ABBR,
    build_day_token_ids,
    day_distributions,
    frame_uses_chat,
    load_model,
    wrap,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_HF), default="llama31_8b")
    ap.add_argument("--frame", default="fewshot_neutral")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype-save", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)["prompts"]
    cores = [p["text_core"] for p in prompts]

    use_chat = frame_uses_chat(args.frame)
    pipe = load_model(args.model, use_chat)
    tok = pipe.tokenizer
    model = pipe.model
    model.eval()
    variant_ids, canonical_ids, tok_report = build_day_token_ids(tok)

    n_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    print(f"model={args.model} layers={n_layers} hidden={hidden} n_prompts={len(prompts)}")

    all_last_logits = []          # (N, vocab) cpu float32
    acts = []                     # (N, n_layers+1, hidden) cpu
    save_dtype = torch.bfloat16 if args.dtype_save == "bf16" else torch.float32

    for s in range(0, len(cores), args.batch_size):
        batch = cores[s : s + args.batch_size]
        inputs = [{"raw_input": wrap(c, args.frame)} for c in batch]
        enc = pipe.load(inputs)  # left-padded; last token at index -1
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        # last-token logits -> distribution
        last_logits = out.logits[:, -1, :].float().cpu()
        all_last_logits.append(last_logits)
        # last-token hidden state at every layer: tuple len (n_layers+1), each (B,seq,H)
        hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)  # (B, L+1, H)
        acts.append(hs.to(save_dtype).cpu())
        del out, hs, last_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  batch {s//args.batch_size}: {s+len(batch)}/{len(cores)}", flush=True)

    full_logits = torch.cat(all_last_logits, dim=0)            # (N, vocab)
    activations = torch.cat(acts, dim=0)                        # (N, L+1, H)
    probs = F.softmax(full_logits, dim=-1)
    dists, canon = day_distributions(probs, variant_ids, canonical_ids)  # (N,8)
    wm = dists[:, :7].sum(dim=-1)
    other = dists[:, 7]
    retained = (wm >= 0.90) & (other <= 0.10)

    os.makedirs(args.experiment_root, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {
            "activations": activations.contiguous(),        # (N, L+1, H)
            "dists": dists.float().contiguous(),            # (N, 8)
            "retained": retained.to(torch.uint8).contiguous(),
        },
        os.path.join(args.experiment_root, "activations.safetensors"),
    )
    meta = {
        "model": args.model,
        "frame": args.frame,
        "n_prompts": len(prompts),
        "n_retained": int(retained.sum()),
        "n_layers_plus_embed": activations.shape[1],
        "hidden_size": activations.shape[2],
        "save_dtype": args.dtype_save,
        "token_position": "last",
        "ids": [p["id"] for p in prompts],
        "families": [p["family"] for p in prompts],
        "argmax_day": [ABBR[int(dists[i, :7].argmax())] for i in range(len(prompts))],
        "retained_mask": retained.tolist(),
        "tokenisation_all_single_token": all(
            d["canonical_single_token"] for d in tok_report.values()
        ),
    }
    with open(os.path.join(args.experiment_root, "activations_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"saved activations {tuple(activations.shape)} ({args.dtype_save}); "
        f"retained {int(retained.sum())}/{len(prompts)} -> {args.experiment_root}"
    )


if __name__ == "__main__":
    main()
