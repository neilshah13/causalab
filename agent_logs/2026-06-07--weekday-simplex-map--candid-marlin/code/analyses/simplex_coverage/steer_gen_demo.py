"""Readable steered-generation demo for the slides.

The slide-3 text panel showed continuations of the neutral few-shot carrier, which read as
nonsense out of context ("Friday ... A color: red ... a fruit that is not an apple"). This
generates steered continuations on NATURAL PROSE carriers instead: patch the last-token
residual at the layer with the subspace-steered activation for a target day, then generate
greedily and record the text. Also records the unsteered continuation for contrast.

Run from repo root (GPU + activations.safetensors).
"""

from __future__ import annotations

import argparse
import json
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

CARRIERS = [
    "I glanced at the calendar on the wall to check the day. It was",
    "She asked me what my favorite day of the week was, and I said",
    "The meeting got moved again. It will now happen on",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=list(MODEL_HF))
    ap.add_argument("--concept", default="weekdays")
    ap.add_argument("--frame", default="fewshot_neutral", help="frame for the ANCHOR chart only")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--targets", default="Wed,Sat,Sun")
    ap.add_argument("--gen-tokens", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tokens, ABBR = get_concept(args.concept); n = len(tokens)
    pipe = load_model(args.model, False); model = pipe.model; model.eval()
    tok = pipe.tokenizer
    variant_ids, canonical_ids, _ = build_concept_token_ids(tok, tokens)
    dev = next(model.parameters()).device
    L = args.layer; holder = {"H": None}

    def hook(m, i, o):
        if holder["H"] is None:
            return o
        hs = (o[0] if isinstance(o, tuple) else o).clone()
        hs[:, -1, :] = holder["H"].to(hs.dtype)
        holder["H"] = None
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
    handle = model.model.layers[L - 1].register_forward_hook(hook)

    from safetensors.torch import load_file
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    blob = load_file(os.path.join(args.experiment_root, "activations.safetensors"))
    X = blob["activations"][:, L, :].float().numpy()
    dists0 = blob["dists"].float().numpy()
    ridx = np.where(blob["retained"].numpy().astype(bool))[0]
    X = X[ridx]; P = dists0[ridx]
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xz = (X - mu) / sd
    Y = np.sqrt(np.clip(P, 0, None))
    apca = PCA(n_components=min(40, Xz.shape[0] - 1)).fit(Xz)
    k = min(args.k, Y.shape[1])
    cca = CCA(n_components=k, max_iter=1000).fit(apca.transform(Xz), Y)
    Q, _ = np.linalg.qr((cca.x_rotations_.T @ apca.components_).T); Q = Q.T
    c = Xz.mean(0); coords = (Xz - c) @ Q.T
    Cb = np.concatenate([coords, np.ones((len(coords), 1))], 1)
    Wb, *_ = np.linalg.lstsq(Cb, Y, rcond=None); W = Wb[:-1]; b = Wb[-1]
    Wpinv = np.linalg.pinv(W)

    def steered_H(day):
        t = np.zeros(n + 1); t[day] = 1.0
        z = (np.sqrt(t) - b) @ Wpinv
        return ((c + z @ Q) * sd + mu).astype(np.float32)

    @torch.no_grad()
    def generate(carrier, H):
        enc = tok(carrier, return_tensors="pt").to(dev)
        holder["H"] = None if H is None else torch.from_numpy(H[None]).to(dev)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True)
        past = out.past_key_values
        first = F.softmax(out.logits[:, -1, :].float(), -1)
        dd, _ = concept_distributions(first.cpu(), variant_ids, canonical_ids)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [nxt]
        for _ in range(args.gen_tokens - 1):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
        text = tok.decode(torch.cat(gen, 1)[0])
        return text, dd[0].numpy()

    results = []
    tgt_ids = [ABBR.index(x) for x in args.targets.split(",")]
    for carrier in CARRIERS:
        base_text, base_dd = generate(carrier, None)
        entry = {"carrier": carrier, "unsteered": {"text": base_text,
                 "top_day": ABBR[int(base_dd[:n].argmax())], "P": round(float(base_dd[:n].max()), 3)}}
        for day in tgt_ids:
            txt, dd = generate(carrier, steered_H(day))
            entry[ABBR[day]] = {"text": txt, "P_target": round(float(dd[day]), 3),
                                "argmax_is_target": bool(dd[:n].argmax() == day)}
            print(f"[{ABBR[day]}] {carrier!r} -> {txt!r} (P={dd[day]:.2f})", flush=True)
        results.append(entry)
    handle.remove()

    out_path = os.path.join(args.experiment_root, f"steer_gen_demo_L{L}.json")
    with open(out_path, "w") as f:
        json.dump({"layer": L, "k": int(k), "results": results}, f, indent=2)
    print(f"saved -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
