"""Discover which color words are SINGLE tokens under the Llama-3.1 tokenizer.

The single-token colors define Z for the color probability simplex (we can only read a
clean next-token distribution over single-token classes). Scans a large candidate list,
keeps colors whose " <Color>" (leading-space, capitalized) form is exactly one token,
and writes colors.json (the token set + ids + which semantic groups each belongs to).

CPU only — run on the cinaps login node:
  HF_HUB_OFFLINE=1 uv run python <...>/color_tokens.py --out <...>/colors.json
"""

from __future__ import annotations

import argparse
import json
import os

# Curated candidate color names (basic + secondary + many named shades).
CANDIDATES = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "cyan", "magenta", "violet", "indigo", "turquoise", "teal", "maroon", "navy",
    "olive", "lime", "aqua", "gold", "silver", "beige", "tan", "peach", "coral", "salmon",
    "crimson", "scarlet", "lavender", "plum", "mint", "ivory", "charcoal", "amber", "azure",
    "cerulean", "vermilion", "ochre", "sienna", "umber", "mauve", "fuchsia", "chartreuse",
    "burgundy", "rose", "ruby", "emerald", "sapphire", "jade", "cobalt", "khaki", "mustard",
    "bronze", "copper", "brass", "rust", "wine", "cream", "snow", "ebony", "jet", "slate",
    "steel", "ash", "stone", "sand", "clay", "blush", "apricot", "tangerine", "lemon", "honey",
    "lilac", "periwinkle", "orchid", "fuschia", "raspberry", "cherry", "brick", "terracotta",
    "forest", "moss", "sage", "fern", "pine", "sky", "ocean", "sea", "denim", "royal",
    "midnight", "ink", "pearl", "platinum", "titanium", "graphite", "onyx", "coal",
    "lavender", "magenta", "turquoise", "burgundy",
]


def main():
    import torch  # noqa: F401  (ensures the uv env is the model env)
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "colors.json"))
    args = ap.parse_args()
    try:
        # apply the repo's transformers<5 patch (Gemma-4 ships list-valued
        # extra_special_tokens which AutoTokenizer otherwise chokes on)
        try:
            from causalab.neural.pipeline import _patch_extra_special_tokens
            _patch_extra_special_tokens()
        except Exception:
            pass
        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"FAILED to load tokenizer for {args.model}: {type(e).__name__}: {e}")
        return

    single, multi = [], []
    seen = set()
    for c in CANDIDATES:
        if c in seen:
            continue
        seen.add(c)
        cap = c.capitalize()
        enc_space = tok.encode(f" {cap}", add_special_tokens=False)
        enc_lower = tok.encode(f" {c}", add_special_tokens=False)
        rec = {"color": cap, "lower": c,
               "id_space_cap": enc_space, "id_space_lower": enc_lower,
               "single_cap": len(enc_space) == 1, "single_lower": len(enc_lower) == 1}
        # accept if EITHER capitalized or lowercase leading-space form is single-token
        if rec["single_cap"] or rec["single_lower"]:
            single.append(rec)
        else:
            multi.append(rec)

    out = {
        "n_single": len(single),
        "single_token_colors": [r["color"] for r in single],
        "detail": single,
        "multi_token_dropped": [r["color"] for r in multi],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"single-token colors ({len(single)}): {[r['color'] for r in single]}")
    print(f"dropped (multi-token, {len(multi)}): {[r['color'] for r in multi]}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
