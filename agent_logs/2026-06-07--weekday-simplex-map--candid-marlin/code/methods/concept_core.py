"""Concept-general core for the simplex<->activation mapping routine.

Generalises the weekday-specific token/distribution helpers to ANY concept that
defines a probability simplex over a token set Z (weekdays, months, ...). The
mapping machinery (capture_concept.py, map_subspace.py) is concept-agnostic given
this module; only the token set + prompt set change per concept.

Reuses the concept-AGNOSTIC frame/model helpers from run_coverage (load_model,
wrap, FRAMES, frame_uses_chat) — the neutral few-shot prefix contains no concept
tokens, so it transfers across concepts.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyses", "simplex_coverage"))
from run_coverage import FRAMES, frame_uses_chat, load_model, wrap  # noqa: E402,F401

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

CONCEPTS = {
    "weekdays": {"tokens": WEEKDAYS, "abbr": [t[:3] for t in WEEKDAYS]},
    "months": {"tokens": MONTHS, "abbr": [t[:3] for t in MONTHS]},
}


# Curated canonical colour names. The colour concept Z = these ∩ {single-token under
# the tokenizer} — drops ambiguous object/material words (Sky, Ocean, Jet, Coal, Pearl…)
# that color_tokens.py picks up but that aren't really colours, giving a clean simplex.
CANONICAL_COLORS = [
    "Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Brown", "Black", "White",
    "Gray", "Grey", "Cyan", "Magenta", "Violet", "Indigo", "Turquoise", "Teal", "Maroon", "Navy",
    "Olive", "Lime", "Aqua", "Gold", "Silver", "Beige", "Tan", "Peach", "Coral", "Salmon",
    "Crimson", "Scarlet", "Lavender", "Plum", "Mint", "Ivory", "Charcoal", "Amber", "Azure",
    "Rose", "Ruby", "Emerald", "Sapphire", "Jade", "Mustard", "Burgundy", "Mauve", "Khaki",
    "Cream", "Bronze", "Copper",
]


# Curated 12-hue colour WHEEL (cyclic, hue-ordered, all single-token in Llama-3.1) — the
# colour analogue of the weekday cycle. Restricting to this wheel removes the 45-gamut's
# materials/shades/near-synonyms whose effective dim (~10) makes sparse recovery ill-posed.
HUES12 = ["Red", "Orange", "Yellow", "Lime", "Green", "Teal",
          "Cyan", "Azure", "Blue", "Violet", "Purple", "Pink"]


def get_concept(name):
    """Return (tokens, abbr). 'colors' = CANONICAL_COLORS ∩ single-token set discovered by
    color_tokens.py (colors.json), since the colour simplex Z is defined by the tokenizer."""
    if name == "hues12":
        return HUES12, HUES12
    if name == "colors":
        import json
        cj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_prompts", "colors.json")
        single = set(json.load(open(cj))["single_token_colors"])
        toks = [c for c in CANONICAL_COLORS if c in single]  # curated + single-token, stable order
        return toks, toks
    c = CONCEPTS[name]
    return c["tokens"], c["abbr"]


def build_concept_token_ids(tokenizer, tokens):
    """Return (variant_ids, canonical_ids, report) for a concept token list.

    variant_ids[i]   : single-token ids for token i across surface variants.
    canonical_ids[i] : the " <Token>" (leading-space) single token id, or None.
    """
    variant_ids, canonical_ids, report = [], [], {}
    for w in tokens:
        variants = [f" {w}", w, f" {w.lower()}", w.lower()]
        ids, detail = [], {}
        for v in variants:
            enc = tokenizer.encode(v, add_special_tokens=False)
            detail[v] = enc
            if len(enc) == 1:
                ids.append(enc[0])
        ids = sorted(set(ids))
        variant_ids.append(ids)
        canon = tokenizer.encode(f" {w}", add_special_tokens=False)
        canonical_ids.append(canon[0] if len(canon) == 1 else None)
        report[w] = {"variant_token_ids": ids, "canonical_single_token": len(canon) == 1, "encodings": detail}
    return variant_ids, canonical_ids, report


def concept_distributions(full_probs, variant_ids, canonical_ids):
    """(N, vocab) -> dists (N, n+1)=[token masses..., other], canon (N, n+1)."""
    n = len(variant_ids)
    N = full_probs.shape[0]
    var = torch.zeros(N, n)
    can = torch.zeros(N, n)
    for i in range(n):
        if variant_ids[i]:
            var[:, i] = full_probs[:, variant_ids[i]].sum(dim=-1)
        if canonical_ids[i] is not None:
            can[:, i] = full_probs[:, canonical_ids[i]]
    other_v = (1.0 - var.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    other_c = (1.0 - can.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    return torch.cat([var, other_v], dim=-1), torch.cat([can, other_c], dim=-1)
