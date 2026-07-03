"""Hue-wheel prompt set (Δ¹² over 12 single-token hue names, cyclic like weekdays/months).

The 45-colour gamut is messy (materials, shades, near-synonyms; effective dim ~10 ≪ 45) —
sparse recovery fails on it and the "ring" question is ill-posed. This restricts to a curated
12-hue COLOUR WHEEL, all single-token in Llama-3.1, hue-ordered so cyclic adjacency is
meaningful: the direct analogue of the weekday cycle, with a paper-aligned prediction
(hue centroids should form a ring in the top PCs, like weekdays' Tue→…→Mon cycle).

Families mirror the weekday set: open (interior), negation (faces), between/mix (edges,
interior), association (vertices), first-letter (edges, incl. the bimodal P→{Purple,Pink}),
temperature (warm/cool faces). All answer-first colon stems.
"""

import json
import os

HUES = ["Red", "Orange", "Yellow", "Lime", "Green", "Teal",
        "Cyan", "Azure", "Blue", "Violet", "Purple", "Pink"]


def build():
    P = []

    def add(fam, text):
        P.append({"text_core": text, "family": fam})

    for t in ["A color:", "A color of the rainbow:", "A bright color:",
              "A primary color:", "A color of the visible spectrum:"]:
        add("A_open", t)

    for h in HUES:
        add("B_neg1", f"A color of the rainbow that is not {h.lower()}:")

    for i, h in enumerate(HUES):
        h2 = HUES[(i + 1) % len(HUES)]
        add("C_neg2", f"A color that is neither {h.lower()} nor {h2.lower()}:")

    for i, h in enumerate(HUES):
        h2 = HUES[(i + 2) % len(HUES)]
        add("D_between", f"The color between {h.lower()} and {h2.lower()} on the color wheel:")

    assoc = [("the sky on a clear day", None), ("fresh grass", None), ("a ripe lemon", None),
             ("blood", None), ("a carrot", None), ("a flamingo", None), ("an eggplant", None),
             ("a pumpkin", None), ("the deep ocean", None), ("a lime", None),
             ("a violet flower", None), ("a stop sign", None), ("a banana", None),
             ("cotton candy", None), ("tropical shallow water", None), ("new spring leaves", None)]
    for obj, _ in assoc:
        add("E_assoc", f"The color of {obj}:")

    mixes = [("red", "yellow"), ("blue", "yellow"), ("red", "blue"), ("red", "white"),
             ("green", "blue"), ("orange", "red"), ("yellow", "green"), ("blue", "purple")]
    for c1, c2 in mixes:
        add("F_mix", f"Mixing {c1} and {c2} paint gives:")

    for letter in ["R", "O", "Y", "L", "G", "T", "C", "A", "B", "V", "P"]:
        add("G_letter", f"A color that starts with the letter {letter}:")

    for t in ["A warm color:", "A cool color:", "The warmest color:", "The coolest color:",
              "A pastel color:", "A neon color:"]:
        add("H_temp", t)

    return P


if __name__ == "__main__":
    prompts = build()
    out = {"n": len(HUES), "hues": HUES, "prompts": prompts}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "prompts.json"), "w") as f:
        json.dump(out, f, indent=1)
    from collections import Counter
    print(f"{len(prompts)} prompts:", dict(Counter(p['family'] for p in prompts)))
