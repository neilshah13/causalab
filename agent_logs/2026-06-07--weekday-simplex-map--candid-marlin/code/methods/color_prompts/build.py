"""Build color prompts from the discovered single-token color set (colors.json).

Answer-first colon stems, families analogous to weekdays/months but using colour
semantics (warm/cool, primary/secondary, shades, object-colours). All families are
filtered to colours that are actually single-token (so every target is readable).
Run AFTER color_tokens.py. Pure-Python.
"""

from __future__ import annotations

import argparse
import json
import os

# Hardcoded semantic groups; intersected with the available single-token set at build time.
GROUPS = {
    "warm": ["Red", "Orange", "Yellow", "Pink", "Coral", "Amber", "Gold", "Crimson", "Scarlet", "Salmon", "Peach"],
    "cool": ["Blue", "Green", "Purple", "Cyan", "Teal", "Navy", "Indigo", "Violet", "Turquoise", "Azure", "Mint"],
    "primary": ["Red", "Blue", "Yellow"],
    "secondary": ["Orange", "Green", "Purple"],
    "neutral": ["Gray", "Grey", "Beige", "Brown", "Black", "White", "Tan", "Charcoal", "Ivory", "Silver"],
}
SHADES = {
    "blue": ["Blue", "Navy", "Cyan", "Teal", "Azure", "Cobalt", "Sapphire", "Indigo", "Turquoise"],
    "red": ["Red", "Crimson", "Scarlet", "Maroon", "Ruby", "Burgundy", "Cherry", "Rose"],
    "green": ["Green", "Lime", "Olive", "Emerald", "Jade", "Mint", "Sage", "Teal"],
    "purple": ["Purple", "Violet", "Lavender", "Indigo", "Plum", "Mauve", "Orchid"],
}
OBJECTS = [
    ("The color of the sky:", ["Blue"]), ("The color of grass:", ["Green"]),
    ("The color of blood:", ["Red"]), ("The color of snow:", ["White"]),
    ("The color of the sun:", ["Yellow"]), ("The color of the night sky:", ["Black"]),
    ("The color of a banana:", ["Yellow"]), ("The color of an orange:", ["Orange"]),
    ("The color of coal:", ["Black"]), ("The color of a clear ocean:", ["Blue"]),
    ("The color of a flamingo:", ["Pink"]), ("The color of chocolate:", ["Brown"]),
    ("The color of gold:", ["Gold"]), ("The color of a ripe tomato:", ["Red"]),
]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--colors", default=os.path.join(here, "colors.json"))
    ap.add_argument("--out", default=os.path.join(here, "prompts.json"))
    args = ap.parse_args()
    # use the SAME curated set the analyses use (CANONICAL ∩ single-token, via concept_core)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from concept_core import get_concept
    avail = set(get_concept("colors")[0])
    print(f"curated single-token colors ({len(avail)}): {sorted(avail)}")

    recs = []

    def add(fam, core, intended, pid=0):
        recs.append({"family": fam, "text_core": core, "intended_days": intended, "paraphrase_id": pid})

    def keep(lst):
        return [c for c in lst if c in avail]

    # A: open
    for pid, c in enumerate(["A color:", "An example of a color is", "Here is a color:",
                             "One example of a color:", "A color, for instance,", "A random color:",
                             "Name a color. Answer:", "A color is", "Some color:", "A common color:"]):
        add("A_open", c, sorted(avail), pid)

    # B: single negation (per available color)
    for pid, t in enumerate(["A color that is not {X}:", "A color other than {X} is",
                             "A color, but not {X}:", "Any color except {X}:"]):
        for X in sorted(avail):
            add("B_neg1", t.format(X=X), [c for c in sorted(avail) if c != X], pid)

    # D: first-letter
    letters = sorted({c[0] for c in avail})
    for L in letters:
        grp = [c for c in sorted(avail) if c[0] == L]
        add("D_letter", f"A color that starts with the letter {L}:", grp)
        add("D_letter_neg", f"A color that does not start with the letter {L}:",
            [c for c in sorted(avail) if c not in grp])

    # F: warm/cool/primary/secondary/neutral
    for g, members in GROUPS.items():
        km = keep(members)
        if len(km) >= 1:
            add("F_group", f"A {g} color:", km)
    # F: shades
    for base, members in SHADES.items():
        km = keep(members)
        if len(km) >= 2:
            add("F_shade", f"A shade of {base}:", km)
    # F: object colors (only if the answer color is available)
    for core, ans in OBJECTS:
        if all(a in avail for a in ans):
            add("F_object", core, ans)

    # I: multi-constraint (intersections), only where non-empty
    multi = [
        ("A warm color that is not red:", keep([c for c in GROUPS["warm"] if c != "Red"])),
        ("A cool color that is not blue:", keep([c for c in GROUPS["cool"] if c != "Blue"])),
        ("A primary color that is not blue:", keep(["Red", "Yellow"])),
        ("A neutral color that is not black:", keep([c for c in GROUPS["neutral"] if c != "Black"])),
        ("A shade of blue that is not navy:", keep([c for c in SHADES["blue"] if c != "Navy"])),
        ("A warm color that starts with the letter O:", keep([c for c in GROUPS["warm"] if c[0] == "O"])),
    ]
    for core, ans in multi:
        if ans:
            add("I_multi", core, ans)

    # dedup
    seen, out = set(), []
    for r in recs:
        k = r["text_core"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        r = dict(r); r["id"] = f"{r['family']}_{len(out):03d}"
        out.append(r)
    with open(args.out, "w") as f:
        json.dump({"n": len(out), "prompts": out}, f, indent=2)
    from collections import Counter
    fam = Counter(r["family"] for r in out)
    print(f"Built {len(out)} color prompts -> {args.out}")
    for k in sorted(fam):
        print(f"  {k:14s} {fam[k]:3d}")


if __name__ == "__main__":
    main()
