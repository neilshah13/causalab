"""Weekday prompt-set builder (session-local, step 1 of weekday-simplex-map).

Builds a diverse set of weekday prompts whose *next-token* distributions are
designed to scatter across the behavioural simplex Δ^7 over {Mon..Sun} (plus an
"other" class). Pure-Python, no torch — runs anywhere. Writes ``prompts.json``.

DESIGN (revised after the format probe): each prompt is a **completion stem**
that ends exactly where a weekday should appear, so the model's FIRST generated
token is a day name (the user's "first output token should be the day"
requirement). This avoids the conversational lead-ins (" I", " The", "...is")
that wrecked weekday-mass under a "Q:\\nA:" frame or an instruct chat template
(probe: base 0/45 pass, instruct 42% with relational families at ~0%).

Two stem shapes are used, whichever reads naturally:
  - colon "give an example" stems:  "A day of the week that is not Monday:"
  - copular " is/was" stems:         "The day after Monday is"

The base model consumes the stem verbatim. ``intended_days`` is a *diagnostic
annotation only* (colours plots, sanity-checks hypothesis H1); the coverage
analysis never assumes it is correct.

Families (~230 prompts after dedup):
  A open "a day"             -> interior / typical-day bias
  B single negation          -> 6-day faces
  C double negation          -> 5-day faces
  D first-letter (+converse) -> edges {Tue,Thu},{Sat,Sun} and vertices
  E successor/arithmetic      -> near-vertex + cyclic neighbour spread (paper H3)
  F semantic constraints      -> edges / specific vertices / bimodal
  G ordinal/positional        -> spread by Mon-start vs Sun-start convention
  H today/tomorrow + graded   -> vertices + soft interior
  I multi-constraint          -> small faces / edges
"""

from __future__ import annotations

import argparse
import json
import os

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ABBR = {d: d[:3] for d in DAYS}  # Monday -> Mon
ALL = [ABBR[d] for d in DAYS]
WEEKDAYS = [ABBR[d] for d in DAYS[:5]]  # Mon..Fri
WEEKEND = [ABBR[d] for d in DAYS[5:]]  # Sat, Sun


def _abbr(day_full: str) -> str:
    return ABBR[day_full]


def _after(i: int, k: int) -> str:
    return ALL[(i + k) % 7]


def _records():
    recs: list[dict] = []

    def add(family: str, core: str, intended: list[str], pid: int = 0):
        recs.append(
            {
                "family": family,
                "text_core": core,
                "intended_days": intended,
                "paraphrase_id": pid,
            }
        )

    # ---- A: open "a day" (interior) --------------------------------------
    A = [
        "A day of the week:",
        "An example of a day of the week is",
        "Here is a day of the week:",
        "One example of a day of the week:",
        "A day of the week, for example,",
        "My favorite example of a day of the week is",
        "A random day of the week:",
        "Pick a day of the week. Answer:",
        "A day of the week is",
        "One day of the week is",
        "For example, a day of the week is",
        "A day of the week (any one):",
    ]
    for pid, core in enumerate(A):
        add("A_open", core, ALL, pid)

    # ---- B: single negation -> 6-day face --------------------------------
    B_tmpl = [
        "A day of the week that is not {X}:",
        "A day of the week other than {X} is",
        "A day of the week, but not {X}:",
        "Any day of the week except {X}:",
        "A day of the week that isn't {X}:",
        "Not {X}, but another day of the week:",
        "A weekday or weekend day other than {X} is",
    ]
    for pid, t in enumerate(B_tmpl):
        for d in DAYS:
            intended = [a for a in ALL if a != _abbr(d)]
            add("B_neg1", t.format(X=d), intended, pid)

    # ---- C: double negation -> 5-day face --------------------------------
    for i in range(7):
        for j in range(i + 1, 7):
            x, y = DAYS[i], DAYS[j]
            intended = [a for a in ALL if a not in (ABBR[x], ABBR[y])]
            add(
                "C_neg2",
                "A day of the week that is neither {X} nor {Y}:".format(X=x, Y=y),
                intended,
            )

    # ---- D: first-letter (+ converse) -> edges / vertices ----------------
    letters = sorted({d[0] for d in DAYS})  # F M S T W
    by_letter = {L: [ABBR[d] for d in DAYS if d[0] == L] for L in letters}
    D_pos = [
        "A day of the week that starts with the letter {L}:",
        "A day of the week beginning with {L} is",
    ]
    for pid, t in enumerate(D_pos):
        for L in letters:
            add("D_letter", t.format(L=L), by_letter[L], pid)
    for L in letters:
        intended = [a for a in ALL if a not in by_letter[L]]
        add(
            "D_letter_neg",
            "A day of the week that does not start with the letter {L}:".format(L=L),
            intended,
        )

    # ---- E: successor / predecessor / arithmetic -> near-vertex + spread --
    # Answer-first colon form so the FIRST token is the day (copular "... is"
    # makes the base model continue with prose; "...:" makes it lead with the day).
    E_rel = [
        ("The day after {X}:", 1),
        ("The day before {X}:", -1),
        ("Two days after {X}:", 2),
        ("Three days after {X}:", 3),
        ("Two days before {X}:", -2),
        ("The day right after {X}:", 1),
    ]
    for pid, (t, k) in enumerate(E_rel):
        for i, d in enumerate(DAYS):
            add("E_succ", t.format(X=d), [_after(i, k)], pid)

    # ---- F: semantic constraints -> edges / vertices / bimodal -----------
    # Answer-first colon form (day leads the completion).
    F = [
        ("A day of the weekend:", WEEKEND),
        ("A weekend day:", WEEKEND),
        ("A weekday:", WEEKDAYS),
        ("A work day:", WEEKDAYS),
        ("A school day:", WEEKDAYS),
        ("The first day of the week:", ["Mon", "Sun"]),
        ("The last day of the week:", ["Sun", "Sat"]),
        ("The day in the middle of the week:", ["Wed"]),
        ("The day right before the weekend:", ["Fri"]),
        ("The first day of the weekend:", ["Sat"]),
        ("The last day of the weekend:", ["Sun"]),
        ("The first day after the weekend:", ["Mon"]),
        ("The day most people start work:", ["Mon"]),
        ("The day most people relax:", WEEKEND),
        ("The busiest day of the work week:", WEEKDAYS),
        ("The middle day of the work week:", ["Wed"]),
        ("The day most offices are closed:", WEEKEND),
        ("The day children go to school:", WEEKDAYS),
        ("The day before Monday:", ["Sun"]),
        ("The day after Friday:", ["Sat"]),
        ("The day people typically do not work:", WEEKEND),
        ("The day banks are usually open:", WEEKDAYS),
        ("The day often called hump day:", ["Wed"]),
        ("The day that ends the work week:", ["Fri"]),
        ("The day that starts the work week:", ["Mon"]),
        ("The day church services are common:", ["Sun"]),
    ]
    for core, intended in F:
        add("F_semantic", core, intended)

    # ---- G: ordinal / positional -> convention-dependent spread ----------
    # Answer-first colon form (day leads the completion).
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
    for n, ordw in enumerate(ordinals):  # n = 0..6
        mon_start = ALL[n]
        sun_start = ALL[(n - 1) % 7]  # Sun-start: 1st=Sun
        intended = sorted({mon_start, sun_start}, key=ALL.index)
        add("G_ordinal", "The {O} day of the week:".format(O=ordw), intended)
    for n, ordw in enumerate(ordinals):
        add(
            "G_ordinal",
            "Counting from Monday, the {O} day of the week:".format(O=ordw),
            [ALL[n]],
            pid=1,
        )
    for n, ordw in enumerate(ordinals):
        # Sunday-start convention: 1st=Sun, 2nd=Mon, ...
        add(
            "G_ordinal",
            "If the week starts on Sunday, the {O} day:".format(O=ordw),
            [ALL[(n - 1) % 7]],
            pid=2,
        )

    # ---- H: today/tomorrow + graded interior -----------------------------
    # Answer-first colon form (day leads the completion).
    for i, d in enumerate(DAYS):
        add("H_today", "Today is {X}; tomorrow:".format(X=d), [_after(i, 1)])
    for i, d in enumerate(DAYS):
        add("H_today", "Today is {X}; yesterday:".format(X=d), [_after(i, -1)], pid=1)
    H_graded = [
        ("A day near the end of the week:", ["Thu", "Fri", "Sat", "Sun"]),
        ("A day near the start of the week:", ["Mon", "Tue", "Wed"]),
        ("A day in the middle of the week:", ["Tue", "Wed", "Thu"]),
        ("A day close to the weekend:", ["Fri", "Sat", "Sun"]),
    ]
    for core, intended in H_graded:
        add("H_graded", core, intended)

    # ---- I: multi-constraint -> small faces / edges ----------------------
    I = [
        ("A weekend day that is not Sunday:", ["Sat"]),
        ("A weekend day that is not Saturday:", ["Sun"]),
        ("A weekday that starts with the letter T:", ["Tue", "Thu"]),
        ("A work day that is not Monday:", ["Tue", "Wed", "Thu", "Fri"]),
        ("A day between Monday and Friday:", ["Tue", "Wed", "Thu"]),
        ("A day after Wednesday but before Sunday:", ["Thu", "Fri", "Sat"]),
        ("A weekday that is not Monday or Friday:", ["Tue", "Wed", "Thu"]),
        ("A day that comes before Wednesday:", ["Mon", "Tue"]),
        ("A day that comes after Thursday:", ["Fri", "Sat", "Sun"]),
        ("A day of the week that contains the letter r:", ["Thu", "Fri", "Sat"]),
        ("A weekday that starts with the letter W:", ["Wed"]),
        ("A weekday that starts with the letter F:", ["Fri"]),
        ("A weekend day that starts with the letter S:", ["Sat", "Sun"]),
        ("The day after Saturday is", ["Sun"]),
        ("The day before Tuesday is", ["Mon"]),
        ("A work day in the second half of the week:", ["Thu", "Fri"]),
        ("A day that is not a weekday:", WEEKEND),
        ("A day that is not a weekend day:", WEEKDAYS),
        ("A weekday other than Wednesday:", ["Mon", "Tue", "Thu", "Fri"]),
        ("A day in the first half of the week:", ["Mon", "Tue", "Wed"]),
        ("A day between Wednesday and Saturday:", ["Thu", "Fri"]),
        ("A weekday that does not start with the letter T:", ["Mon", "Wed", "Fri"]),
        ("The day immediately after a weekend day:", ["Mon", "Sun"]),
        ("One of the two days that start with the letter S:", ["Sat", "Sun"]),
    ]
    for core, intended in I:
        add("I_multi", core, intended)

    return recs


def build():
    recs = _records()
    # Dedup by text_core (keep first occurrence), assign stable ids.
    seen = set()
    out = []
    for r in recs:
        key = r["text_core"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        r = dict(r)
        r["id"] = f"{r['family']}_{len(out):03d}"
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "prompts.json"),
        help="Where to write prompts.json",
    )
    args = ap.parse_args()
    prompts = build()
    with open(args.out, "w") as f:
        json.dump({"n": len(prompts), "prompts": prompts}, f, indent=2)

    # Summary
    from collections import Counter

    fam = Counter(p["family"] for p in prompts)
    print(f"Built {len(prompts)} prompts -> {args.out}")
    for k in sorted(fam):
        print(f"  {k:16s} {fam[k]:3d}")


if __name__ == "__main__":
    main()
