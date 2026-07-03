"""Month prompt-set builder (concept generalisation of the weekday set).

Same design as weekday_prompts: answer-first completion stems ("...:") so the FIRST
generated token is a month, diverse families to scatter points across the Δ^12
simplex over {January..December}. Pure-Python. Writes prompts.json.
"""

from __future__ import annotations

import argparse
import json
import os

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
AB = {m: m[:3] for m in MONTHS}
ALL = [AB[m] for m in MONTHS]
SPRING = [AB[m] for m in ("March", "April", "May")]
SUMMER = [AB[m] for m in ("June", "July", "August")]
AUTUMN = [AB[m] for m in ("September", "October", "November")]
WINTER = [AB[m] for m in ("December", "January", "February")]
D31 = [AB[m] for m in ("January", "March", "May", "July", "August", "October", "December")]
D30 = [AB[m] for m in ("April", "June", "September", "November")]


def _after(i, k):
    return ALL[(i + k) % 12]


def _records():
    recs = []

    def add(family, core, intended, pid=0):
        recs.append({"family": family, "text_core": core, "intended_days": intended, "paraphrase_id": pid})

    # A: open
    A = ["A month of the year:", "An example of a month of the year is", "Here is a month of the year:",
         "One example of a month of the year:", "A month of the year, for instance,", "A random month of the year:",
         "A month of the year is", "One month of the year is", "For example, a month of the year is",
         "A month (any one):", "Name a month. Answer:", "A month from the calendar:"]
    for pid, c in enumerate(A):
        add("A_open", c, ALL, pid)

    # B: single negation
    B = ["A month of the year that is not {X}:", "A month of the year other than {X} is",
         "A month of the year, but not {X}:", "Any month except {X}:", "A month that isn't {X}:"]
    for pid, t in enumerate(B):
        for m in MONTHS:
            add("B_neg1", t.format(X=m), [a for a in ALL if a != AB[m]], pid)

    # C: double negation (subsample to keep count reasonable)
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    for idx in range(0, len(pairs), 3):  # every 3rd pair -> ~22
        i, j = pairs[idx]
        add("C_neg2", f"A month of the year that is neither {MONTHS[i]} nor {MONTHS[j]}:",
            [a for a in ALL if a not in (AB[MONTHS[i]], AB[MONTHS[j]])])

    # D: first-letter (+ converse)
    letters = sorted({m[0] for m in MONTHS})
    by_letter = {L: [AB[m] for m in MONTHS if m[0] == L] for L in letters}
    for pid, t in enumerate(["A month of the year that starts with the letter {L}:",
                             "A month of the year beginning with {L} is"]):
        for L in letters:
            add("D_letter", t.format(L=L), by_letter[L], pid)
    for L in letters:
        add("D_letter_neg", f"A month of the year that does not start with the letter {L}:",
            [a for a in ALL if a not in by_letter[L]])

    # E: successor / arithmetic (cyclic mod 12)
    for pid, (t, k) in enumerate([("The month after {X}:", 1), ("The month before {X}:", -1),
                                  ("Two months after {X}:", 2), ("Three months after {X}:", 3),
                                  ("Six months after {X}:", 6)]):
        for i, m in enumerate(MONTHS):
            add("E_succ", t.format(X=m), [_after(i, k)], pid)

    # F: semantic (seasons, calendar)
    F = [("A summer month:", SUMMER), ("A winter month:", WINTER), ("A spring month:", SPRING),
         ("An autumn month:", AUTUMN), ("A fall month:", AUTUMN),
         ("The first month of the year:", ["Jan"]), ("The last month of the year:", ["Dec"]),
         ("The middle month of the year:", ["Jun", "Jul"]),
         ("A month with 31 days:", D31), ("A month with 30 days:", D30),
         ("The shortest month:", ["Feb"]), ("The month of Christmas:", ["Dec"]),
         ("The month of Halloween:", ["Oct"]), ("The hottest month of the year:", ["Jul", "Aug"]),
         ("The coldest month of the year:", ["Jan", "Feb"]), ("The month school usually starts:", ["Sep"]),
         ("The month the year begins:", ["Jan"]), ("The month the year ends:", ["Dec"]),
         ("A month in the first half of the year:", ALL[:6]), ("A month in the second half of the year:", ALL[6:]),
         ("The month after summer ends:", ["Sep"]), ("A holiday month:", ["Dec", "Jul"])]
    for c, i in F:
        add("F_semantic", c, i)

    # G: ordinal
    ords = ["first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"]
    for n, o in enumerate(ords):
        add("G_ordinal", f"The {o} month of the year:", [ALL[n]])
    for n, o in enumerate(ords):
        add("G_ordinal", f"Counting from January, the {o} month:", [ALL[n]], pid=1)

    # H: this/next month + graded
    for i, m in enumerate(MONTHS):
        add("H_today", f"If this month is {m}, next month:", [_after(i, 1)])
    for i, m in enumerate(MONTHS):
        add("H_today", f"If this month is {m}, last month:", [_after(i, -1)], pid=1)
    for c, i in [("A month near the end of the year:", ALL[8:]), ("A month near the start of the year:", ALL[:4]),
                 ("A month in the middle of the year:", ["May", "Jun", "Jul", "Aug"]),
                 ("A month close to summer:", ["May", "Jun", "Jul", "Aug", "Sep"])]:
        add("H_graded", c, i)

    # I: multi-constraint
    I = [("A summer month that is not July:", ["Jun", "Aug"]), ("A winter month that is not January:", ["Dec", "Feb"]),
         ("A summer month that starts with J:", ["Jun", "Jul"]), ("A winter month that starts with J:", ["Jan"]),
         ("A month with 31 days in summer:", ["Jul", "Aug"]), ("A month after June but before October:", ["Jul", "Aug", "Sep"]),
         ("A spring month that is not May:", ["Mar", "Apr"]), ("A month that comes before March:", ["Jan", "Feb"]),
         ("A month that comes after October:", ["Nov", "Dec"]), ("A month with 30 days in autumn:", ["Sep", "Nov"]),
         ("A month that starts with M:", ["Mar", "May"]), ("A month that starts with A:", ["Apr", "Aug"]),
         ("A summer or winter month:", SUMMER + WINTER), ("A month that is not in summer:", [a for a in ALL if a not in SUMMER]),
         ("The month before the last month:", ["Nov"]), ("The month after the first month:", ["Feb"]),
         ("A month between March and July:", ["Apr", "May", "Jun"]), ("A holiday month that is not December:", ["Jul"])]
    for c, i in I:
        add("I_multi", c, i)

    return recs


def build():
    seen, out = set(), []
    for r in _records():
        key = r["text_core"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        r = dict(r); r["id"] = f"{r['family']}_{len(out):03d}"
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "prompts.json"))
    args = ap.parse_args()
    p = build()
    with open(args.out, "w") as f:
        json.dump({"n": len(p), "prompts": p}, f, indent=2)
    from collections import Counter
    fam = Counter(x["family"] for x in p)
    print(f"Built {len(p)} month prompts -> {args.out}")
    for k in sorted(fam):
        print(f"  {k:16s} {fam[k]:3d}")


if __name__ == "__main__":
    main()
