# Conclusions — mapping a concept's region instead of fitting a curve through its centroids

*Session 2026-06-07 weekday-simplex-map · final, 2026-06-12. The complete experiment log lives in
REPORT.md; this document makes the claims. Figures C1–C4 (+ animated G1/G2) in `result/figures/`.*

---

## Lexicon

| Term | Meaning here |
|---|---|
| **activation space** | The model's internal state: the vector of numbers at one layer, for the last token of a prompt. |
| **behaviour space** | What the model is about to say: its probability distribution over the answer words (Monday…Sunday, plus "anything else"). |
| **centroid** | The average activation over all prompts that share the same answer (e.g. "the Monday point"). The Goodfire paper uses 7 of these. |
| **manifold** | The paper's object: a smooth curve drawn through the 7 centroids by a fitting algorithm. |
| **anchor prompt** | One of our prompts. Each produces a known (activation, behaviour) pair. We use 194, designed so their *behaviours* cover the whole range — confident answers, hesitations between two days, near-uniform uncertainty. |
| **the map** | The correspondence we measure between activation space and behaviour space, built from the anchors and from walking around them. |
| **walking** | Taking a new point in activation space near the anchors, forcing the model's state to that point, and reading which behaviour comes out. |
| **steering** | Walking on purpose: choosing activation points so the behaviour follows a chosen path (e.g. Monday → Tuesday → …). |
| **naturalness (E_BC)** | The paper's own score for steering quality: how far the steered behaviours stray from behaviours the model produces on its own. Lower is better. |
| **natural ceiling** | The highest probability the model *ever* gives a word across all our prompts. A word with ceiling 0.01 is one the model essentially never says. |

---

## The problem with the manifold

The paper describes a manifold over an entire region of activation space — but what is actually
computed is **7 centroid points and a curve-fitting step**. Nothing in the procedure looks at what
the space *between* the points is like; the curve's shape there is an assumption of the fitting
algorithm, not a measurement. We believe this under-explores the conceptual geometry that is
actually embedded in the model.

## Evidence: the manifold is not doing the work — its points are

We compared the fitted manifold against plain **vectors drawn from centroid to centroid**
(data: the 2026-06-03 manifold-vs-vector session, plus this session's head-to-head). Three results,
all pointing the same way *(figure C1)*:

1. Steering through intermediate days: centroid-to-centroid vectors match the manifold exactly
   (both put ~0.65 probability on the intended day at every waypoint; a single straight vector
   collapses to ~0.1).
2. Placing a day the method was never shown: a simple vector construction recovers the missing
   day **better** than every manifold variant (0.76 vs 0.48).
3. On the paper's own naturalness score, by the last layers of the model the fitted manifold is
   indistinguishable from a straight vector between centroids.

**Conclusion: the manifold's value is entirely in where its centroid points sit — the curve
connecting them is not "real".** There is genuine geometry in how the concepts are *arranged*,
but the paper's method never measures the geometry of the *connections* between them.

## Our approach: map the region, not the curve

Instead of one point per day plus a fitted curve, we map the **whole "day of the week"
representation** *(figure C2; animated in G1)*:

1. Write anchor prompts whose behaviours spread over the entire region — not just "the answer is
   Monday" but "not Tuesday", "a weekend day", "any day" — so the interior of behaviour space
   (hesitation, uncertainty) is represented, not only the seven confident corners.
2. Find the small set of activation directions that carry this behaviour (about 6 directions for
   7 days; the rest of the 4096 don't matter for it).
3. **Walk** around the anchors in those directions and record which behaviour every step produces.
   The geometry of the map is then *discovered*, not assumed.

What the walk finds: close to the anchors, activation and behaviour correspond smoothly and almost
linearly — hesitant, in-between behaviours live there. Further out, the model snaps to confident
single-day answers no matter the details of the position. The seven-day cycle appears in the
arrangement of the anchors themselves (and the twelve-month cycle likewise, perfectly ordered) —
consistent with the paper — but no fitted curve is needed to see or to use it.

## Result 1 — steering along the discovered map is more natural than the paper's steering

Measured with the paper's **own** naturalness score, on identical start/end points and identical
prompts *(figure C3)*: steering with our map stays 2–5× closer to the model's natural behaviour
than either of the paper's methods (straight vectors **and** the fitted manifold), at every
mid-to-late layer, with very high statistical confidence (p < 10⁻¹²). And the output stays
coherent: after every steering intervention the model continues writing normal text (we generated
continuations and scored their fluency — no method breaks the model, but only ours also keeps the
behaviour on the natural path).

Practical detail that matters: steering works best at layers 23–27, not at the very last layer.

## Result 2 — a few prompts are enough to recover the whole concept

Because the map is a *region* and not a list of memorized points, it extrapolates *(figure C4;
animated in G2)*:

- Keep only the **Monday and Thursday** anchors, throw everything else away, rebuild the map from
  those alone — then walk it. All five other days are reached with probability 0.8–1.0, **without
  a single prompt about them**.
- The same works per-region: delete every anchor for any one day (or three consecutive days —
  over half of all anchors) and the map rebuilt from the rest still reaches the deleted region.
- It even works past the model's habits: from a map built only on red/green/blue colour prompts,
  steering reaches colours like *scarlet* with probability 0.86 — a word whose natural ceiling is
  0.01, i.e. an output the model never produces on its own *(figure F4)*.

## Why we believe this matters

These two results together mean: **behaviour can be discovered without prompting for it.** If the
geometry of the map is measured well enough, a handful of prompts pins down an entire concept
region, and steering toward any part of that region — including parts the model was never asked
about, and outputs it never produces spontaneously — is strong and stays natural.

## Fine print

Three honest limits. (1) When anchors are scarce (our 12-colour wheel had only ~4 anchors per
colour), the simple map is poorly estimated and the paper's centroid-plus-curve approach is
actually the safer tool — the same is true at the early layer where the concept first forms. Our
advantage needs enough anchors per concept value. (2) Rare words have low natural ceilings; "failed
to recover beige" usually means the model never says beige, not that the map is wrong — recovery
must be judged against ceilings. (3) Everything here is one model (Llama-3.1-8B), one-word answers,
and concept sets with a cyclic structure; months and a colour wheel reproduce the story, but
sequential concepts and other models remain to be tested.

*Full experimental record, every number, and all robustness checks: REPORT.md (Steps 1–12).
Figure index: C1 manifold-vs-vectors · C2 how we map (G1 animated) · C3 naturalness + coherence ·
C4 few-anchor recovery (G2 animated) · F4 steering past the natural ceiling.*
