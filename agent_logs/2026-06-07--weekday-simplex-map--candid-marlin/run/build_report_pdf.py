#!/usr/bin/env python3
"""
Build the consolidated results PDF for the weekday-simplex-map session.
Curated narrative figures (explained in depth, by step) + representative sweep frames.
Pure-local; uses reportlab + DejaVuSans (from matplotlib) for full Unicode (Δ⁷, ≈, →, R²).
"""
import os, glob, re, matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, KeepTogether, PageBreak, CondPageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT = "/Users/johan/Desktop/Projects/TARA/causalab"
SESSION = os.path.join(ROOT, "agent_logs/2026-06-07--weekday-simplex-map--candid-marlin")
RES = os.path.join(SESSION, "result/figures")
WK  = os.path.join(SESSION, "artifacts/weekday_simplex/llama31_8b/simplex_coverage/figures")
WKI = os.path.join(SESSION, "artifacts/weekday_simplex/llama31_8b_instruct/simplex_coverage/figures")
MO  = os.path.join(SESSION, "artifacts/months_simplex/llama31_8b/simplex_coverage/figures")
CO  = os.path.join(SESSION, "artifacts/colors_simplex/llama31_8b/simplex_coverage/figures")
OUT = os.path.join(SESSION, "result/weekday-simplex-map-RESULTS.pdf")

# ---- fonts -----------------------------------------------------------------
TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
def reg(name, fname):
    p = os.path.join(TTF, fname)
    if os.path.exists(p):
        pdfmetrics.registerFont(TTFont(name, p)); return True
    return False
reg("DJ", "DejaVuSans.ttf")
reg("DJ-B", "DejaVuSans-Bold.ttf")
has_obl = reg("DJ-I", "DejaVuSans-Oblique.ttf")
if not has_obl:  # fall back to BoldOblique-less: map italic to normal
    reg("DJ-I", "DejaVuSans.ttf")
reg("DJ-BI", "DejaVuSans-BoldOblique.ttf")
reg("DJM", "DejaVuSansMono.ttf")
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I", boldItalic="DJ-BI")

# ---- palette ---------------------------------------------------------------
INK   = colors.HexColor("#1a1a2e")
ACC   = colors.HexColor("#2d5d7b")   # steel blue
ACC2  = colors.HexColor("#7b2d4f")   # plum
LIGHT = colors.HexColor("#eef2f5")
RULE  = colors.HexColor("#c4d0d8")
MUT   = colors.HexColor("#5a6672")
GOOD  = colors.HexColor("#1f6f43")

# ---- styles ----------------------------------------------------------------
def S(name, **kw):
    base = dict(fontName="DJ", textColor=INK, fontSize=10, leading=14)
    base.update(kw); return ParagraphStyle(name, **base)

st_title   = S("title", fontName="DJ-B", fontSize=23, leading=27, textColor=INK)
st_sub     = S("sub", fontSize=12.5, leading=17, textColor=ACC)
st_meta    = S("meta", fontSize=9.5, leading=14, textColor=MUT)
st_h1       = S("h1", fontName="DJ-B", fontSize=16, leading=20, textColor=colors.white,
                spaceBefore=2, spaceAfter=2)
st_h2       = S("h2", fontName="DJ-B", fontSize=12.5, leading=16, textColor=ACC2,
                spaceBefore=12, spaceAfter=4)
st_body    = S("body", fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
st_body_t  = S("body_t", fontSize=10, leading=14.5, alignment=TA_JUSTIFY)  # tight, no spaceAfter
st_lead    = S("lead", fontSize=11, leading=15.5, textColor=INK, alignment=TA_LEFT, spaceAfter=6)
st_cap     = S("cap", fontSize=8.6, leading=11.6, textColor=MUT, alignment=TA_LEFT)
st_cell    = S("cell", fontSize=8.4, leading=10.6)
st_cellc   = S("cellc", fontSize=8.4, leading=10.6, alignment=TA_CENTER)
st_hd      = S("hd", fontName="DJ-B", fontSize=8.4, leading=10.6, textColor=colors.white, alignment=TA_CENTER)
st_toc     = S("toc", fontSize=10.5, leading=18)
st_code    = S("code", fontName="DJM", fontSize=8.2, leading=11.4, textColor=colors.HexColor("#22303a"))
st_note    = S("note", fontSize=9.2, leading=13, textColor=colors.HexColor("#33404a"))
st_kicker  = S("kicker", fontName="DJ-B", fontSize=8.5, leading=11, textColor=GOOD)

# ---- markup helper ---------------------------------------------------------
# Robust to a mix of conventions in the authored content:
#   literal <b>/<i>/<font>/<super> tags, **bold**, ~~mono~~, and pre-typed
#   &gt;/&lt;/&amp; entities. Everything else (stray < > &) is escaped safely.
_KEEP = ["b", "i", "super", "sub"]
def conv(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)                  # **bold**
    s = re.sub(r"~~(.+?)~~", r'<font name="DJM">\1</font>', s)     # ~~mono~~
    s = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;)", "&amp;", s) # escape raw & only
    s = s.replace("<", "&lt;").replace(">", "&gt;")               # escape all angle brackets
    for t in _KEEP:                                               # then restore whitelisted tags
        s = s.replace(f"&lt;{t}&gt;", f"<{t}>").replace(f"&lt;/{t}&gt;", f"</{t}>")
    s = re.sub(r"&lt;font ([^&]*?)&gt;", r"<font \1>", s)
    s = s.replace("&lt;/font&gt;", "</font>").replace("&lt;br/&gt;", "<br/>")
    return s
def P(text, style=st_body): return Paragraph(conv(text), style)

story = []
FIG = [0]
def figure(path, caption, max_h=18.2*cm, max_w=None, kicker=None):
    if not os.path.exists(path):
        story.append(P(f"[missing figure: {os.path.basename(path)}]", st_cap)); return
    iw, ih = PILImage.open(path).size
    CW = 17.4*cm if max_w is None else max_w
    s = min(CW/iw, max_h/ih)
    img = Image(path, width=iw*s, height=ih*s)
    FIG[0] += 1
    bits = [img, Spacer(1, 3)]
    if kicker:
        bits.append(P(kicker, st_kicker))
    bits.append(P(f"**Figure {FIG[0]}.** {caption}", st_cap))
    story.append(KeepTogether(bits))
    story.append(Spacer(1, 11))

def band(title):
    t = Table([[P(title, st_h1)]], colWidths=[17.4*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),ACC),
                           ("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),
                           ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(Spacer(1,4)); story.append(t); story.append(Spacer(1,8))

def h2(title): story.append(P(title, st_h2))

def para(text, style=st_body): story.append(P(text, style))

def table(rows, widths, align_center_from=1):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, val in enumerate(row):
            if r == 0:
                cells.append(P(str(val), st_hd))
            else:
                cells.append(P(str(val), st_cellc if c >= align_center_from else st_cell))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),ACC2),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT]),
        ("GRID",(0,0),(-1,-1),0.4,RULE),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(t); story.append(Spacer(1, 8))

def callout(text, label="KEY RESULT", hexcol="#2d5d7b"):
    accent = colors.HexColor(hexcol)
    inner = P(f'<font color="{hexcol}"><b>{label}</b></font>  ' + text, st_note)
    t = Table([[inner]], colWidths=[17.4*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f4f7f9")),
                           ("BOX",(0,0),(-1,-1),0.6,accent),
                           ("LINEBEFORE",(0,0),(0,-1),3,accent),
                           ("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),
                           ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(t); story.append(Spacer(1, 9))

# ===========================================================================
# TITLE PAGE
# ===========================================================================
story.append(Spacer(1, 2.0*cm))
story.append(P("Mapping Behaviour Space to Activation Space", st_title))
story.append(P("for cyclic concepts in Llama-3.1-8B", st_title))
story.append(Spacer(1, 6))
story.append(P("Consolidated results — weekday simplex map, and its generalisation to "
               "months (Δ¹²) and colours (Δ⁴⁵)", st_sub))
story.append(Spacer(1, 18))
hr = Table([[""]], colWidths=[17.4*cm]); hr.setStyle(TableStyle([("LINEABOVE",(0,0),(-1,0),1.2,ACC)]))
story.append(hr); story.append(Spacer(1, 10))
story.append(P("<b>Session</b>  2026-06-07--weekday-simplex-map--candid-marlin", st_meta))
story.append(P("<b>Model</b>  Llama-3.1-8B (base) · neutral few-shot, answer-first colon frame · "
               "last-token residual stream", st_meta))
story.append(P("<b>Compute</b>  cinaps SLURM cluster (1× A6000) · jobs 12418–12508", st_meta))
story.append(P("<b>Scope</b>  8 completed steps + Day-3 steering demo · weekdays Δ⁷ → months Δ¹² → colours Δ⁴⁵", st_meta))
story.append(Spacer(1, 16))
story.append(P("Abstract", st_h2))
story.append(P(
  "This report consolidates an eight-step investigation of the <b>map between behaviour space</b> "
  "(the probability simplex over a concept's tokens — e.g. Δ⁷ over {Mon…Sun}) and "
  "<b>activation space</b> (the residual stream of Llama-3.1-8B). The work deliberately studies the "
  "<b>linear subspace map</b> rather than fitting a low-dimensional curved manifold. Starting from a "
  "hand-designed prompt set that densely covers the behavioural simplex, we show the behaviour "
  "subspace is <b>linearly embedded in the late residual stream</b> (sharp onset ≈ L19, cleanest at "
  "L31), that the valid region is a <b>continuous, ~|Z|−1-dimensional, traversable</b> subspace that "
  "<b>maps onto</b> the reachable simplex, that the map <b>recovers held-out regions</b> and can be "
  "<b>inverted</b> to hit target distributions, and that it supports <b>continuous steering</b> across a "
  "concept's gamut from only a sparse set of anchored points. Every claim is reproduced on months "
  "(Δ¹²) and extended toward colours (Δ⁴⁵).", st_body))
story.append(Spacer(1, 10))
story.append(P("Auto-generated from <b>result/REPORT.md</b> and the session figure set. Figures are the "
               "curated narrative set; an appendix shows one representative frame per parameter sweep.",
               st_cap))
story.append(PageBreak())

# ===========================================================================
# CONTENTS
# ===========================================================================
story.append(P("Contents", st_h2))
toc = [
  ("Overview & method", "the bet: subspaces, not manifolds"),
  ("Step 1 — Prompt set & simplex coverage", "194/225 retained; frame > content"),
  ("Step 2 — Behaviour vs activation subspace (linear)", "linearly embedded by L31, R²=0.86"),
  ("Step 3 — Causal perturbation map", "valid region continuous & traversable"),
  ("Step 4 — Shape of the valid region: is the map onto?", "7.8× coverage gain — yes"),
  ("Step 5 — Concept-general routine + months generalisation", "depth & k laws concept-independent"),
  ("Step 6 — Leave-a-region-out recovery", "87–100% region recovery"),
  ("Step 7 — Inverse map (sharp onto-test)", "vertices exact, interior residual ~0.20"),
  ("Step 8 — Sparse-anchor steering completeness", "anchor 2 → recover all at k≥8"),
  ("Day 3 — Continuous steering demo + colours", "walk the subspace along a target path"),
  ("Methodological notes & issues", "the frame investigation; pitfalls"),
  ("Appendix — representative sweep frames", "layer / k / margin / seed / concept"),
]
for i,(a,b) in enumerate(toc):
    story.append(P(f"<b>{a}</b>  —  <font color='#5a6672'>{b}</font>", st_toc))
story.append(PageBreak())

# ===========================================================================
# OVERVIEW
# ===========================================================================
band("Overview & method")
para("The repository originates from <b>“Manifold Steering Reveals the Shared Geometry of Neural "
     "Network Representation and Behaviour”</b> (Wurgaft, Rager, Kowal et al.), which fits an "
     "activation manifold M_h and a behaviour manifold M_y and shows that steering along M_h "
     "produces behaviours that follow M_y while linear steering does not. The paper drives the "
     "weekday simplex using only arithmetic prompts (“what day is k days after z?”), whose "
     "distributions sit near simplex vertices. To study the behaviour↔activation <b>map itself</b> we "
     "need belief states scattered across the interior, edges and faces of the simplex — which is "
     "where this session starts.")
para("<b>The methodological bet.</b> Rather than assume a low-dimensional curved manifold captures the "
     "concept and fitting one, this session studies the <b>linear subspace map</b> between the "
     "behavioural simplex and the residual stream directly, per layer, with causal interventions. "
     "Step 2 vindicates the choice: the behaviour-aligned directions are a ~6-D slice of a larger "
     "(~17–18-D) activation subspace, so a 6-D manifold would not capture the activation variation.")
callout("Across 8 steps the session establishes a behaviour↔activation <b>subspace map</b> that is "
        "continuous, ~|Z|−1-dimensional, behaviour-aligned, lives in the late residual stream "
        "(onset ≈ L19, cleanest at L31), maps <b>onto</b> the reachable simplex, recovers held-out "
        "regions, inverts to target distributions, and supports continuous steering from sparse "
        "anchors — reproduced on Δ⁷ (weekdays), Δ¹² (months) and extended to Δ⁴⁵ (colours).",
        label="HEADLINE", hexcol="#7b2d4f")
h2("Pipeline at a glance")
table([
  ["Step","Question","Headline result"],
  ["1","Can a prompt set densely cover the behavioural simplex?","194/225 retained, 6 PCA dims, all days reachable; frame matters more than content"],
  ["2","Is behaviour linearly embedded in activations?","Yes, in the late stream — L31: 6/6 canon. corr > 0.9, decode R² = 0.86; onset ≈ L19"],
  ["3","Is the valid region traversable (causal)?","Continuous & connected; in-subspace moves behaviour 5–6× more than random R⁴⁰⁹⁶"],
  ["4","Does the map fill the simplex (onto)?","Largely yes — 7.8× cell-coverage gain; fills interior between anchors"],
  ["5","Does it generalise + a reusable routine?","Months reproduces everything; behaviour dim ≈ |Z|−1; depth/k laws concept-independent"],
  ["6","Recover a deleted region from the rest?","87–100% region recovery, P(target) up to 1.0 — method validated"],
  ["7","Invert to a target distribution?","Vertices exact (Hellinger 0.03), edges ~0.15, interior mixtures ~0.20 residual"],
  ["8","Steer the whole set from sparse anchors?","Anchor {Mon,Thu} → recover 5/5 unprompted days at k≥8; CCA caps at |Z|+1"],
], [1.0*cm, 6.0*cm, 10.4*cm], align_center_from=99)
story.append(PageBreak())

# ===========================================================================
# STEP 1
# ===========================================================================
band("Step 1 — Prompt set & simplex coverage")
para("<b>Goal.</b> Build a prompt set whose next-token weekday distributions densely and diversely "
     "cover the behavioural simplex — the region of Δ⁷ over {Mon…Sun} where weekday mass &gt; 90% and "
     "“other” mass &lt; 10% — the foundation for mapping behaviour (M_y) to activations (M_h).")
para("<b>What was built.</b> A 225-prompt set across 9 families, each an <b>answer-first completion "
     "stem</b> (“…:”) so the first token is a day. Families target distinct simplex regions: open "
     "(interior), negation (6- and 5-day faces), first-letter (edges {Tue,Thu}/{Sat,Sun}), successor "
     "(vertex + cyclic-neighbour spread), semantic/ordinal (convention-dependent), and "
     "multi-constraint (small faces).")
callout("A 225-prompt set; <b>194/225 (86%) pass the mass filter</b> on base Llama-3.1-8B. Retained "
        "points span <b>6 Hellinger-PCA dimensions</b> (participation ratio 5.85 ≈ the full simplex "
        "dimension), entropy 0.01→1.93 nats (near-vertex → near-uniform), <b>all 7 days reachable & "
        "balanced</b>, and <b>39% genuinely interior</b>. All four success criteria satisfied.")
h2("The frame investigation — the substantive result")
para("The challenge was not <i>which</i> prompts but <i>how to frame</i> them so the first token is a "
     "day without distorting the model’s genuine belief. Four regimes:")
table([
  ["Regime","pass (other&lt;10%)","interior frac","what happens"],
  ["base, Q:\\nA: / bare stem","0%","—","argmax right 98% of the time, but 30–50% mass bleeds to prose lead-ins (‘ I’, ‘ The’, ‘ a’)"],
  ["instruct + “answer in one word”","98%","0.26","high mass but collapses the simplex to its vertices (RLHF decisiveness) — interior destroyed"],
  ["base + neutral few-shot, colon stems  ✓","86%","0.39","strips prose scaffolding without prescribing a day; preserves graded belief"],
  ["instruct + “reply with a day”","100%","0.26","naming the answer space buys nothing — rejected as leaky"],
], [5.6*cm, 2.3*cm, 1.9*cm, 7.6*cm], align_center_from=1)
para("<b>Two decisions.</b> (1) <b>Don’t constrain content, constrain form</b>: a neutral few-shot "
     "demonstrating terse answer-first completions on <i>non-weekday</i> tasks (colour, fruit, letter) "
     "induces the format with zero answer-space leakage, and matches the leaky upper bound on mass. "
     "(2) <b>Use the base model</b>: “answer in one word” on the instruct model gives 98% mass but "
     "collapses every prompt to a vertex (interior 0.26); base keeps the spread (0.39) the "
     "behaviour-manifold work needs, and matches the model the activation work uses.")
para("Frame engineering also fixed stem shape: copular “…is/…on” stems made the base model continue "
     "with prose; switching every family to answer-first colon “…:” raised the relational/ordinal "
     "families from ~0% to ~100% retained (E 1→42, G 0→20, H 0→14) with no leakage.")

figure(os.path.join(RES,"pca_scatter_by_argmax_day.png"),
  "Behavioural simplex (base, 194 retained), Hellinger-PCA, coloured by arg-max day. The points trace "
  "a clean cyclic ring Tue→Wed→Thu→Fri→Sat→Sun→Mon→Tue with interior points filling the centre — "
  "the weekday cycle is recovered in behaviour space, and coverage is not collapsed onto a few vertices.",
  max_h=10.0*cm)
figure(os.path.join(RES,"pca_scatter_by_family.png"),
  "Same projection coloured by prompt family. Families occupy distinct, predictable regions "
  "(negation → faces, first-letter → {Tue,Thu}/{Sat,Sun} edges, open/semantic → interior), supporting "
  "H1: families land where intended (arg-max-in-intended = 97%).",
  max_h=10.0*cm)
figure(os.path.join(RES,"family_centroid_heatmap.png"),
  "Per-family mean weekday distribution (centroid). Each family carries a distinct day-bias signature; "
  "negation families suppress the excluded day, letter families concentrate on the matching pair.",
  max_h=9.5*cm)
figure(os.path.join(RES,"entropy_hist.png"),
  "Distribution of per-prompt entropy over the 7 days (max log 7 = 1.95 nats). Mass spans the whole "
  "range from near-deterministic answers to near-uniform beliefs; 39% of retained prompts are interior "
  "(entropy ≥ ½·log 7), giving genuine interior coverage rather than only vertex-like one-hots.",
  max_h=8.6*cm)
figure(os.path.join(RES,"pairwise_hellinger_hist.png"),
  "Pairwise Hellinger distances among retained distributions (p10/median/p90 = 0.39/0.74/0.95, min "
  "0.01, max 0.99). The wide spread confirms the points are not clustered — they range from near-"
  "identical to near-maximally separated.",
  max_h=8.6*cm)
figure(os.path.join(RES,"instruct_pca_scatter_by_argmax_day.png"),
  "Contrast — the instruct model + “one word”. Same ring, but points hug the corners as tight vertex "
  "clusters (interior fraction 0.26 vs base 0.39): RLHF decisiveness destroys the graded interior the "
  "map-fitting needs. This is why the base model was chosen.",
  max_h=10.0*cm)

# ===========================================================================
# STEP 2
# ===========================================================================
band("Step 2 — Behaviour subspace vs activation subspace (linear)")
para("<b>Reframing.</b> Compare the <b>behavioural subspace</b> (span of the Hellinger coords √p, "
     "~6-D) with the <b>activation subspace</b> (residual-stream h_ℓ) <b>linearly</b>, per layer — we "
     "do not assume a low-D curved manifold captures the whole subspace.")
para("<b>Method.</b> Capture last-token residual activations at every layer for all 225 prompts "
     "(one forward pass; shape 225×33×4096). On the 194 retained, per layer: standardise → PCA(top-40); "
     "then (a) <b>CCA canonical correlations</b> between activation top-PCs and behaviour coords "
     "(= cosines of principal angles between the subspaces), (b) <b>5-fold CV R²</b> of a linear decode "
     "activation→behaviour, (c) activation effective dimension (participation ratio).")
callout("The behavioural subspace is <b>linearly embedded in the late residual stream</b>. Best layer "
        "<b>L31</b>: all 6 canonical correlations &gt; 0.9 (mean 0.965), linear-decode <b>R² = 0.86</b>. "
        "Alignment shows a <b>sharp onset at ≈ L19</b> and saturates near the output.")
table([
  ["layers","mean canon. corr","# canon &gt; 0.9 (of 6)","linear-decode R² (CV)"],
  ["0–18 (early/mid)","0.31 – 0.55","0","negative (not linearly decodable)"],
  ["≈19 (onset)","0.55 → 0.82","2","turns positive (0.36)"],
  ["23–28","0.90 – 0.94","3 – 5","0.57 – 0.76"],
  ["29–32 (late)","0.95 – 0.965","6 / 6","0.81 – 0.86"],
], [4.0*cm, 4.0*cm, 4.7*cm, 4.7*cm])
para("Crucially, activation effective dimension at late layers is <b>~17–18</b> vs the <b>6-D</b> "
     "behavioural subspace — only ~6 activation directions are behaviour-aligned; the other ~12 encode "
     "off-behaviour structure (surface form, family). This is direct evidence that a 6-D manifold would "
     "<i>not</i> capture the activation variation, vindicating the subspace approach.")
figure(os.path.join(RES,"subspace_alignment_by_layer.png"),
  "Subspace alignment vs layer. Mean canonical correlation between the behaviour subspace and the "
  "activation top-PCs is flat-low through ~L18, then jumps sharply at ≈ L19 and saturates ~0.96 by L31. "
  "The weekday concept is a mid-to-late feature, absent at the input embedding.",
  max_h=9.0*cm)
figure(os.path.join(RES,"canonical_correlations_heatmap.png"),
  "All 6 canonical correlations across layers. The late layers turn the full set of 6 behaviour "
  "directions ‘on’ (all > 0.9 by L29–32): the entire behavioural subspace — not just its first axis — "
  "becomes linearly present in activations.",
  max_h=9.0*cm)
figure(os.path.join(RES,"best_layer_cca_scatter.png"),
  "Best layer L31: behaviour vs activation canonical variates. The two are essentially the same "
  "coordinate — canonical pair 1 lays the days out as a continuum (Sat/Sun→Fri→Mon/Tue/Wed), pair 2 is "
  "a second axis. Points lie on the diagonal (correlation 0.96).",
  max_h=8.6*cm)
figure(os.path.join(RES,"activation_effdim_by_layer.png"),
  "Activation effective dimension (participation ratio) by layer. Late layers carry ~17–18 effective "
  "dimensions; the behaviour-aligned subspace is only ~6 of them, so the concept is a low-D slice of a "
  "broader activation subspace.",
  max_h=8.6*cm)

# ===========================================================================
# STEP 3
# ===========================================================================
band("Step 3 — Mapping the activation subspace by perturbing anchors (causal)")
para("<b>Framing.</b> The 194 prompts are <b>anchors</b>: activation points known to decode to valid "
     "weekday behaviour. To map the activation subspace causally, perturb the last-token residual around "
     "each anchor at L29/30/31, patch it back through the model, and test whether the output is still a "
     "valid weekday distribution (Σweekday ≥ 0.90, other ≤ 0.10).")
para("<b>Design.</b> A forward hook overwrites the last-token residual on the anchor’s own prompt "
     "(earlier context intact). Sanity: zero-perturbation reproduces the anchor distribution "
     "(Hellinger 0.002–0.015) → the patch is correct. Directions are random unit vectors in the "
     "<b>anchor-spanned subspace</b> (PCA top-20) or in <b>raw R⁴⁰⁹⁶</b> (control); scale is in units of "
     "the median anchor-to-anchor distance (NN-dist ≈ 10–13).")
callout("The valid weekday region is a <b>continuous, low-dimensional, traversable</b> subspace. "
        "In-subspace steps move behaviour <b>5–6× more per unit distance</b> than random full-space "
        "steps; neighbouring anchors’ valid regions overlap into <b>one connected</b> region; a random "
        "walk stays 90% valid after 10 steps, 56% after 30.")
table([
  ["radius (×NN)","subspace valid%","subspace Hellinger-moved","full-space valid%","full Hellinger-moved"],
  ["1.0","97–99%","0.22–0.26","99%","0.04"],
  ["2.0","87–92%","0.39–0.44","90–97%","0.08–0.10"],
  ["4.0","44–58%","0.57–0.61","14–57%","0.16–0.18"],
  ["6.0","18–30%","0.65–0.68","0–1%","—"],
], [3.0*cm, 3.3*cm, 4.2*cm, 3.3*cm, 3.6*cm])
para("Random full-space directions are mostly orthogonal to the read-out — they barely change "
     "behaviour and merely break the model once pushed far (validity → 0 by r≈6). In-subspace the "
     "region degrades <i>gracefully</i> (still 17–30% valid at r=6, reaching the far simplex), whereas "
     "full-space fails <i>catastrophically</i>. So behaviour-relevant variation lives in the anchor "
     "subspace, and one should walk it — not raw R⁴⁰⁹⁶.")
figure(os.path.join(RES,"perturb_radial_all_layers.png"),
  "Radial validity and behaviour-displacement vs perturbation radius, for in-subspace (anchor-spanned) "
  "vs full-space directions, at L29/30/31. In-subspace steps move behaviour far more per unit distance "
  "and stay valid much further; full-space steps barely move behaviour then collapse. The three late "
  "layers behave almost identically (L29 slightly more robust at large radius).",
  max_h=11.5*cm)

# ===========================================================================
# STEP 4
# ===========================================================================
band("Step 4 — Shape of the valid region: is the map onto?")
para("<b>Question.</b> Does walking the valid activation subspace <b>fill</b> the behavioural simplex "
     "(map onto), or only smear locally around each anchor? And how <b>natural</b> are the reached "
     "points? Method: at L31, all 194 anchors × 24 in-subspace directions × radii {0.5,1,1.5,2,3}; for "
     "each valid perturbation record the simplex point reached and a naturalness score (distance to the "
     "nearest anchor, in NN-dist). 20,544 valid perturbations (88% of samples).")
callout("The map is largely <b>ONTO</b>. On a 40×40 Hellinger-plane grid, anchors occupy 159 cells; "
        "valid perturbations occupy <b>1234</b> (1075 new) — a <b>7.8× coverage gain</b>, with <b>73%</b> "
        "of valid perturbations landing where no anchor was. Same extent as the anchors (no blow-up), "
        "median naturalness 1.4 NN, all 7 day-regions filled.")
para("Reading, combined with Steps 2–3: the weekday concept occupies a <b>continuous, ~6-effective-dim, "
     "behaviour-aligned subspace</b> inside the late residual stream; the discrete anchors are a sample "
     "of it, and perturbing within the anchor span <b>interpolates the whole valid region</b>, which maps "
     "onto the behavioural simplex — established without assuming or fitting a manifold.")
figure(os.path.join(RES,"region_shape_overlay.png"),
  "Anchors (left) vs valid in-subspace perturbations (right) on the Hellinger plane. Perturbations "
  "densely fill the interior between anchors at the same overall extent (std 0.36 vs 0.36) — the valid "
  "activation region maps onto essentially the convex region the anchors outline, rather than spilling "
  "beyond it. 73% of valid points land in cells no anchor occupied.",
  max_h=10.5*cm)
figure(os.path.join(RES,"region_shape_naturalness.png"),
  "Naturalness of reached points = nearest-anchor distance (NN-dist units): median 1.4, p90 2.7, max "
  "3.0. The filled points sit <i>between</i> real anchors, not in far-off activation territory, and "
  "naturalness degrades smoothly toward the region’s edges.",
  max_h=9.0*cm)

# ===========================================================================
# STEP 5
# ===========================================================================
band("Step 5 — Concept-general routine + months (Δ¹²) generalisation")
para("The mapping procedure is packaged as a <b>concept-parametrised routine</b> "
     "(~~concept_core.py~~, ~~capture_concept.py~~, ~~map_subspace.py --concept~~): given a token set Z "
     "and a prompt set, it captures activations, filters to the valid region, isolates the "
     "behaviour-relevant low-D subspace at a layer, and densely samples it to map the valid activation "
     "region — for any concept.")
callout("<b>The routine generalises.</b> Months (Δ¹²): 269 prompts → 230 retained (86%), all 12 months "
        "single-token and balanced, 11/12 Hellinger-PCA dims, interior 0.40. Subspace map at L31: "
        "carrier-sanity 0.009, coverage-gain 6.2×. An overnight batch then swept layers, k and margin "
        "for both concepts.")
h2("Overnight sweep findings (both concepts)")
para("<b>Continuity with depth — concept-independent.</b> For both weekdays and months, map "
     "faithfulness improves monotonically L16≈0.40 → L20≈0.13 → L24≈0.04 → L31≈0.010, and subspace "
     "alignment jumps at ≈L19 (0.64→0.85) and climbs to ~0.96–0.97 by L31. The two concepts’ curves "
     "nearly overlap.")
para("<b>Behaviour-relevant dimensionality ≈ |Z|−1.</b> Coverage-gain saturates near k≈4–6 for "
     "weekdays (|Z|−1 = 6) and k≈6–10 for months (|Z|−1 = 11): the behaviour-relevant activation "
     "subspace scales with the simplex dimension. <b>The map is robust</b> (reseeded L31 maps "
     "identical; 120k-sample map matches 20–40k), and the valid region is bounded by the anchor hull "
     "(pushing the box past the hull lowers valid-fraction while coverage holds).")
figure(os.path.join(RES,"overnight_depth.png"),
  "Continuity with depth, weekdays vs months. Map faithfulness (carrier-sanity, lower = better) "
  "improves monotonically into the late layers and the two concepts’ curves nearly coincide — the "
  "geometry of how the concept subspace forms with depth is the same for Δ⁷ and Δ¹². Late layers "
  "(L30–31) give the cleanest map.",
  max_h=9.5*cm)
figure(os.path.join(RES,"overnight_ksweep.png"),
  "Simplex coverage-gain vs subspace dimension k, weekdays vs months. Coverage saturates near "
  "k ≈ |Z|−1 for each concept (≈6 for days, ≈11 for months); beyond the knee, extra dims add invalid "
  "volume without improving coverage. The behaviour-relevant dimension tracks the simplex dimension.",
  max_h=9.5*cm)
figure(os.path.join(MO,"map_subspace_k12_L31.png"),
  "Months (Δ¹²) subspace map at L31, k=12 — the generalisation in action. Densely sampling the "
  "behaviour-relevant activation subspace fills the month simplex (≈6× coverage gain), reproducing the "
  "weekday onto-ness on a 12-class concept.",
  max_h=11.0*cm)

# ===========================================================================
# STEP 6
# ===========================================================================
band("Step 6 — Leave-a-region-out recovery (method validation)")
para("<b>Test.</b> Delete <i>all</i> anchors in one behavioural region (e.g. every Wednesday-argmax "
     "prompt), rebuild the behaviour-relevant subspace + sampling box from the <b>remaining</b> anchors "
     "only, then sample that map and ask: do we <b>recover</b> the deleted region? A matched random "
     "hold-out (same count, scattered) is the control. Deleting a whole region is the hard extrapolation "
     "test; random deletion should recover trivially by interpolation. (L31, k=8, margin 1.5.)")
callout("<b>The method recovers deleted regions.</b> 87–100% of a deleted region’s behaviour cells are "
        "recovered by walking the training-only map, reaching P(deleted day) up to 1.0 — comparably to "
        "the random-deletion control. Monday (the dominant default, 49/194 anchors) is hardest and still "
        "recovers 87%.")
table([
  ["deleted region","# removed","region cell-recovery","max P(target)","proj. keeps-argmax","random-control"],
  ["Mon (dominant)","49","0.87","1.00","0.80","0.98"],
  ["Wed","31","0.96","1.00","0.77","0.93"],
  ["Fri","30","1.00","1.00","0.67","0.90"],
  ["Sat","19","1.00","1.00","0.74","1.00"],
], [3.1*cm, 2.0*cm, 3.3*cm, 2.4*cm, 3.2*cm, 2.4*cm])
para("67–80% of deleted anchors keep their day when projected onto the training-only subspace → the "
     "subspace built without the region still spans the region’s direction. The concept subspace is "
     "genuinely continuous, not a set of per-anchor memorised points: the anchors are a <b>sufficient</b> "
     "sample (regions recover) and <b>near-complete</b> (little valid mass outside their hull).")
for nm, cap in [
  ("recovery_Mon.png","Recovery of the deleted <b>Monday</b> region (the hardest — dominant default, 49 anchors removed). Walking the training-only map still reaches the Monday corner, recovering 87% of its cells with P(Mon) up to 1.0."),
  ("recovery_Wed.png","Recovery of the deleted <b>Wednesday</b> region (31 anchors removed): 96% cell-recovery, vertex reached."),
  ("recovery_Sat.png","Recovery of the deleted <b>Saturday</b> region (19 anchors removed): 100% cell-recovery, on par with the random-deletion control."),
]:
    figure(os.path.join(RES,nm), cap, max_h=8.2*cm)
figure(os.path.join(WK,"recovery_Fri.png"),
  "Recovery of the deleted <b>Friday</b> region (30 anchors removed): 100% cell-recovery. Across all "
  "four hold-outs the deleted day’s vertex is reachable from a map that never saw it.",
  max_h=8.2*cm)

# ===========================================================================
# STEP 7
# ===========================================================================
band("Step 7 — Inverse map (sharp onto-test)")
para("<b>Question.</b> Sharper than coverage: pick <b>target</b> simplex points and find an in-subspace "
     "activation that produces each; measure achievable behavioural error (Hellinger to target). Method: "
     "fit a linear forward map (subspace coords → √behaviour) on the anchors, invert via pseudo-inverse "
     "per target, reconstruct + patch to read the achieved distribution, then refine with a few local "
     "steps. Targets: 7 vertices, 21 two-token edges, the centroid, 60 Dirichlet interior + 40 sparse "
     "mixtures. (L29 & L31, k=8.)")
callout("The map is onto the model’s <b>natural reachable set</b>: any single-day distribution is hit "
        "essentially exactly (Hellinger 0.03, all 7), and 50/50 two-day edges are reachable "
        "(~86% within 0.20). Arbitrary interior mixtures carry a residual ~0.20 — not every abstract "
        "simplex point is a distribution the model naturally emits.")
table([
  ["target type","median Hellinger","reached ≤ 0.10","reached ≤ 0.20"],
  ["vertex (one-hot day)","0.03","100%","100%"],
  ["edge (50/50 two-day)","0.15","24%","86%"],
  ["uniform (centroid)","0.17","0%","100%"],
  ["interior (Dirichlet)","0.20","0%","50%"],
  ["sparse mixture","0.22","3%","38%"],
], [5.4*cm, 4.0*cm, 4.0*cm, 4.0*cm])
para("Part of the interior residual is the linear-inverse + light-refinement budget; a full gradient "
     "optimiser would tighten it (a follow-up). L29 ≈ L31 (vertices 0.06 vs 0.03). This complements "
     "Step 4/6: the reachable set is <b>vertex/edge-dense with a bounded interior residual</b>.")
figure(os.path.join(RES,"inverse_map_L31.png"),
  "Inverse map at L31: targets (open) vs achieved distributions (filled), by target type. Vertices and "
  "edges are hit tightly; interior/mixture targets show a bounded residual — the reachable set densely "
  "covers vertices and pairwise mixtures.",
  max_h=9.0*cm)
figure(os.path.join(MO,"inverse_map_L31.png"),
  "Months inverse map at L31 — the same vertex-tight, bounded-interior pattern holds on Δ¹², confirming "
  "the inverse behaviour is concept-general.",
  max_h=9.0*cm)

# ===========================================================================
# STEP 8
# ===========================================================================
band("Step 8 — Sparse-anchor steering completeness")
para("<b>Question (the steering goal).</b> Can we anchor a <b>sparse</b> subset of days and still reach "
     "valid activations for the <b>unprompted</b> regions? Keep only a few days’ anchors, build the "
     "subspace from those alone, and measure how many held-out days become reachable — sweeping subspace "
     "dim k and method (CCA / PCA / diff). (L31, margin 2.0; “recovered” = a valid sample reaches "
     "P(day) ≥ 0.5.)")
callout("<b>Sparse anchoring + enough dimensions recovers the whole set.</b> From just {Mon,Thu} we "
        "reach P≈1.0 on all 5 unprompted days at k≥8; the contiguous case (held days all on one side — "
        "pure extrapolation) also recovers fully. <b>k ≈ |Z|−1 is the knob.</b>")
table([
  ["anchored (kept)","held-out (unprompted)","recovered @ k=4","recovered @ k≥8","strength"],
  ["{Mon, Thu}  (2)","Tue,Wed,Fri,Sat,Sun  (5)","3/5","5/5 (all 3 methods)","P(day) ≈ 1.0 each, 100s samples/day"],
  ["{Mon,Wed,Fri,Sun}  (4)","Tue,Thu,Sat  (3)","1/3","3/3","—"],
  ["{Mon,Tue,Wed,Thu}  (4, contig.)","Fri,Sat,Sun  (3, extrapolation)","1/3","3/3 (cca/pca; diff @ k≥12)","—"],
], [4.3*cm, 4.3*cm, 2.6*cm, 3.0*cm, 3.2*cm], align_center_from=2)
para("<b>The dimensionality cap matters for wide gamuts.</b> CCA is hard-capped at the behaviour "
     "dimension |Z|+1 (=8 for days): its k=12/20/40 requests collapse to 8 — fine for days, but for a "
     "wide gamut needing more effective directions, <b>CCA cannot supply them; PCA/diff (pure activation "
     "variance) can</b>. This is the key thing carried to colours (Δ⁴⁵).")
for nm, cap in [
  ("sparse_recovery_MonThu.png","Anchor only {Mon,Thu}; recover all 5 unprompted days. At k≥8 every held-out day reaches P≈1.0 with hundreds of valid samples — two anchored days suffice to span the directions toward the rest."),
  ("sparse_recovery_MonWedFriSun.png","Anchor {Mon,Wed,Fri,Sun}; recover the 3 interleaved held-out days (Tue,Thu,Sat) — 3/3 at k≥8."),
  ("sparse_recovery_MonTueWedThu.png","Anchor contiguous {Mon–Thu}; recover {Fri,Sat,Sun}. This is pure extrapolation (held days all on one side) and still recovers fully (cca/pca at k≥8)."),
]:
    figure(os.path.join(RES,nm), cap, max_h=7.6*cm)

# ===========================================================================
# DAY 3
# ===========================================================================
band("Day 3 — Continuous steering demo + colours")
para("The payoff. ~~steer_trajectory.py~~ inverts the map at each waypoint of a target path (the "
     "weekday cycle, month cycle, colour rainbow), patches the activation, and records the achieved "
     "behaviour — i.e. <b>continuously steering the model across a gamut by walking the activation "
     "subspace</b>. Good steering = low fidelity Hellinger (achieved tracks target), small uniform steps, "
     "high valid fraction, and an arg-max sequence that sweeps the intended order.")
para("<b>Colours (Δ⁴⁵).</b> Of a 150-name candidate list, 80 colours are single-token on Llama; "
     "curating to canonical colours gives Δ⁴⁵ (45 single-token colours), 127 anchors, 50% retained, ~10 "
     "behaviour dims — usable, though messier than days/months. (Gemma’s tokenizer has 94 single-token "
     "colours, a strict superset — a richer gamut for cross-model work.) Two fixes landed this batch: "
     "months recovery was a <b>false negative at k=8 → corrected to k=13</b> (k must scale with |Z|), and "
     "a global OOM/speed fix (~~logits_to_keep=1~~, last-position logits only).")
callout("The steering trajectories track their target paths through behaviour space — the model is "
        "steered continuously across the weekday cycle, month cycle and colour gamut by walking the "
        "activation subspace. (Day-3 jobs 12505–12508; colour fidelity improves with k, confirming the "
        "k ≈ |Z|−1 law on Δ⁴⁵.)", label="DEMO", hexcol="#1f6f43")
figure(os.path.join(WK,"steer_weekdays_L31k8.png"),
  "Weekday continuous steering at L31, k=8. The target path (the weekday cycle through behaviour space) "
  "vs the achieved steered trajectory over the anchors: inverting the map at each waypoint and patching "
  "the activation sweeps the model’s belief Mon→Tue→…→Sun.",
  max_h=9.5*cm)
figure(os.path.join(MO,"steer_months_L23k11.png"),
  "Month continuous steering (L23, k=11 ≈ |Z|−1). The achieved trajectory tracks the month-cycle target "
  "path on Δ¹², demonstrating the steering demo is concept-general.",
  max_h=9.5*cm)
figure(os.path.join(CO,"map_subspace_k24_L31.png"),
  "Colour (Δ⁴⁵) subspace map at L31, k=24. Even on a 45-class concept the behaviour-relevant subspace, "
  "sampled densely, covers the colour simplex — the curated 45-colour set is messier than days/months "
  "but the onto-ness survives at adequate k (k ≈ |Z|−1 ⇒ needs the larger k here).",
  max_h=10.5*cm)
figure(os.path.join(CO,"recovery_Blue.png"),
  "Colour recovery — deleting the Blue region and rebuilding from the rest still reaches the Blue corner "
  "(at adequate k), extending the leave-a-region-out validation from Δ⁷/Δ¹² to Δ⁴⁵.",
  max_h=8.2*cm)

# ===========================================================================
# METHOD NOTES / ISSUES
# ===========================================================================
band("Methodological notes & issues")
para("<b>1 — Prompt frame dominates (resolved).</b> Eliciting high-mass <i>and</i> spread distributions "
     "was entirely a function of the prompt frame, not content. Bare/Q:A stems leak 30–50% mass to prose "
     "lead-ins; copular stems leak; <b>answer-first colon stems</b> fix it. The few-shot must be "
     "<b>neutral</b> (non-weekday exemplars — else it leaks the answer space) and <b>terse</b> (adding "
     "“…, for example” pushed ‘other’ mass 0.05→0.19 and broke the spread families).")
para("<b>2 — Instruct + “one word” collapses the simplex (decided against).</b> 98% mass but interior "
     "0.26 — points pile on the 7 vertices. Base preserves graded belief (0.39) and matches the "
     "manifold-steering activation work.")
para("<b>3 — cinaps: always set ~~--mem~~.</b> No ~~--mem~~ → the job defaults to a full node’s RAM "
     "(~773 GB) and sits PENDING ~a year out, never running. All sbatch scripts use "
     "~~--mem=96G --cpus-per-task=8 --gpus=1~~.")
para("<b>4 — k must scale with |Z|.</b> Recovery/sparse need k ≈ |Z|−1 (months ~13, not the weekday 8); "
     "the k=8 months-recovery run was a false negative. CCA is capped at |Z|+1, so wide gamuts (colours) "
     "must use PCA/diff.")
para("<b>5 — Frame is a held-fixed confound for M_h.</b> The neutral few-shot prefix is part of every "
     "prompt, so captured activations encode it too — fine because it is identical across all prompts, "
     "but recorded so M_y↔M_h pairing compares like with like.")
h2("Open / next")
para("(1) Full <b>gradient inverse optimiser</b> to tighten the ~0.20 interior-mixture residual. "
     "(2) <b>Colours suite at the right k</b> + <b>Gemma cross-model</b> (richer 94-colour simplex; "
     "does the subspace story hold on a different architecture?). (3) <b>Alphabet Δ²⁶</b> to stress-test "
     "large |Z|. (4) Housekeeping: top up weekday prompts to ≥200 retained; prune the weak first-letter "
     "paraphrase.")
h2("Reproduce")
story.append(Table([[Paragraph(
  'python code/methods/weekday_prompts/prompts.py            # prompts (local)<br/>'
  'sbatch code/analyses/simplex_coverage/sbatch_full_base.sh  # coverage on cinaps (--mem REQUIRED)<br/>'
  'bash   run/pull_results.sh                                 # pull artifacts + render figures',
  st_code)]], colWidths=[17.4*cm], style=TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#eef2f5")),
    ("BOX",(0,0),(-1,-1),0.5,RULE),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)])))
story.append(Spacer(1,6))
para("Artifacts: ~~artifacts/{weekday,months,colors}_simplex/{model}/simplex_coverage/{distributions."
     "safetensors, prompts_scored.json, coverage_metrics.json, figures/}~~.", st_cap)

# ===========================================================================
# APPENDIX — representative sweep frames
# ===========================================================================
story.append(PageBreak())
band("Appendix — representative sweep frames")
para("One representative frame per parameter sweep. These are the raw exploratory grids underlying the "
     "curated figures above; the full sweeps (every k, every layer, every margin, both seeds, and all "
     "months/colours/instruct variants) live under the session’s ~~artifacts/~~ tree.")

h2("A. Weekday subspace-map sweeps (L31 unless noted)")
for path, cap in [
  (os.path.join(WK,"map_subspace_L31.png"), "Canonical weekday map at L31 (default k). The reference frame the curated onto-ness figures derive from."),
  (os.path.join(WK,"map_subspace_k8_L31.png"), "k-axis representative (k=8). Coverage-gain saturates near k ≈ |Z|−1 = 6; k=8 is just above the knee."),
  (os.path.join(WK,"map_subspace_layersweep_L31.png"), "Layer-axis representative (L31). The layer sweep (L16→L31) shows the map sharpening into the late stream; L31 is cleanest."),
  (os.path.join(WK,"map_subspace_margin1.5_L31.png"), "Margin-axis representative (margin 1.5). Extending the sampling box past the anchor hull lowers valid-fraction (0.81→0.29 over 0.6→4.0) while coverage holds — the region is anchor-bounded."),
  (os.path.join(WK,"map_subspace_seed1_L31.png"), "Seed/robustness representative (seed 1). Reseeded maps are essentially identical (carrier 0.011–0.012, cov-gain 7.62–7.67) — metrics are converged, not sample-starved."),
  (os.path.join(WK,"map_subspace_complete_L31.png"), "Highest-resolution ‘complete’ weekday map (120k samples, L31). Matches the 20–40k runs — convergence confirmation."),
]:
    figure(path, cap, max_h=8.8*cm)

h2("B. Months (Δ¹²)")
for path, cap in [
  (os.path.join(MO,"map_subspace_layersweep_L31.png"), "Months layer-sweep representative (L31) — same depth-sharpening as weekdays."),
  (os.path.join(MO,"recovery_Jan_k13.png"), "Months leave-a-region-out recovery for January at the corrected k=13 (k=8 was a false negative)."),
  (os.path.join(MO,"sparse_recovery_JanAprJulOct.png"), "Months sparse-anchor steering: anchor 4 quarter-start months, recover the rest at k ≈ |Z|−1."),
]:
    figure(path, cap, max_h=8.6*cm)

h2("C. Colours (Δ⁴⁵)")
for path, cap in [
  (os.path.join(CO,"map_subspace_layersweep_L31.png"), "Colour layer-sweep representative (L31)."),
  (os.path.join(CO,"map_subspace_validate_L31.png"), "Colour map validation frame (L31) — held-out check on the colour subspace map."),
]:
    figure(path, cap, max_h=8.8*cm)

h2("D. Instruct-model comparison (rejected frame)")
for path, cap in [
  (os.path.join(WKI,"pca_scatter_by_family.png"), "Instruct model, behaviour PCA by family — tighter, more vertex-hugging clusters than base; the interior the map needs is suppressed."),
  (os.path.join(WKI,"family_centroid_heatmap.png"), "Instruct family centroids — sharper (more one-hot) than base, reflecting RLHF decisiveness."),
  (os.path.join(WKI,"entropy_hist.png"), "Instruct entropy histogram — mass shifted toward low entropy (vertices), interior fraction 0.26 vs base 0.39."),
]:
    figure(path, cap, max_h=8.4*cm)

# ---- page furniture --------------------------------------------------------
TITLE_RUN = "Weekday-simplex-map · behaviour↔activation subspace map · Llama-3.1-8B"
def deco(canvas, doc):
    canvas.saveState()
    canvas.setFont("DJ", 7.5); canvas.setFillColor(MUT)
    canvas.drawString(2.0*cm, 1.05*cm, TITLE_RUN)
    canvas.drawRightString(A4[0]-2.0*cm, 1.05*cm, f"p. {doc.page}")
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(2.0*cm, 1.30*cm, A4[0]-2.0*cm, 1.30*cm)
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        topMargin=1.7*cm, bottomMargin=1.7*cm,
                        title="Weekday-simplex-map — consolidated results",
                        author="causalab research session (candid-marlin)")
doc.build(story, onFirstPage=deco, onLaterPages=deco)
print("WROTE", OUT)
print("FIGURES", FIG[0])
