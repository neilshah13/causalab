"""Build the fully ILLUSTRATED conclusions PDF (result/CONCLUSIONS.pdf).

Takes result/CONCLUSIONS.md, converts to HTML, and injects ~16 figures with captions at
anchored points in the text (summary figures F1-F6 + the strongest per-section evidence
figures from the session), then lays it out to A4 via PyMuPDF Story.

Run from the session root:  python3 code/analyses/simplex_coverage/build_conclusions_pdf.py
"""

import os
import re

import fitz
import markdown

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RF = "result/figures"
WF = "artifacts/weekday_simplex/llama31_8b/simplex_coverage/figures"
CF = "artifacts/colors_simplex/llama31_8b/simplex_coverage/figures"
MF = "artifacts/months_simplex/llama31_8b/simplex_coverage/figures"


def fig(path, caption, width=500):
    return (f'<div class="figblock"><img src="{path}" style="width:{width}px"/>'
            f'<p class="cap">{caption}</p></div>')


# anchor substring (in the markdown) -> figure html injected AFTER the paragraph containing it
INJECT = [
    ("never measures the geometry of the *connections* between them",
     fig(f"{RF}/C1_vectors_vs_manifold.png",
         "<b>C1 — The manifold is its points.</b> Left: steering through intermediate days — "
         "plain centroid-to-centroid vectors (red) match the fitted manifold (blue) exactly, while "
         "a single straight vector (grey) fails. Middle: placing a never-seen day — a simple vector "
         "beats every manifold variant. Right: this session, the paper's own naturalness score — by "
         "the last layers the manifold is indistinguishable from a straight vector.", 520)),
    ("no fitted curve is needed to see or to use it",
     fig(f"{RF}/C2_mapping_the_geometry.png",
         "<b>C2 — How we map the geometry.</b> Anchor prompts cover the whole day-of-week region in "
         "behaviour space (left); the same anchors carry the week's arrangement in activation space "
         "(middle); walking around them discovers the full activation&rarr;behaviour map (right). "
         "Animated version: <i>result/figures/G1_discovering_the_map.gif</i>.", 520)),
    ("not at the very last layer",
     fig(f"{RF}/C3_natural_steering.png",
         "<b>C3 — More natural steering, coherent output.</b> Left: the paper's naturalness score "
         "on identical endpoints — our map-based steering (red) stays 2&ndash;5&times; closer to "
         "natural behaviour than the paper's straight-vector and manifold steering. Right: example "
         "continuations after steering — the model keeps writing normal text in every case.", 520)),
    ("a word whose natural ceiling is",
     fig(f"{RF}/C4_few_anchors_full_concept.png",
         "<b>C4 — Anchor two days, recover the whole week.</b> The map is rebuilt from Monday and "
         "Thursday anchors only; walking it reaches every other day at probability 0.8&ndash;1.0. "
         "Animated version: <i>result/figures/G2_recovering_unprompted_days.gif</i>.", 430) +
     fig(f"{RF}/F4_unprompted_steering.png",
         "<b>F4 — Steering past the model's habits.</b> Map built from red/green/blue prompts only; "
         "bars = probability reached for each never-prompted colour, ticks = that colour's natural "
         "ceiling. &#9650; marks colours pushed <i>above</i> anything the model produces on its own "
         "(scarlet: steered 0.86 vs ceiling 0.01).", 520)),
]


def main():
    md_src = open(os.path.join(S, "result", "CONCLUSIONS.md")).read()
    # inject figure blocks after the paragraph containing each anchor
    paras = md_src.split("\n\n")
    out_paras = []
    used = set()
    for p in paras:
        out_paras.append(p)
        for i, (anchor, html) in enumerate(INJECT):
            key = " ".join(anchor.split())
            flat = " ".join(p.split())
            if i not in used and key in flat:
                out_paras.append(html)
                used.add(i)
    missing = [INJECT[i][0][:50] for i in range(len(INJECT)) if i not in used]
    if missing:
        print("WARNING — unmatched anchors:", missing)
    body = markdown.markdown("\n\n".join(out_paras), extensions=["tables", "fenced_code"])

    css = """
    body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.45; }
    h1 { font-size: 16pt; color: #1a1a1a; border-bottom: 2px solid #444; padding-bottom: 4pt; }
    h2 { font-size: 12.5pt; color: #b03030; margin-top: 14pt; }
    code { font-family: monospace; font-size: 8.5pt; color: #333; }
    li { margin-bottom: 3pt; }
    strong { color: #111; }
    hr { border: 0.5pt solid #999; }
    .figblock { margin: 10pt 0 12pt 0; }
    table { font-size: 8.5pt; } td, th { padding: 2pt 6pt; border-bottom: 0.4pt solid #ccc; }
    .cap { font-size: 8pt; color: #555; margin-top: 2pt; line-height: 1.3; }
    """
    story = fitz.Story(html=body, user_css=css, archive=fitz.Archive(S))
    writer = fitz.DocumentWriter(os.path.join(S, "result", "CONCLUSIONS.pdf"))
    page_rect = fitz.paper_rect("a4")
    content = page_rect + (36, 40, -36, -40)
    more = True
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(content)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = fitz.open(os.path.join(S, "result", "CONCLUSIONS.pdf"))
    print(f"CONCLUSIONS.pdf written: {len(doc)} pages, {len(used)}/{len(INJECT)} figure blocks injected")


if __name__ == "__main__":
    main()
