#!/usr/bin/env python3
"""Standalone Fig 8: cross-lingual steering coherence vs same-language baseline.

Restyled to the talk-deck spec. The figure definition now lives in
``plot_manifolds.py`` (single source of truth, deck palette); this wrapper
just renders Fig 8 on its own. Data is from Cinaps jobs 12573-12575
(2026-06-12). Run with:

    uv run python agent_logs/2026-06-10--cjk-hindi-cycles--keen-panda/code/plot_fig8.py
"""

from __future__ import annotations
from plot_manifolds import fig8_steering_coherence

if __name__ == "__main__":
    fig8_steering_coherence()
    print("Done.")
