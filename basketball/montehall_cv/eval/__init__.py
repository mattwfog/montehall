"""Broadcast×PBP evaluation: clock alignment, ESPN PBP fetch, truth bridge.

The PBP truth is the only scoreboard for broadcast-trained models.
This package holds
clock OCR (`clock_ocr`), play alignment (`pbp_align`), the ESPN label
source (`espn_pbp`), and the truth bridge (`pbp_truth`) that turns an
aligned game into the anchors-truth format scored by the ONE scorer,
scripts/score_e2e_vs_anchors.py (the three-level `pbp_score` was retired
into it 2026-08-26).
"""
