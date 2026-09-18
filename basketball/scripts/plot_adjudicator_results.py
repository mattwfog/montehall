"""Static SVG figures for the adjudicator results, light and dark.

Reads results/adjudicators-sim-recalibration.json and
results/adjudicators-sim-sweep.json, writes docs/figures/*.svg. Standard library
only, so the figures rebuild anywhere the results do:

    python scripts/plot_adjudicator_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

# Each figure carries two series: jev in blue, its comparison in orange. The pair
# passes the palette validator (CVD, normal-vision and contrast checks) on both
# surfaces; a third hue that also passed against both could not be found.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "jev": "#2a78d6", "recalibrated": "#eb6834", "rule": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "jev": "#3987e5", "recalibrated": "#d95926", "rule": "#d95926",
    },
}
FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""


class Panel:
    def __init__(self, x, y, w, h, xlim, ylim):
        self.x, self.y, self.w, self.h, self.xlim, self.ylim = x, y, w, h, xlim, ylim

    def px(self, v):
        return self.x + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h


def _text(x, y, s, fill, size=12, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{s}</text>')


def _frame(panel: Panel, t, xticks, yticks, xfmt, yfmt, xlabel, ylabel) -> list[str]:
    out = []
    for v in yticks:
        y = panel.py(v)
        out.append(f'<line x1="{panel.x}" x2="{panel.x + panel.w}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(_text(panel.x - 8, y + 4, yfmt(v), t["muted"], 11, "end"))
    for v in xticks:
        out.append(_text(panel.px(v), panel.y + panel.h + 18, xfmt(v), t["muted"], 11, "middle"))
    base = panel.y + panel.h
    out.append(f'<line x1="{panel.x}" x2="{panel.x + panel.w}" y1="{base}" y2="{base}" '
               f'stroke="{t["axis"]}" stroke-width="1"/>')
    out.append(_text(panel.x + panel.w / 2, base + 38, xlabel, t["ink2"], 12, "middle"))
    if ylabel:
        cx, cy = panel.x - 44, panel.y + panel.h / 2
        out.append(f'<text x="{cx}" y="{cy}" {FONT} font-size="12" fill="{t["ink2"]}" '
                   f'text-anchor="middle" transform="rotate(-90 {cx} {cy})">{ylabel}</text>')
    return out


def _series(panel: Panel, points, color, t, label=None, label_dy=0, weights=None) -> list[str]:
    """weights (one count per point) scales marker area, so a bin of 3 verdicts
    cannot look as heavy as a bin of 130."""
    path = " ".join(f'{"M" if i == 0 else "L"}{panel.px(x):.1f},{panel.py(y):.1f}'
                    for i, (x, y) in enumerate(points))
    out = [f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity='
           f'"{0.45 if weights else 1}" stroke-linejoin="round" stroke-linecap="round"/>']
    top = max(weights) if weights else 1
    for i, (x, y) in enumerate(points):
        r = 4.5 if not weights else 3.0 + 8.0 * (weights[i] / top) ** 0.5
        out.append(f'<circle cx="{panel.px(x):.1f}" cy="{panel.py(y):.1f}" r="{r:.1f}" fill="{color}" '
                   f'stroke="{t["surface"]}" stroke-width="2"/>')
    if label:
        x, y = points[-1]
        out.append(_text(panel.px(x) + 10, panel.py(y) + 4 + label_dy, label, t["ink2"], 12))
    return out


def _legend(x, y, items, t) -> list[str]:
    out = []
    for name, color in items:
        out.append(f'<rect x="{x}" y="{y - 9}" width="14" height="4" rx="2" fill="{color}"/>')
        out.append(_text(x + 20, y - 3, name, t["ink2"], 12))
        x += 30 + 7 * len(name)
    return out


def _svg(w, h, t, title, subtitle, body) -> str:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{title}">',
        f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>',
        _text(24, 34, title, t["ink"], 16, weight="600"),
        _text(24, 54, subtitle, t["ink2"], 12),
    ]
    return "\n".join(head + body + ["</svg>"]) + "\n"


def reliability(theme: str) -> str:
    t = THEMES[theme]
    data = json.loads((RESULTS / "adjudicators-sim-recalibration.json").read_text())
    panel = Panel(72, 96, 420, 300, (0.0, 1.0), (0.0, 1.0))
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    body = _frame(panel, t, ticks, ticks, lambda v: f"{v:.2f}", lambda v: f"{v:.0%}",
                  "confidence the adjudicator reported", "share of those verdicts that were right")
    body.append(f'<line x1="{panel.px(0)}" y1="{panel.py(0)}" x2="{panel.px(1)}" y2="{panel.py(1)}" '
                f'stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="4 4"/>')
    body.append(_text(panel.px(0.04), panel.py(0.04) - 14, "perfectly calibrated", t["muted"], 11))
    before, after = data["test_before"]["reliability"], data["test_after"]["reliability"]
    body += _series(panel, [(b["mean_confidence"], b["accuracy"]) for b in before], t["jev"], t,
                    weights=[b["n"] for b in before])
    body += _series(panel, [(b["mean_confidence"], b["accuracy"]) for b in after], t["recalibrated"], t,
                    weights=[b["n"] for b in after])
    body += _legend(72, 84, [(f'as reported (ECE {data["test_before"]["ece"]:.2f})', t["jev"]),
                             (f'recalibrated on another seed (ECE {data["test_after"]["ece"]:.2f})',
                              t["recalibrated"])], t)
    subtitle = (f'{data["possessions_each"]} simulated possessions, seed {data["test_seed"]}; '
                f'map fitted on seed {data["fit_seed"]}; marker area = number of verdicts in the bin')
    return _svg(720, 452, t, "Reported confidence against observed accuracy", subtitle, body)


def sensor_sweep(theme: str) -> str:
    t = THEMES[theme]
    data = json.loads((RESULTS / "adjudicators-sim-sweep.json").read_text())
    detects = sorted({c["shot_detect_p"] for c in data["cells"]})
    body: list[str] = []
    ticks_y = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, detect in enumerate(detects):
        panel = Panel(72 + i * 330, 112, 250, 280, (0.78, 1.02), (0.5, 1.0))
        cells = sorted((c for c in data["cells"] if c["shot_detect_p"] == detect),
                       key=lambda c: c["made_flag_p"])
        xticks = [c["made_flag_p"] for c in cells]
        body += _frame(panel, t, xticks, ticks_y, lambda v: f"{v:.1%}".replace(".0%", "%"),
                       lambda v: f"{v:.0%}", "made flag is right", "outcome accuracy" if i == 0 else "")
        body.append(_text(panel.x, panel.y - 12, f"shots detected {detect:.0%} of the time",
                          t["ink"], 12, weight="600"))
        body += _series(panel, [(c["made_flag_p"], c["naive_rule_accuracy"]) for c in cells], t["rule"], t)
        body += _series(panel, [(c["made_flag_p"], c["accuracy"]) for c in cells], t["jev"], t)
    body += _legend(72, 84, [("jev, on the possessions it answers", t["jev"]),
                             ("no model: trust the last shot flag, on all possessions", t["rule"])], t)
    subtitle = f'{data["possessions"]} simulated possessions per point, seed {data["seed"]}; same games, only the sensors change'
    return _svg(760, 448, t, "Outcome accuracy climbs with sensor quality, for the model and the rule alike", subtitle, body)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        for name, draw in (("adjudicator-reliability", reliability), ("adjudicator-sensor-sweep", sensor_sweep)):
            path = FIGURES / f"{name}-{theme}.svg"
            path.write_text(draw(theme))
            print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
