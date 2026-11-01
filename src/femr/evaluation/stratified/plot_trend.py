"""Forest plot of ORA's gain in the high- vs low-history regime.

One panel per task type, one row per baseline. A point at zero means ORA's advantage
is the same whether the prediction is made after little or a lot of history.

    python plot_trend.py
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from femr.evaluation.stratified.paths import BASELINES, OUT_ROOT  # noqa: E402
from femr.evaluation.stratified.uniformity import (  # noqa: E402
    TASK_TYPES,
    delta_samples,
)

INK = "#0b0b0b"
MUTED = "#52514e"
ACCENT = "#2a78d6"
GRID = "#d8d8d4"
BAND = "#ececeb"
LABEL = {"cc_llama": "Context Clues", "icarefm": "iCareFM", "motor": "MOTOR",
         "transformer_survivehr": "SurvivEHR"}
NEGLIGIBLE = 0.01
PANEL = {"classification": ("classification", "AUROC"),
         "tte": ("time-to-event", "C-index"),
         "regression": ("regression", "R\u00b2")}


def pooled_trend(ttype, baseline):
    """gain(high) - gain(low), averaged over the subtasks of one task type."""
    stack = []
    for family, task in TASK_TYPES[ttype]:
        hi = delta_samples(family, task, "high", baseline)
        lo = delta_samples(family, task, "low", baseline)
        n = min(len(hi), len(lo))
        stack.append(hi[:n] - lo[:n])
    n = min(len(s) for s in stack)
    pooled = np.mean([s[:n] for s in stack], axis=0)
    lo95, hi95 = np.percentile(pooled, [2.5, 97.5])
    return float(pooled.mean()), float(lo95), float(hi95)


def main():
    types = list(TASK_TYPES)
    fig, axes = plt.subplots(1, len(types), figsize=(10.5, 3.1), sharey=True)
    y = np.arange(len(BASELINES))[::-1]

    for ax, ttype in zip(axes, types):
        est = [pooled_trend(ttype, b) for b in BASELINES]
        span = max(max(abs(l), abs(h)) for _, l, h in est) * 1.25

        ax.axvspan(-NEGLIGIBLE, NEGLIGIBLE, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)

        for yi, (m, lo, hi) in zip(y, est):
            excludes_zero = lo > 0 or hi < 0
            ax.plot([lo, hi], [yi, yi], color=ACCENT, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([m], [yi], marker="o", ms=8, zorder=3, color=ACCENT,
                    markerfacecolor=ACCENT if excludes_zero else "white",
                    markeredgecolor=ACCENT, markeredgewidth=2)

        ax.set_xlim(-span, span)
        ax.set_ylim(-0.6, len(BASELINES) - 0.4)
        name, metric = PANEL[ttype]
        ax.set_title(f"{name}  ({len(TASK_TYPES[ttype])} tasks)",
                     fontsize=10, color=INK, pad=8)
        ax.set_xlabel(f"Δ {metric}   (high − low history)",
                      fontsize=9, color=MUTED)
        ax.tick_params(axis="both", labelsize=9, colors=MUTED, length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"ORA − {LABEL[b]}" for b in BASELINES],
                            fontsize=9, color=INK)

    fig.suptitle("ORA's advantage does not grow with the amount of preceding history",
                 fontsize=11, color=INK, y=1.04, x=0.02, ha="left")
    fig.text(0.02, -0.10,
             "Point = pooled gain difference between the longest- and shortest-history "
             "thirds; bar = 95% bootstrap CI.\nHollow point = CI covers zero. "
             f"Shaded band = ±{NEGLIGIBLE:g}, a negligible difference.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout()

    path = os.path.join(OUT_ROOT, "trend_forest.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {path} and {path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
