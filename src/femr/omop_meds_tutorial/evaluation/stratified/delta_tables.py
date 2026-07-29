"""One compact table per task type: ORA minus each baseline, by history regime.

Rows are the four baselines, columns the three regimes, cells the gain averaged over the
subtasks of that type. The point of the layout is to make it obvious at a glance whether
the gap is the same size in every regime.

    python delta_tables.py
"""

import os

import numpy as np
import pandas as pd

from femr.omop_meds_tutorial.evaluation.stratified.common import REGIMES, to_markdown
from femr.omop_meds_tutorial.evaluation.stratified.paths import BASELINES, OUT_ROOT
from femr.omop_meds_tutorial.evaluation.stratified.uniformity import (
    TASK_TYPES,
    TYPE_METRIC,
    delta_samples,
)


def build(ttype):
    cohorts = TASK_TYPES[ttype]
    rows, raw = [], []
    for base in BASELINES:
        row = {"baseline": base}
        per_regime = {}
        for regime in REGIMES:
            stack = [delta_samples(family, task, regime, base) for family, task in cohorts]
            n = min(len(s) for s in stack)
            avg = np.mean([s[:n] for s in stack], axis=0)      # across subtasks, per replicate
            per_task = np.array([s.mean() for s in stack])     # one gain per subtask
            lo, hi = np.percentile(avg, [2.5, 97.5])
            row[regime] = f"{avg.mean():+.3f}"
            row[f"{regime}_spread"] = f"{per_task.min():+.3f} to {per_task.max():+.3f}"
            per_regime[regime] = avg
            raw.append({"task_type": ttype, "baseline": base, "regime": regime,
                        "mean_gain": float(avg.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                        "min_subtask_gain": float(per_task.min()),
                        "max_subtask_gain": float(per_task.max()),
                        "n_subtasks": len(cohorts)})
        n = min(len(per_regime["high"]), len(per_regime["low"]))
        diff = per_regime["high"][:n] - per_regime["low"][:n]
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        row["high - low"] = f"{diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        row["high - low (mean (std))"] = f"{diff.mean():+.4f} ({diff.std(ddof=1):.4f})"
        rows.append(row)
        raw[-1].update({"trend_mean": float(diff.mean()), "trend_std": float(diff.std(ddof=1)),
                        "trend_ci_lo": float(dlo), "trend_ci_hi": float(dhi)})
    return pd.DataFrame(rows), pd.DataFrame(raw)


def per_subtask_trend(ttype, base):
    """high - low for one baseline, computed separately within each subtask."""
    out = []
    for family, task in TASK_TYPES[ttype]:
        hi = delta_samples(family, task, "high", base)
        lo = delta_samples(family, task, "low", base)
        n = min(len(hi), len(lo))
        d = hi[:n] - lo[:n]
        out.append((task, float(d.mean()), float(d.std(ddof=1))))
    return out


def main():
    lines, raws = [], []
    w = lines.append
    w("# ORA minus baseline, by history-length regime\n")
    w("Each cell is the gain averaged over the subtasks of that task type, with a 95%")
    w("bootstrap interval. `*_spread` gives the range of the per-subtask gains, so a wide")
    w("spread flags that the average is hiding disagreement between subtasks.\n")

    for ttype in TASK_TYPES:
        table, raw = build(ttype)
        raws.append(raw)
        n = len(TASK_TYPES[ttype])
        w(f"## {ttype} -- {TYPE_METRIC[ttype]}, averaged over {n} subtasks\n")
        w(to_markdown(table[["baseline"] + list(REGIMES) + ["high - low"]]))
        w("")
        w("Per-subtask range within each regime:\n")
        w(to_markdown(table[["baseline"] + [f"{r}_spread" for r in REGIMES]]))
        w("")
        w("Per-subtask `high - low`, mean (std):\n")
        cols = [t for _, t in TASK_TYPES[ttype]]
        per = pd.DataFrame(
            [{"baseline": b, **{task: f"{m:+.4f} ({s:.4f})"
                                for task, m, s in per_subtask_trend(ttype, b)}}
             for b in BASELINES]
        )[["baseline"] + cols]
        w(to_markdown(per))
        w("")

    w("## `high - low` only, mean (std)\n")
    w("Averaged over the subtasks of each task type.\n")
    for ttype in TASK_TYPES:
        table, _ = build(ttype)
        w(f"**{ttype}** ({TYPE_METRIC[ttype]}, {len(TASK_TYPES[ttype])} subtasks)\n")
        w(to_markdown(table[["baseline", "high - low (mean (std))"]]
                      .rename(columns={"high - low (mean (std))": "high - low"})))
        w("")

    path = os.path.join(OUT_ROOT, "delta_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.concat(raws, ignore_index=True).to_csv(
        os.path.join(OUT_ROOT, "delta_tables.csv"), index=False)
    print(f"wrote {path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
