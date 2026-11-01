"""Test whether ORA's gain over each baseline is uniform across history-length regimes.

A non-significant difference is not evidence of uniformity, so this runs three layers:

1. Homogeneity (Cochran's Q) per task x baseline -- H0: gain is the same in all three
   regimes. The three strata are disjoint, so the estimates are independent and Q ~ chi2_2.
   I^2 reports the share of the observed spread that exceeds sampling noise.
2. Linear trend per task x baseline -- the directional alternative the reviewer raises,
   L = gain(high) - gain(low), with a bootstrap two-sided p.
3. Pooled trend per task-type x baseline, with an equivalence bound. delta_star is the
   tightest margin for which the trend is bounded inside +/- delta_star, turning "we did
   not detect a trend" into "any trend is smaller than delta_star".

    python uniformity.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from femr.evaluation.analysis.stratified.common import (
    REGIMES,
    load_samples,
    to_markdown,
)
from femr.evaluation.analysis.stratified.paths import (
    BASELINES,
    ORA,
    OUT_ROOT,
    PHENOTYPE_TASKS,
    OUTCOME_TASKS,
    REGRESSION_TASKS,
)

# task type -> [(family, task), ...]
TASK_TYPES = {
    "classification": ([("outcome", t) for t in OUTCOME_TASKS]
                       + [("phenotype_binary", t) for t in PHENOTYPE_TASKS]),
    "tte": [("phenotype_tte", t) for t in PHENOTYPE_TASKS],
    "regression": [("regression", t) for t in REGRESSION_TASKS],
}
TYPE_METRIC = {"classification": "AUROC", "tte": "C-index", "regression": "R^2"}


def delta_samples(family, task, regime, baseline):
    """Per-replicate ORA-minus-baseline values for one stratum."""
    _, samples = load_samples(family, task, regime)
    return samples[ORA] - samples[baseline]


def collect():
    """deltas[(type, family, task, baseline)][regime] -> bootstrap replicate array."""
    out = {}
    for ttype, cohorts in TASK_TYPES.items():
        for family, task in cohorts:
            for base in BASELINES:
                out[(ttype, family, task, base)] = {
                    r: delta_samples(family, task, r, base) for r in REGIMES
                }
    return out


def cochran_q(point, se):
    """Homogeneity of independent stratum estimates. Returns Q, p, I^2, pooled mean."""
    w = 1.0 / np.square(se)
    pooled = float((w * point).sum() / w.sum())
    q = float((w * np.square(point - pooled)).sum())
    df = len(point) - 1
    p = float(stats.chi2.sf(q, df))
    i2 = float(max(0.0, (q - df) / q)) if q > 0 else 0.0
    return q, p, i2, pooled


def trend(delta_by_regime):
    """L = gain(high) - gain(low) from paired replicates, plus a two-sided bootstrap p."""
    hi, lo = delta_by_regime["high"], delta_by_regime["low"]
    n = min(len(hi), len(lo))
    diff = hi[:n] - lo[:n]
    p = 2 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
    return diff, min(1.0, p)


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    n = len(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate(
        (p[order] * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    return np.clip(q, 0, 1)


def main():
    deltas = collect()

    # ---- Layers 1 and 2: per task x baseline -------------------------------------
    per_task = []
    for (ttype, family, task, base), d in deltas.items():
        point = np.array([d[r].mean() for r in REGIMES])
        se = np.array([d[r].std(ddof=1) for r in REGIMES])
        q, q_p, i2, pooled = cochran_q(point, se)
        diff, t_p = trend(d)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        per_task.append({
            "task_type": ttype, "family": family, "task": task, "baseline": base,
            **{f"gain_{r}": point[i] for i, r in enumerate(REGIMES)},
            **{f"se_{r}": se[i] for i, r in enumerate(REGIMES)},
            "pooled_gain": pooled, "Q": q, "Q_df": 2, "Q_p": q_p, "I2": i2,
            "trend_high_minus_low": float(diff.mean()),
            "trend_ci_lo": float(lo), "trend_ci_hi": float(hi), "trend_p": t_p,
        })
    per_task = pd.DataFrame(per_task)
    per_task["Q_q_bh"] = bh(per_task["Q_p"])
    per_task["trend_q_bh"] = bh(per_task["trend_p"])
    per_task.to_csv(os.path.join(OUT_ROOT, "uniformity_per_task.csv"), index=False)

    # ---- Layer 3: pooled trend per task type, with an equivalence bound -----------
    pooled_rows = []
    for ttype, cohorts in TASK_TYPES.items():
        for base in list(BASELINES) + ["ALL_BASELINES"]:
            bases = BASELINES if base == "ALL_BASELINES" else [base]
            stacks = []
            for family, task in cohorts:
                for b in bases:
                    diff, _ = trend(deltas[(ttype, family, task, b)])
                    stacks.append(diff)
            n = min(len(s) for s in stacks)
            pooled = np.mean([s[:n] for s in stacks], axis=0)
            lo95, hi95 = np.percentile(pooled, [2.5, 97.5])
            lo90, hi90 = np.percentile(pooled, [5, 95])
            p = 2 * min(float((pooled <= 0).mean()), float((pooled >= 0).mean()))
            pooled_rows.append({
                "task_type": ttype, "metric": TYPE_METRIC[ttype], "baseline": base,
                "n_cohorts": len(cohorts), "n_contrasts": len(stacks),
                "pooled_trend": float(pooled.mean()),
                "ci95_lo": float(lo95), "ci95_hi": float(hi95),
                "p_two_sided": min(1.0, p),
                # tightest margin at which the trend is bounded inside +/- delta_star
                "delta_star": float(max(abs(lo90), abs(hi90))),
            })
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(os.path.join(OUT_ROOT, "uniformity_pooled_trend.csv"), index=False)

    # ---- Report ------------------------------------------------------------------
    lines = []
    w = lines.append
    w("# Is ORA's gain uniform across history-length regimes?\n")
    w("Three layers, because a non-significant contrast is not evidence of uniformity.\n")

    w("## 1. Homogeneity (Cochran's Q, H0: gain identical in all three regimes)\n")
    w("The three strata are disjoint, so the stratum gains are independent estimates of "
      "the same quantity under H0 and Q ~ chi2 with 2 df. `I2` is the share of the "
      "observed spread not explained by sampling noise.\n")
    cols = ["task_type", "family", "task", "baseline", "gain_low", "gain_mid",
            "gain_high", "Q", "Q_p", "Q_q_bh", "I2"]
    w(to_markdown(per_task[cols].round(4)))
    w("")
    sig_q = int((per_task.Q_q_bh < 0.05).sum())
    w(f"Rejected at q < 0.05 (BH over {len(per_task)} tests): {sig_q}/{len(per_task)}.\n")

    w("## 2. Linear trend (H0: gain(high) = gain(low))\n")
    cols = ["task_type", "family", "task", "baseline", "trend_high_minus_low",
            "trend_ci_lo", "trend_ci_hi", "trend_p", "trend_q_bh"]
    w(to_markdown(per_task[cols].round(4)))
    w("")
    sig_t = int((per_task.trend_q_bh < 0.05).sum())
    up = int(((per_task.trend_q_bh < 0.05) & (per_task.trend_high_minus_low > 0)).sum())
    w(f"Rejected at q < 0.05: {sig_t}/{len(per_task)} ({up} favour high-history, "
      f"{sig_t - up} favour low-history).\n")

    w("## 3. Pooled trend per task type, with equivalence bound\n")
    w("`pooled_trend` averages gain(high) - gain(low) over every cohort in the task type. "
      "`delta_star` is the tightest margin for which the pooled trend is bounded inside "
      "+/- delta_star at 95% confidence -- the equivalence statement the reviewer's "
      "question calls for.\n")
    w(to_markdown(pooled_df.round(4)))
    w("")

    w("## Reading\n")
    for ttype in TASK_TYPES:
        row = pooled_df[(pooled_df.task_type == ttype)
                        & (pooled_df.baseline == "ALL_BASELINES")].iloc[0]
        w(f"- **{ttype}** ({TYPE_METRIC[ttype]}): pooled trend "
          f"{row.pooled_trend:+.4f} [{row.ci95_lo:+.4f}, {row.ci95_hi:+.4f}], "
          f"p = {row.p_two_sided:.3f}. Any systematic change in ORA's advantage between "
          f"the low- and high-history regimes is smaller than {row.delta_star:.4f} "
          f"{TYPE_METRIC[ttype]}.")

    path = os.path.join(OUT_ROOT, "uniformity.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(pooled_df[pooled_df.baseline == "ALL_BASELINES"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
