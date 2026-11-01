"""Regime assignment, shared-index bootstrapping, and table writers."""

import os

import numpy as np
import pandas as pd

from femr.evaluation.common.eval_utils import (
    _format_csv_cell,
    _format_latex_cell_from_csv,
)
from femr.evaluation.analysis.stratified.paths import OUT_ROOT

REGIMES = ["low", "mid", "high"]
N_BOOT = 1000
SEED = 42


def tertile_thresholds(n_events):
    """33rd/67th percentiles of the history-length distribution for one task."""
    c33, c67 = np.quantile(np.asarray(n_events, dtype=float), [1 / 3, 2 / 3])
    return float(c33), float(c67)


def assign_regime(n_events, c33, c67):
    n_events = np.asarray(n_events)
    out = np.full(len(n_events), "mid", dtype=object)
    out[n_events < c33] = "low"
    out[n_events >= c67] = "high"
    return out


def bootstrap_indices(n, n_boot=N_BOOT, seed=SEED):
    """Indices shared by every model in a stratum so deltas are paired."""
    rng = np.random.RandomState(seed)
    return np.array([rng.choice(n, n) for _ in range(n_boot)])


def summarize(samples, point=None):
    samples = np.asarray(samples, dtype=float)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "point": float(point) if point is not None else float(np.mean(samples)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def paired_delta(ora_samples, base_samples):
    """One-sided bootstrap test for H1: ORA > baseline, on shared replicates."""
    delta = np.asarray(ora_samples, dtype=float) - np.asarray(base_samples, dtype=float)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {
        "delta_mean": float(np.mean(delta)),
        "delta_ci_lo": float(lo),
        "delta_ci_hi": float(hi),
        "p_one_sided": float(np.mean(delta <= 0)),
    }, delta


def heterogeneity(delta_high, delta_low):
    """Is ORA's gain different in the high- vs low-history regime?

    The two strata are disjoint, so replicate b of each is an independent draw; their
    difference across replicates approximates the sampling distribution of the gap.
    Replicate counts can differ when a stratum drops degenerate draws, so trim to the
    shorter run -- the pairing across strata is arbitrary either way.
    """
    delta_high = np.asarray(delta_high, dtype=float)
    delta_low = np.asarray(delta_low, dtype=float)
    n = min(len(delta_high), len(delta_low))
    diff = delta_high[:n] - delta_low[:n]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = 2 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0)))
    return {
        "diff_high_minus_low": float(np.mean(diff)),
        "diff_ci_lo": float(lo),
        "diff_ci_hi": float(hi),
        "p_two_sided": min(1.0, p),
    }


def write_wide_table(rows, metric_name, ndigits=3):
    """rows: records with task, regime, model, point, std -> one CSV + LaTeX per metric."""
    df = pd.DataFrame(rows)
    df["cell"] = [
        _format_csv_cell(r.point, r.std, ndigits=ndigits) for r in df.itertuples()
    ]
    df["Task"] = df["task"] + " / " + df["regime"]
    wide = df.pivot(index="Task", columns="model", values="cell")

    regime_order = {r: i for i, r in enumerate(REGIMES + ["all"])}
    task_order = {t: i for i, t in enumerate(dict.fromkeys(df["task"]))}
    wide = wide.loc[
        sorted(wide.index, key=lambda s: (task_order[s.split(" / ")[0]],
                                          regime_order[s.split(" / ")[1]]))
    ]

    os.makedirs(OUT_ROOT, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT, f"{metric_name}_by_regime.csv")
    tex_path = os.path.join(OUT_ROOT, f"{metric_name}_by_regime.tex")
    wide.to_csv(csv_path)
    tex = wide.map(_format_latex_cell_from_csv).to_latex(
        index=True, escape=False, na_rep="",
        column_format="l" + "c" * len(wide.columns),
    )
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {csv_path} and {tex_path}")
    return wide


def to_markdown(df, index=False):
    """Minimal markdown renderer (avoids a tabulate dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    return "\n".join(
        ["| " + " | ".join(cols) + " |",
         "|" + "|".join("---" for _ in cols) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


SAMPLE_DIR = os.path.join(OUT_ROOT, "bootstrap_samples")


def save_samples(family, task, regime, metric, samples):
    """Persist each model's per-replicate metric values for downstream aggregate tests."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    path = os.path.join(SAMPLE_DIR, f"{family}__{task}__{regime}.npz")
    np.savez_compressed(path, metric=np.array(metric), **{m: np.asarray(v, dtype=float)
                                                          for m, v in samples.items()})


def load_samples(family, task, regime):
    path = os.path.join(SAMPLE_DIR, f"{family}__{task}__{regime}.npz")
    data = np.load(path)
    metric = str(data["metric"])
    return metric, {k: data[k] for k in data.files if k != "metric"}


def append_csv(records, name):
    if not records:
        return
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, name)
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"wrote {path}")
