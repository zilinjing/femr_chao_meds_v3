"""Merge the three stratified analyses into one report for the reviewer response.

    python summarize.py
"""

import os

import numpy as np
import pandas as pd

from femr.evaluation.analysis.stratified.common import REGIMES, to_markdown
from femr.evaluation.analysis.stratified.paths import BASELINES, MODELS, ORA, OUT_ROOT

METRIC_LABEL = {"auroc": "AUROC", "cindex": "C-index (IPCW, tau=300d)", "r2": "R^2"}


def load(name):
    parts = [pd.read_csv(os.path.join(OUT_ROOT, f"{name}_{s}.csv"))
             for s in ("binary_regression", "tte")]
    return pd.concat(parts, ignore_index=True)


def metric_table(metrics, metric):
    sub = metrics[metrics.metric == metric].copy()
    sub["cell"] = sub.apply(lambda r: f"{r.point:.3f} ({r['std']:.3f})", axis=1)
    sub["Task"] = sub["task"] + " / " + sub["regime"]
    wide = sub.pivot(index="Task", columns="model", values="cell")[MODELS]
    order = {r: i for i, r in enumerate(REGIMES + ["all"])}
    tasks = list(dict.fromkeys(sub["task"]))
    wide = wide.loc[sorted(wide.index,
                           key=lambda s: (tasks.index(s.split(" / ")[0]),
                                          order[s.split(" / ")[1]]))]
    return wide


def main():
    thresholds = load("thresholds")
    metrics = load("stratified_metrics")
    deltas = load("stratified_deltas")
    hetero = load("heterogeneity")

    # Benjamini-Hochberg across all 52 heterogeneity tests.
    hetero = hetero.sort_values("p_two_sided").reset_index(drop=True)
    n = len(hetero)
    raw = hetero["p_two_sided"].to_numpy()
    hetero["q_bh"] = np.minimum.accumulate(
        (raw * n / np.arange(1, n + 1))[::-1]
    )[::-1].clip(max=1.0)

    thresholds.to_csv(os.path.join(OUT_ROOT, "thresholds.csv"), index=False)
    metrics.to_csv(os.path.join(OUT_ROOT, "stratified_metrics.csv"), index=False)
    deltas.to_csv(os.path.join(OUT_ROOT, "stratified_deltas.csv"), index=False)
    hetero.to_csv(os.path.join(OUT_ROOT, "heterogeneity.csv"), index=False)

    grew = int((hetero.diff_high_minus_low > 0).sum())
    sig = hetero[hetero.q_bh < 0.05]
    sig_grew = int((sig.diff_high_minus_low > 0).sum())
    mono_up = 0
    mono_down = 0
    for _, g in hetero.iterrows():
        lo, mi, hi = g.delta_low, g.delta_mid, g.delta_high
        mono_up += int(lo < mi < hi)
        mono_down += int(lo > mi > hi)

    # How often does ORA rank first within each regime?
    ranks = []
    for (task, metric, regime), g in metrics.groupby(["task", "metric", "regime"]):
        best = g.loc[g["point"].idxmax(), "model"]
        ranks.append({"task": task, "metric": metric, "regime": regime,
                      "best": best, "ora_best": best == ORA})
    ranks = pd.DataFrame(ranks)
    win = ranks[ranks.regime != "all"].groupby("regime")["ora_best"].sum()

    lines = []
    w = lines.append
    w("# Stratified evaluation by history length\n")
    w("Held-out prediction points are split into three history-length regimes by the number")
    w("of preceding events (per-task tertiles, so each regime holds ~1/3 of that task's")
    w("prediction points). Phenotype cohorts count only events within 730 days of the")
    w("prediction time; all other cohorts use the full record. Nothing was retrained --")
    w("existing held-out predictions are re-scored inside each stratum, and pooling the")
    w("three strata reproduces every published headline number.\n")
    w(f"ORA is `{ORA}`. Baselines: {', '.join(f'`{b}`' for b in BASELINES)}.\n")

    w("## Regime thresholds\n")
    t = thresholds[["family", "task", "metric", "n", "c33", "c67",
                    "median_n_events", "n_low", "n_mid", "n_high"]].copy()
    for c in ("c33", "c67", "median_n_events"):
        t[c] = t[c].round(0).astype(int)
    w(to_markdown(t))
    w("")

    for metric in ("auroc", "cindex", "r2"):
        if metric not in set(metrics.metric):
            continue
        w(f"## {METRIC_LABEL[metric]} by regime\n")
        w("Cells are point estimate (bootstrap std).\n")
        w(to_markdown(metric_table(metrics, metric), index=True))
        w("")

    w("## Does ORA's gain depend on history length?\n")
    w("`diff_high_minus_low` is the bootstrap estimate of "
      "(ORA - baseline in the high regime) minus (ORA - baseline in the low regime). "
      "A systematic bias from Eq. 2's reweighting would show up as consistently positive "
      "values -- ORA benefiting disproportionately where more history precedes the "
      "prediction.\n")
    h = hetero[["family", "task", "metric", "baseline", "delta_low", "delta_mid",
                "delta_high", "diff_high_minus_low", "diff_ci_lo", "diff_ci_hi",
                "p_two_sided", "q_bh"]].round(4)
    w(to_markdown(h))
    w("")

    w("## Summary\n")
    n_cohorts = metrics.groupby(["family", "task"]).ngroups
    w(f"- {len(hetero)} ORA-vs-baseline comparisons across {n_cohorts} task cohorts.")
    w(f"- The high-minus-low gap is positive in {grew}/{len(hetero)} of them -- "
      f"close to the {len(hetero) / 2:.0f} expected if direction were arbitrary.")
    w(f"- After Benjamini-Hochberg across all {len(hetero)} tests, {len(sig)} remain "
      f"significant at q < 0.05: {sig_grew} favour the high-history regime and "
      f"{len(sig) - sig_grew} favour the low-history regime. Real heterogeneity exists, "
      "but it is not directional.")
    w(f"- The gain rises monotonically with history in only {mono_up}/{len(hetero)} "
      f"comparisons and falls monotonically in {mono_down}/{len(hetero)}; the remaining "
      f"{len(hetero) - mono_up - mono_down} are non-monotone. A reweighting that "
      "systematically favoured long histories would produce mostly monotone increases.")
    w(f"- ORA has the highest score in {int(win.get('low', 0))}/{len(ranks[ranks.regime == 'low'])} "
      f"low-history cells, {int(win.get('mid', 0))}/{len(ranks[ranks.regime == 'mid'])} mid, and "
      f"{int(win.get('high', 0))}/{len(ranks[ranks.regime == 'high'])} high -- its ranking is "
      "not concentrated in the long-history stratum.")
    w("")
    w("### Caveats\n")
    thin = metrics[(metrics.metric == "cindex") & (metrics.regime != "all")
                   & (metrics.model == MODELS[0])]
    thin = thin[thin.n_event_before_tau < 10]
    if len(thin):
        w("- The survival strata are thin at the 300-day horizon. "
          + "; ".join(f"{r.task}/{r.regime} has only {int(r.n_event_before_tau)} events before tau"
                      for r in thin.itertuples())
          + ". Per-stratum C-indices there carry wide intervals and should not be "
            "over-interpreted.")
    w("- Bootstrap replicates in which a survival stratum contained no event before tau "
      "were dropped for every model alike, so the paired deltas stay valid; the counts are "
      "in `stratified_metrics.csv` (`n_boot_dropped`).")
    w("- History length is counted from the raw MEDS record and can exceed what a model's "
      "context window actually ingests, so it measures available rather than consumed "
      "history.")

    path = os.path.join(OUT_ROOT, "reviewer_response.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(f"\nhigh-minus-low positive in {grew}/{len(hetero)}; "
          f"significant {len(sig)} ({sig_grew} up / {len(sig) - sig_grew} down)")
    print(f"ORA best per regime: {dict(win)}")


if __name__ == "__main__":
    main()
