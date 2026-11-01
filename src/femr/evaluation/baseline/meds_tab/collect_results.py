"""Build the final results table: AUROC and C-index as mean +- std.

Reads the metric pickles written by run_all.py and reports:

* **Binary tasks** -- AUROC from `meds_evaluation.evaluate_bootstrapped_binary_classification`
  (`mean_roc_auc_score` +- `std_roc_auc_score`, 100 bootstraps). The legacy plain row
  bootstrap is shown alongside for continuity with the submitted numbers.
* **Time-to-event tasks** -- C-index (`concordance_index_ipcw`) and integrated Brier
  score, mean +- std over 100 bootstraps.

IMPORTANT: meds_evaluation implements **binary classification only** -- there is no
survival entry point in the package, so the C-index cannot come from it. If the
foundation-model arms are re-scored with meds_evaluation for AUROC, their C-index must
keep using the same `concordance_index_ipcw` bootstrap used here, or the TTE error bars
across arms will not be comparable.

Usage::

    python collect_results.py
    python collect_results.py --results_dir /path/to/results --csv results.csv
"""

import argparse
import glob
import os
import pickle
from typing import Optional

import numpy as np

DEFAULT_RESULTS = "/data/zilinjing/ethos_fm/storage/ora/result/meds_tab"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=DEFAULT_RESULTS)
    p.add_argument("--csv", default=None)
    return p.parse_args()


def ms(values) -> tuple:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std())


def fmt(mean: Optional[float], std: Optional[float], nd: int = 3) -> str:
    if mean is None:
        return "-"
    return f"{mean:.{nd}f} ({std:.{nd}f})"


def main() -> None:
    args = parse_args()
    rows = []

    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.pickle"))):
        base = os.path.basename(path)[: -len(".pickle")]
        if base.endswith("_xgboost"):
            task, model, family = base[: -len("_xgboost")], "XGBoost", "Binary"
        elif base.endswith("_deephit"):
            task, model, family = base[: -len("_deephit")], "DeepHit", "TTE"
        else:
            continue

        with open(path, "rb") as f:
            m = pickle.load(f)

        row = {"task": task, "family": family, "model": model,
               "primary": None, "primary_std": None, "primary_source": None,
               "secondary": None, "secondary_std": None, "secondary_name": None,
               "legacy": None, "legacy_std": None, "best_params": None,
               "n_saved": None, "saved_path": None}

        meds = m.get("_meds_evaluation")
        if meds and "mean_roc_auc_score" in meds:
            row["primary"] = meds["mean_roc_auc_score"]
            row["primary_std"] = meds["std_roc_auc_score"]
            row["primary_source"] = "meds_evaluation"
            if "mean_average_precision_score" in meds:
                row["secondary"] = meds["mean_average_precision_score"]
                row["secondary_std"] = meds["std_average_precision_score"]
                row["secondary_name"] = "AUPRC"
        if "AUROC" in m:
            row["legacy"], row["legacy_std"] = ms(m["AUROC"])
            if row["primary"] is None:
                row["primary"], row["primary_std"] = row["legacy"], row["legacy_std"]
                row["primary_source"] = "row bootstrap"
        if "C-Index" in m:
            row["primary"], row["primary_std"] = ms(m["C-Index"])
            row["primary_source"] = "concordance_index_ipcw"
            if "Int-Brier" in m:
                row["secondary"], row["secondary_std"] = ms(m["Int-Brier"])
                row["secondary_name"] = "Int-Brier"

        bp = m.get("_best_params")
        if bp:
            row["best_params"] = bp[0]
        sb = m.get("_saved_bootstrap") or {}
        row["n_saved"] = sb.get("n_save")
        row["saved_path"] = sb.get("path")
        rows.append(row)

    if not rows:
        print(f"No result pickles under {args.results_dir} -- run run_all.py first.")
        return

    rows.sort(key=lambda r: (r["family"], r["task"]))

    print("\n### Binary classification -- AUROC, mean (std)\n")
    print("| Task | AUROC (meds_evaluation) | AUPRC | AUROC (row bootstrap) | Selected config |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["family"] != "Binary":
            continue
        print(f"| {r['task']} | {fmt(r['primary'], r['primary_std'])} | "
              f"{fmt(r['secondary'], r['secondary_std'])} | "
              f"{fmt(r['legacy'], r['legacy_std'])} | `{r['best_params']}` |")

    print("\n### Time-to-event -- C-index, mean (std)\n")
    print("| Task | C-index (ipcw) | Int-Brier | Selected config |")
    print("|---|---|---|---|")
    for r in rows:
        if r["family"] != "TTE":
            continue
        print(f"| {r['task']} | {fmt(r['primary'], r['primary_std'])} | "
              f"{fmt(r['secondary'], r['secondary_std'])} | `{r['best_params']}` |")

    linear = [r["task"] for r in rows
              if r["family"] == "TTE" and (r["best_params"] or {}).get("layers") == []]
    if linear:
        print(f"\nDeepHit selected a **linear** head (`layers=[]`) for: {', '.join(linear)} -- "
              "identical in form to the foundation-model probe head, so the "
              "'should the baseline get hidden layers?' question is moot for those tasks.")

    print("\n### Saved resamples for paired testing\n")
    print("| Task | Family | n saved | Path |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['task']} | {r['family']} | {r['n_saved'] or '-'} | `{r['saved_path'] or '-'}` |")
    print("\nPair against the FM arms with `p_test.tte.utils.paired_bootstrap_test`; it asserts "
          "`test_order` equality, and `generate_bootstrap_indices(n, 10000, seed=42)` is "
          "deterministic in n, so the indices already match.")

    sources = {r["primary_source"] for r in rows if r["primary_source"]}
    print(f"\nMetric sources present: {sorted(sources)}")
    print("Report the same bootstrap procedure for every arm, or the error bars are not comparable.")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
