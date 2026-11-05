#!/usr/bin/env bash
# Gather all evaluation results into one mean +/- std table.
#
# Three producers, all bootstrap-based (100 resamples):
#   results_boot.json  meds_evaluation  -> mean_<metric> / std_<metric>   (classification)
#   metrics.json       finetune_motor   -> auroc, auprc, C-Index, Int-Brier as (mean, std)
#   metrics.json       finetune_regression -> r2, rmse, mae as (mean, std)
#
# Usage: bash collect_eval_results.sh [RESULTS_ROOT]
set -uo pipefail

ROOT="${1:-/data/zilinjing/ethos_fm/storage/ora/result/mtpp_factorized}"
VENV="${VENV:-/data/zilinjing/ethos_fm/envs/tte}"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || PY=python3

"$PY" - "$ROOT" <<'PY'
import json, sys, csv
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    sys.exit(f"no results at {root}")

FAMILY = {"outcome": "classification", "phenotype": "classification/TTE", "regression": "regression"}

def family_of(path):
    for part in path.parts:
        if part in FAMILY:
            return FAMILY[part]
    return "?"

def pair(v):
    """metrics.json stores either a scalar or a [mean, std] pair."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]), float(v[1])
    if isinstance(v, (int, float)):
        return float(v), None
    return None, None

rows = []

# meds_evaluation bootstrap output (classification only)
for path in sorted(root.rglob("results_boot.json")):
    data = json.loads(path.read_text())
    task = path.parts[-3]
    for metric, label in [("roc_auc_score", "AUROC"), ("average_precision_score", "AUPRC"),
                          ("brier_score", "Brier")]:
        if f"mean_{metric}" in data:
            rows.append(dict(task=task, family=family_of(path), metric=label,
                             mean=data[f"mean_{metric}"], std=data.get(f"std_{metric}"),
                             source="meds_evaluation"))

# finetune_motor / finetune_regression
for path in sorted(root.rglob("metrics.json")):
    data = json.loads(path.read_text())
    task = path.parts[-3]
    for key, label in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("C-Index", "C-index"),
                       ("Int-Brier", "IntBrier"), ("r2", "R2"), ("rmse", "RMSE"), ("mae", "MAE")]:
        if key in data:
            m, s = pair(data[key])
            if m is not None:
                rows.append(dict(task=task, family=family_of(path), metric=label,
                                 mean=m, std=s, source="probe"))

# Regression never writes metrics.json (the json.dump is commented out in
# finetune_regression.py); it only maintains summary_tables/<metric>.csv with 'mean (std)' cells.
import re
CELL = re.compile(r"([-\d.]+)\s*\(([-\d.]+)\)")
for csv_path in sorted(root.rglob("summary_tables/*.csv")):
    metric = csv_path.stem.lower()
    label = {"r2": "R2", "rmse": "RMSE", "mae": "MAE"}.get(metric)
    if label is None:
        continue
    with csv_path.open() as fh:
        for rec in csv.DictReader(fh):
            task = (rec.get("Task") or "").strip()
            for col, cell in rec.items():
                if col in ("Task", "") or not cell or not cell.strip():
                    continue
                m = CELL.match(cell.strip())
                if m:
                    rows.append(dict(task=task, family="regression", metric=label,
                                     mean=float(m.group(1)), std=float(m.group(2)),
                                     source="summary_table"))

if not rows:
    sys.exit(f"no results yet under {root}")

# meds_evaluation is the authority for classification AUROC/AUPRC; drop the probe duplicate.
best = {}
for r in rows:
    key = (r["task"], r["metric"])
    if key not in best or (r["source"] == "meds_evaluation" and best[key]["source"] != "meds_evaluation"):
        best[key] = r
rows = sorted(best.values(), key=lambda r: (r["family"], r["task"], r["metric"]))

fam = None
print(f"\n{root}\n")
for r in rows:
    if r["family"] != fam:
        fam = r["family"]
        print(f"\n=== {fam} ===")
    std = f" ({r['std']:.3f})" if r["std"] is not None else ""
    print(f"  {r['task']:<24} {r['metric']:<9} {r['mean']:.3f}{std}   [{r['source']}]")

out = root / "all_metrics.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["family", "task", "metric", "mean", "std", "source"])
    w.writeheader()
    w.writerows(rows)
print(f"\n{len(rows)} metrics -> {out}")

# Family averages, the headline numbers for the ablation figure.
print("\n=== averages across tasks ===")
from statistics import mean as _mean
for fam_name in ["classification", "classification/TTE", "regression"]:
    for metric in ["AUROC", "C-index", "R2"]:
        vals = [r["mean"] for r in rows if r["family"] == fam_name and r["metric"] == metric]
        if vals:
            print(f"  {fam_name:<20} {metric:<8} {_mean(vals):.3f}  (n={len(vals)})")
# Classification spans both outcome and phenotype in the paper's tables.
auroc = [r["mean"] for r in rows if r["metric"] == "AUROC"]
if auroc:
    print(f"  {'ALL classification':<20} {'AUROC':<8} {_mean(auroc):.3f}  (n={len(auroc)})")
PY
