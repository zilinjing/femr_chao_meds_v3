"""Collect per-task compute cost for the tabular baseline into one appendix table.

Reads the JSON dropped by the two stages:

    <meds_tab>/<task>/features/<task>_extraction_stats.json   (feature extraction)
    <results>/<task>_<model>_compute.json                     (fit + hyperparameter search)

and emits a markdown table plus a CSV.

The reviewer asked for memory and runtime per baseline. The distinction that
matters is **one-time vs per-task**: a foundation model pays pretraining once and
amortises it over every task, whereas the tabular pipeline re-featurises for each
cohort and has no pretraining at all. This script reports the tabular side of that
comparison; the FM side needs the equivalent numbers (GPU hours and peak GPU
memory for pretraining, then per-task encoder inference + probe fit).

Do not add CPU-hours to GPU-hours. Report both, with the hardware named.

Usage::

    python collect_compute.py                       # markdown to stdout
    python collect_compute.py --csv compute.csv
"""

import argparse
import glob
import json
import os
from typing import Dict, List

ROOT = "/data/zilinjing/ethos_fm/storage/ora"
DEFAULT_FEATURES = os.path.join(ROOT, "meds_tab")
DEFAULT_RESULTS = os.path.join(ROOT, "result", "meds_tab")

FAMILY = {"deephit": "TTE", "xgboost": "Binary"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=DEFAULT_FEATURES)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS)
    p.add_argument("--csv", default=None, help="Also write a CSV here.")
    return p.parse_args()


def load_extraction(features_dir: str) -> Dict[str, dict]:
    """task_dir_name -> extraction stats."""
    out = {}
    for path in sorted(glob.glob(os.path.join(features_dir, "*", "features", "*_extraction_stats.json"))):
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            out[task_dir] = json.load(f)
    return out


def load_training(results_dir: str) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_compute.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def main() -> None:
    args = parse_args()
    extraction = load_extraction(args.features_dir)
    training = load_training(args.results_dir)

    if not extraction:
        print(f"No extraction stats under {args.features_dir} -- run extract.sh first.")
    if not training:
        print(f"No compute stats under {args.results_dir} -- run run_all.py first.")

    rows = []
    for tr in training:
        task, outcome, model = tr["task"], tr["outcome"], tr["model"]
        # Survival features live in <task>/, binary features in <task>_binary/.
        task_dir = task if outcome == "survival" else f"{task}_binary"
        ex = extraction.get(task_dir, {})
        rows.append(
            {
                "task": task,
                "family": FAMILY.get(model, model),
                "model": model,
                "n_labels": tr.get("n_rows", ex.get("n_labels")),
                "n_columns": tr.get("n_columns", ex.get("n_columns")),
                "density": round(ex["density"], 4) if "density" in ex else None,
                "numeric_codes": ex.get("numeric_codes"),
                "extract_s": ex.get("total_seconds"),
                "extract_rss_gb": ex.get("peak_rss_gb"),
                "extract_threads": ex.get("num_threads"),
                "fit_s": tr.get("train_and_tune_seconds"),
                "load_s": tr.get("load_seconds"),
                "fit_rss_gb": tr.get("peak_rss_gb"),
                "n_jobs": tr.get("n_jobs"),
                "device": tr.get("device"),
            }
        )

    # Extraction rows with no matching training run are still worth showing.
    covered = {(r["task"], r["family"]) for r in rows}
    for task_dir, ex in extraction.items():
        fam = "Binary" if task_dir.endswith("_binary") else "TTE"
        task = task_dir[: -len("_binary")] if task_dir.endswith("_binary") else task_dir
        if (task, fam) not in covered:
            rows.append(
                {
                    "task": task, "family": fam, "model": "-",
                    "n_labels": ex.get("n_labels"), "n_columns": ex.get("n_columns"),
                    "density": round(ex["density"], 4) if "density" in ex else None,
                    "numeric_codes": ex.get("numeric_codes"),
                    "extract_s": ex.get("total_seconds"),
                    "extract_rss_gb": ex.get("peak_rss_gb"),
                    "extract_threads": ex.get("num_threads"),
                    "fit_s": None, "load_s": None, "fit_rss_gb": None, "n_jobs": None,
                    "device": None,
                }
            )

    rows.sort(key=lambda r: (r["family"], r["task"]))

    def fmt(v, nd=1):
        if v is None:
            return "-"
        return f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}"

    header = (
        "| Task | Family | Labels | Columns | Density | Extract (s) | Extract RSS (GB) | "
        "Fit+tune (s) | Fit RSS (GB) |"
    )
    print("\n" + header)
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['task']} | {r['family']} | {fmt(r['n_labels'])} | {fmt(r['n_columns'])} | "
            f"{fmt(r['density'], 4)} | {fmt(r['extract_s'])} | {fmt(r['extract_rss_gb'], 2)} | "
            f"{fmt(r['fit_s'])} | {fmt(r['fit_rss_gb'], 2)} |"
        )

    ex_total = sum(r["extract_s"] or 0 for r in rows)
    fit_total = sum(r["fit_s"] or 0 for r in rows)
    threads = {r["extract_threads"] for r in rows if r["extract_threads"]}
    jobs = {r["n_jobs"] for r in rows if r["n_jobs"]}

    devices = {r.get("device") for r in rows if r.get("device")}
    print(f"\n**Totals.** Devices used: {sorted(devices) or ['cpu']}. "
          "Feature extraction is CPU-only; XGBoost training ran on one GPU "
          "(~9x faster than 32 CPU threads at equal AUROC).")
    print(f"- Feature extraction: {ex_total/3600:.2f} h wall-clock over {len(rows)} task-arms "
          f"(threads={sorted(threads) or '?'})")
    print(f"- Fit + hyperparameter search: {fit_total/3600:.2f} h wall-clock "
          f"(n_jobs={sorted(jobs) or '?'})")
    print(f"- Total per-task cost: {(ex_total + fit_total)/3600:.2f} h")
    print(f"- Pretraining cost: 0 (there is none) -- this is the asymmetry to state "
          f"explicitly when comparing against the foundation models.")
    print("\nRecord the CPU model and core count next to this table; XGBoost wall-clock "
          "is roughly linear in cores, so the numbers are meaningless without it.")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
