"""Build the summary tables from every metrics.json under a results folder.

Reads   <result_root>/<family>/<task>/<row>/metrics.json
Writes  <result_root>/<family>/<family>_<metric>.csv
        rows = models and baselines, columns = tasks, cells = "mean (std)"

    outcome:    outcome_auroc.csv, outcome_auprc.csv
    phenotype:  phenotype_auroc.csv, phenotype_auprc.csv, phenotype_cindex.csv
    regression: regression_r2.csv

The tables are rebuilt from scratch each time, so it is safe to run whenever you like.

    python -m femr.evaluation.common.collect --result_root /path/to/result
"""
import argparse
import json
from pathlib import Path

import pandas as pd

FAMILY_METRICS = {"outcome": ["auroc", "auprc"], "phenotype": ["auroc", "auprc", "cindex"], "regression": ["r2"]}
ROW_ORDER = ["ntp", "motor", "tpp", "mtpp", "icarefm", "survivehr"]  # then any other model, then baselines
BASELINES = ["xgboost", "deephit", "most_recent"]


def row_key(name):
    base = name.removesuffix("_mamba")
    if name in BASELINES:
        return (2, BASELINES.index(name), name)
    if base in ROW_ORDER:
        return (0, ROW_ORDER.index(base), name)
    return (1, 0, name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result_root", required=True)
    root = Path(p.parse_args().result_root)

    for family, metrics in FAMILY_METRICS.items():
        cells = {m: {} for m in metrics}  # metric -> (row, task) -> "mean (std)"
        for path in sorted((root / family).glob("*/*/metrics.json")):
            row, task = path.parent.name, path.parent.parent.name
            for metric, value in json.loads(path.read_text()).items():
                if metric in cells:
                    cells[metric][(row, task)] = f"{value['mean']:.3f} ({value['std']:.3f})"
        for metric, values in cells.items():
            if not values:
                continue
            table = pd.Series(values).unstack()
            table = table.loc[sorted(table.index, key=row_key), sorted(table.columns)]
            table.index.name = "model"
            out = root / family / f"{family}_{metric}.csv"
            table.to_csv(out)
            print(f"{out}\n{table.fillna('').to_string()}\n")


if __name__ == "__main__":
    main()
