import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _bootstrap_regression_metrics(y_true, y_pred, n_boot=100, seed=0):
    """
    Returns dict:
      {"rmse": (mean,std), "mae": (mean,std), "r2": (mean,std)}
    std is bootstrap std (ddof=1). If not enough samples, returns (None,None).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return {"rmse": (None, None), "mae": (None, None), "r2": (None, None)}

    rng = np.random.default_rng(seed)
    rmses, maes, r2s = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        yt = y_true[idx]
        yp = y_pred[idx]

        rmse_b = float(np.sqrt(mean_squared_error(yt, yp)))
        mae_b = float(mean_absolute_error(yt, yp))
        r2_b = float(r2_score(yt, yp))

        rmses.append(rmse_b)
        maes.append(mae_b)
        r2s.append(r2_b)

    def _mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return (mean, std)

    return {"rmse": _mean_std(rmses), "mae": _mean_std(maes), "r2": _mean_std(r2s)}

def _metric_mean_std(metrics: dict, key: str):
    """
    Returns (mean, std) if present.
    - If metrics[key] is (mean, std): use it
    - If metrics[key] is scalar: treat as (mean, None) (won't break)
    """
    if key not in metrics:
        return (None, None)
    v = metrics[key]
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (float(v[0]), float(v[1]))
    return (float(v), None)

def _format_csv_cell(mean, std, ndigits=3):
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{ndigits}f}"
    return f"{mean:.{ndigits}f} ({std:.{ndigits}f})"

def _format_latex_cell_from_csv(s):
    """
    Convert '0.455 (0.028)' -> '\\shortstack{0.455\\\\(0.028)}'
    If no parentheses found, return as-is.
    """
    # Handle NaN, None, and non-string types (like float)
    if pd.isna(s) or s is None:
        return ""
    # Convert to string if it's not already
    s = str(s).strip()
    if not s:
        return ""
    if "(" in s and s.endswith(")"):
        mean = s.split("(")[0].strip()
        std = "(" + s.split("(", 1)[1]
        # shortstack gives the two-line cell without extra packages
        return r"\shortstack{" + mean + r"\\" + std + "}"
    return s

def _update_metric_table(summary_dir: Path, metric_name: str, task: str, model_col: str, mean: float, std: float):
    """
    Maintain ONE CSV per metric with string cells: 'mean (std)'.
    Also write LaTeX with mean on top and std below using \\shortstack.
    """
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{metric_name}.csv"
    tex_path = summary_dir / f"{metric_name}.tex"

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, dtype=str)
    else:
        df = pd.DataFrame(dtype=str)

    # Ensure index/col exist
    # print(df.index,task)
    # print(df)
    if task not in df.index:
        df.loc[task, :] = ""
    if model_col not in df.columns:
        df[model_col] = ""

    df.loc[task, model_col] = _format_csv_cell(mean, std, ndigits=3)

    # Optional: stable ordering
    df.index.name = "Task"
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)

    # Write CSV
    df.to_csv(csv_path)

    # Write LaTeX (convert each cell to a shortstack)
    df_tex = df.applymap(_format_latex_cell_from_csv)
    latex = df_tex.to_latex(
        index=True,
        escape=False,
        na_rep="",
        column_format="l" + "c" * len(df.columns),  # centered numeric columns like your figure
    )
    with open(tex_path, "w") as f:
        f.write(latex)
