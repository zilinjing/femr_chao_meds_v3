import pandas as pd
import numpy as np
from pathlib import Path

regression_base = Path("/shared/share_mala/zj2398/mimic/task/regression")
task = "creatinine"  # change as needed

# ── Load the full parquet (150k rows) ──
full_df = pd.read_parquet(regression_base / "FULL_FILE_PATH_HERE.parquet")  # TODO: set path

# ── Load the two model prediction files ──
mtpp = pd.read_parquet(regression_base / "transformer_mtpp/results" / task / "transformer_mtpp" / "test_predictions_50k.parquet")
motor = pd.read_parquet(regression_base / "motor/results" / task / "motor" / "test_predictions_50k.parquet")

KEY_COLS = ["subject_id", "prediction_time"]

for name, pred_df in [("mtpp", mtpp), ("motor", motor)]:
    merged = pred_df.merge(full_df, on=KEY_COLS, how="left", indicator=True)

    # Check 1: every row in pred_df has a match in full_df
    missing = merged[merged["_merge"] == "left_only"]
    if len(missing) > 0:
        print(f"⚠ {name}: {len(missing)} rows NOT found in full_df:")
        print(missing[KEY_COLS].head(10).to_string(index=False))
    else:
        print(f"✓ {name}: all {len(pred_df)} rows found in full_df")

    # Check 2: y_true == numerical_value where matched
    matched = merged[merged["_merge"] == "both"]
    diff = np.abs(matched["y_true"].values - matched["numerical_value"].values)
    max_diff = np.max(diff) if len(diff) > 0 else 0
    n_mismatch = np.sum(diff > 1e-3)
    print(f"  max |y_true - numerical_value|: {max_diff:.2e}, rows with diff>1e-3: {n_mismatch}")
    if n_mismatch > 0:
        bad = matched[diff > 1e-3][KEY_COLS + ["y_true", "numerical_value"]].head(10)
        print(bad.to_string(index=False))
    print()
