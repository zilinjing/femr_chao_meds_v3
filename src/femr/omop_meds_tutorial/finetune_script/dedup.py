"""
Post-hoc join_labels matching for motor fine-tuning raw output.

Usage:
    python fix_prediction_times.py --raw_path <raw.parquet> --cohort_path <held_out.parquet> --output_path <output.parquet>
"""
import argparse
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", required=True)
    parser.add_argument("--cohort_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    raw = pl.read_parquet(args.raw_path)
    print(f"Raw: {len(raw)} rows")

    test = pl.read_parquet(args.cohort_path, columns=["subject_id", "prediction_time", "boolean_value"])
    test = test.sort(["subject_id", "prediction_time"])
    print(f"Test: {len(test)} rows")

    raw = raw.with_columns(pl.col("prediction_time").cast(pl.Int64).alias("time_ns"))
    test = test.with_columns(pl.col("prediction_time").cast(pl.Datetime("ns")).cast(pl.Int64).alias("time_ns"))

    raw_sids = raw["subject_id"].to_numpy()
    raw_times = raw["time_ns"].to_numpy()
    raw_probs = raw["predicted_boolean_probability"].to_numpy()
    test_sids = test["subject_id"].to_numpy()
    test_times = test["time_ns"].to_numpy()

    order = np.lexsort((raw_times, raw_sids))

    feature_index = 0
    indices = []
    for i in range(len(test_sids)):
        sid = test_sids[i]
        pt = test_times[i]

        while (feature_index + 1) < len(order):
            next_sid = raw_sids[order[feature_index + 1]]
            next_time = raw_times[order[feature_index + 1]]
            if (next_sid, next_time) <= (sid, pt):
                feature_index += 1
            else:
                break

        is_valid = (
            feature_index < len(order)
            and raw_sids[order[feature_index]] == sid
            and raw_times[order[feature_index]] <= pt
        )

        indices.append(order[feature_index] if is_valid else -1)

    matched = sum(1 for i in indices if i >= 0)
    print(f"Matched: {matched}/{len(test_sids)}, Missing: {len(test_sids) - matched}")

    valid_mask = [i >= 0 for i in indices]
    valid_indices = [i for i in indices if i >= 0]

    result = pl.DataFrame({
        "subject_id": test.filter(pl.Series(valid_mask))["subject_id"],
        "prediction_time": test.filter(pl.Series(valid_mask))["prediction_time"],
        "predicted_boolean_probability": [float(raw_probs[i]) for i in valid_indices],
        "predicted_boolean_value": [bool(raw_probs[i] >= 0.5) for i in valid_indices],
        "boolean_value": test.filter(pl.Series(valid_mask))["boolean_value"],
    })

    auroc = roc_auc_score(result["boolean_value"].to_numpy(), result["predicted_boolean_probability"].to_numpy())
    print(f"Result: {len(result)} rows, AUROC: {auroc:.4f}")

    result.write_parquet(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()