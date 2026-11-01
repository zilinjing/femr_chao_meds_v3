"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""
import json
import meds_reader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from pathlib import Path
import pandas as pd
import polars as pl
import femr.featurizers
import pickle
import pathlib
import os
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import femr.splits
import argparse
from femr.evaluation.generate_features import get_model_name
from sklearn.linear_model import LinearRegression
from femr.evaluation.eval_utils import _metric_mean_std, _update_metric_table

from femr.evaluation.generate_labels import (  # type: ignore
        create_omop_meds_tutorial_arg_parser,
        LABEL_NAMES,
    )

def create_arg_parser():
    # args = create_omop_meds_tutorial_arg_parser()
    args = argparse.ArgumentParser(description="Arguments for linear probing")

    args.add_argument(
        "--model_list",
        default=None,
        help="model list",
    )

    args.add_argument(
        "--result_dir",
        default=None,
        help="result dir",
    )

    args.add_argument(
        "--label_dir",
        default=None,
        help="label dir",
    )
    return args

# ["motor","mamba_ntp","mamba_tpp",""]
# /shared/share_mala/zj2398/mimic/task/regression/ mamba_ntp/results/bilirubin/mamba_ntp/features_with_label
def main():
    args = create_arg_parser().parse_args()
    model_list = args.model_list.split(",")
    print(model_list)
    summary_dir = Path(args.result_dir)/"summary_tables"
    # main_split_path = Path("/shared/share_mala/zj2398/models/motor_8k/main_split.csv")
    main_split_path = Path("/data2/processed_datasets/zj2398/femr/cumc/mtpp/main_split.csv")
    main_split = pd.read_csv(main_split_path)
    train_subject_ids = main_split[main_split["split_name"] == "train"]["subject_id"]
    test_subject_ids = main_split[main_split["split_name"] == "test"]["subject_id"]
    for label_name in ["bilirubin","creatinine","spo2","platelets"]:
        labels = pd.read_parquet(Path(args.label_dir)/label_name/f"{label_name}.parquet")
        for model_name in model_list:
            cols = ["subject_id", "prediction_time", "numerical_value"]
            output_dir = Path(args.result_dir)/model_name/"results"/label_name/model_name
            features_label_data = output_dir/"features_with_label"
            train_data = pd.read_parquet(features_label_data / "train.parquet").drop_duplicates(subset=cols)
            test_data = pd.read_parquet(features_label_data / "test.parquet").drop_duplicates(subset=cols)
            # print(f"Total labels: {len(labels)}")
            print(f"Original train features labels: {len(train_data['features'])}")
            print(f"Original test features labels: {len(test_data['features'])}")

            train_data = train_data.merge(labels[cols].drop_duplicates(),   # avoid duplicate matches in B
                on=cols,
                how="inner"
            )
            test_data = test_data.merge(labels[cols].drop_duplicates(),   # avoid duplicate matches in B
                on=cols,
                how="inner"
            )
            print(f"New train features labels: {len(train_data)}")
            print(f"New test features labels: {len(test_data)}")

            # print(f"New train df: {len(train_df)}")
            # print(f"New test df: {len(test_df)}")

            # Prepare dense float32 arrays explicitly to avoid dtype/object issues
            X_train = train_data["features"]
            y_train = train_data["numerical_value"]
            X_test = test_data["features"]

            # if isinstance(X_train, list):
            #     X_train = np.asarray(X_train)
            # if isinstance(X_test, list):
            #     X_test = np.asarray(X_test)
            X_train = np.stack(X_train.values).astype(np.float32)
            X_test = np.stack(X_test.values).astype(np.float32)
            print(X_train.shape)
            X_train = X_train.astype(np.float32, copy=False)
            X_test = X_test.astype(np.float32, copy=False)

            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # potential issue 1 scale back?
            # not ridge... just linear regression

            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", LinearRegression()),
            ])
            model.fit(X_train, y_train)
            y_true = np.asarray(test_data["numerical_value"], dtype=np.float64)
            y_pred = model.predict(X_test)
            # model.fit(train_set['features'].to_list(), train_set["numerical_value"])
            # y_pred = model.predict(test_data['features'])

            # pl.DataFrame({
            #     "subject_id": test_data["subject_id"].tolist(),
            #     "prediction_time": test_data["prediction_time"].tolist(),
            #     "y_true": y_true.tolist(),
            #     "y_pred": y_pred.tolist(),
            #     "residual": (test_data["numerical_value"] - y_pred).tolist(),
            # }).write_parquet(output_dir / "test_predictions_50k.parquet")

            # rmse = float(np.sqrt(mean_squared_error(test_data["numerical_value"], y_pred)))
            # mae = float(mean_absolute_error(test_data["numerical_value"], y_pred))
            # r2 = float(r2_score(test_data["numerical_value"], y_pred))
            # metrics = {
            #     "rmse": rmse,
            #     "mae": mae,
            #     "r2": r2,
            # }

            rng = np.random.default_rng(0)
            rmses, maes, r2s = [], [], []

            n = len(y_true)
            for bi in range(100):
                idx = rng.integers(0, n, size=n)
                yt = y_true[idx]
                yp = y_pred[idx]

                rmses.append(float(np.sqrt(mean_squared_error(yt, yp))))
                maes.append(float(mean_absolute_error(yt, yp)))
                r2s.append(float(r2_score(yt, yp)))

            # with open(test_result_file, "w") as f:
            #     json.dump(metrics, f, indent=4)

            rmse_mean, rmse_std = float(np.mean(rmses)), float(np.std(rmses, ddof=1))
            mae_mean, mae_std   = float(np.mean(maes)),  float(np.std(maes, ddof=1))
            r2_mean, r2_std     = float(np.mean(r2s)),   float(np.std(r2s, ddof=1))

            _update_metric_table(summary_dir, "rmse", label_name,  model_name, rmse_mean, rmse_std)
            _update_metric_table(summary_dir, "mae", label_name,  model_name, mae_mean, mae_std)
            _update_metric_table(summary_dir, "r2", label_name,  model_name, r2_mean, r2_std)

if __name__ == "__main__":
    main()

'''
python rerun_regression.py \
    --result_dir /shared/share_mala/zj2398/mimic/task/regression \
    --model_list transformer_ntp_new,motor \
    --label_dir /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k

python rerun_regression.py \
    --result_dir /data/results/zj2398/femr/cumc/regression \
    --model_list mtpp_shared,tpp \
    --label_dir /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/regression_100k/

    transformer_mtpp,mamba_ntp,mamba_tpp,mamba_mtpp
'''