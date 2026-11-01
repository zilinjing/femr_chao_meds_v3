"""
DeepHit survival evaluation on compressed (count-based) features.
Produces results compatible with tte_evaluation for paired bootstrap testing.

Usage:
  python run_survival_compressed.py --phenotypes ami stroke masld
  python deephit.py --truncation 300 --n_bootstrap 10000
"""

import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from pycox.models import DeepHitSingle
import torchtuples as tt

from utils import (
    generate_bootstrap_indices, bootstrap_cindex_parallel,
    make_structured_array, save_survival_results,
)

# =========================
# Reproducibility
# =========================
DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# CLI
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--phenotypes", nargs="*", default=[])
parser.add_argument("--truncation", type=float, default=300)
parser.add_argument("--n_bootstrap", type=int, default=10000)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
parser.add_argument("--n_jobs", type=int, default=32)
args = parser.parse_args()


# =========================
# Data loading (from your existing code)
# =========================
def remove_duplicates(data, labels):
    return (
        data[~data.index.duplicated(keep="last")],
        labels[~labels.index.duplicated(keep="last")],
    )


def load_features_labels(feature_path, label_path, split_path):
    data = pickle.load(open(feature_path, "rb"))
    labels = (
        pd.read_parquet(label_path)
        .set_index(["subject_id", "prediction_time"])
    )

    X = pd.DataFrame(
        data["features"].todense(),
        index=pd.MultiIndex.from_arrays(
            [data["subject_ids"], data["feature_times"]],
            names=["subject_id", "prediction_time"],
        ),
    )

    X = X.loc[labels.index]
    X, labels = remove_duplicates(X, labels)

    split = pd.read_csv(split_path, index_col="subject_id")
    labels = labels.join(split)

    # Match tte_test ordering before deriving train/tuning/hyper splits.
    X = X.sort_index()
    labels = labels.sort_index()

    # Create val/hyper splits from train
    val = labels[labels.split_name == "train"].index.to_series().sample(
        frac=0.2, replace=False, random_state=DEFAULT_SEED
    )
    hyper = val.sample(frac=0.5, replace=False, random_state=DEFAULT_SEED)
    labels.loc[val, "split_name"] = "tuning"
    labels.loc[hyper, "split_name"] = "hyper"

    return X, labels


# =========================
# Survival split + scale
# =========================
def split_and_scale_survival(X, labels, tte_col="time_to_event_days"):
    split = labels["split_name"].values
    mask = {
        "train": split == "train",
        "val": split == "tuning",
        "hyper": split == "hyper",
        "test": split == "test",
    }

    scaler = StandardScaler().fit(X.values[mask["train"]])
    Xs = {k: scaler.transform(X.values[v]) for k, v in mask.items()}

    def get_target(m):
        return labels[tte_col].values[m], labels["tte_label"].values[m].astype(bool)

    # Build per-split dataframes with (subject_id, prediction_time) for sorting
    idx = labels.index  # MultiIndex: (subject_id, prediction_time)

    splits = {}
    for name, m in mask.items():
        df = pd.DataFrame(Xs[name], columns=[f"f{i}" for i in range(Xs[name].shape[1])])
        sub_idx = idx[m]
        df["subject_id"] = sub_idx.get_level_values("subject_id").values
        df["prediction_time"] = sub_idx.get_level_values("prediction_time").values
        t, e = get_target(m)
        df["time_to_event"] = t
        df["tte_label"] = e
        # Sort for deterministic ordering
        df = df.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        splits[name] = df

    return splits


# =========================
# DeepHit training + evaluation
# =========================
def train_and_evaluate_deephit(splits, truncation, model_name,
                                save_path=None, n_bootstrap=10000,
                                seed=DEFAULT_SEED, n_jobs=8):
    set_seed(seed)

    train_df = splits["train"]
    val_df = splits["val"]
    hyper_df = splits["hyper"]
    test_df = splits["test"]

    # Record test order for alignment
    test_order = list(zip(
        test_df["subject_id"].values.tolist(),
        test_df["prediction_time"].astype(str).values.tolist(),
    ))

    feat_cols = [c for c in train_df.columns if c.startswith("f")]

    x_train = train_df[feat_cols].values.astype(np.float32)
    x_val = val_df[feat_cols].values.astype(np.float32)
    x_hyper = hyper_df[feat_cols].values.astype(np.float32)
    x_test = test_df[feat_cols].values.astype(np.float32)

    get_te = lambda df: (df["time_to_event"].values, df["tte_label"].values)

    labtrans = DeepHitSingle.label_transform(100)
    y_train = labtrans.fit_transform(*get_te(train_df))
    y_val = labtrans.transform(*get_te(val_df))
    y_hyper = labtrans.transform(*get_te(hyper_df))

    durations_train, events_train = get_te(train_df)
    durations_test, events_test = get_te(test_df)

    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    # Match the same DeepHit search space used in tte_test.py.
    # grid = ParameterGrid({
    #     "dropout": [0, 0.1],
    #     "lr": [0.001, 0.0001],
    #     "batch": [1024, 2048],
    # })

    grid = ParameterGrid({
        "dropout": [0.1],
        "lr": [0.001],
        "batch": [1024],
    })


    best_model, best_loss, best_params = None, np.inf, None
    for p in grid:
        net = tt.practical.MLPVanilla(
            in_features, [], out_features,
            batch_norm=True, dropout=p["dropout"],
        )
        model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
        model.optimizer.set_lr(p["lr"])
        model.fit(
            x_train, y_train, p["batch"], 1000,
            callbacks=[tt.callbacks.EarlyStopping()],
            val_data=(x_val, y_val),
        )
        loss = model.score_in_batches(x_hyper, y_hyper)["loss"]
        if loss < best_loss:
            best_loss, best_model, best_params = loss, model, p

    print(f"[{model_name}] Best params: {best_params}, NLL: {best_loss:.4f}")

    surv = best_model.predict_surv_df(x_test)
    et_train = make_structured_array(durations_train, events_train)
    et_test = make_structured_array(durations_test, events_test)

    bootstrap_indices = generate_bootstrap_indices(len(et_test), n_bootstrap, seed)
    c_indices = bootstrap_cindex_parallel(
        surv, et_train, et_test, truncation, bootstrap_indices, n_jobs
    )

    results = {
        "C-Index": c_indices,
        "test_order": test_order,
        "model_name": model_name,
        "best_params": best_params,
    }

    mean_val, std_val = np.mean(c_indices), np.std(c_indices)
    print(f"[{model_name}] C-Index: {mean_val:.4f} (±{std_val:.4f})")

    if save_path:
        save_survival_results(results, best_model, save_path, truncation)

    return results, best_model


# =========================
# Main
# =========================
EXPERIMENTS = {
    "mimic": {
        "data_path": "/shared/share_mala/zj2398/mimic/task/phenotype_task/deephit/tte/",
        "tte_col": "time_to_event_days",
        "phenotypes": ["ami", "stroke", "masld"],
    },
}

if __name__ == "__main__":
    for dataset, cfg in EXPERIMENTS.items():
        split_path = f"/user/zj2398/cache/mtpp_8k_T_8_V_10/main_split.csv"
        results_path = f"/shared/share_mala/zj2398/mimic/task/phenotype_task/results/tte"
        os.makedirs(results_path, exist_ok=True)

        phenotypes = args.phenotypes if args.phenotypes else cfg["phenotypes"]

        for name in phenotypes:
            save_dir = os.path.join(results_path, name)
            out_file = os.path.join(save_dir, f"tte_results_{args.truncation}.pkl")
            if os.path.exists(out_file):
                print(f"Skipping {name}, already exists")
                continue

            print(f"\n{'='*50}")
            print(f"{dataset} | {name} | survival (compressed features)")
            print(f"{'='*50}")

            try:
                base_dir = os.path.join(cfg["data_path"], name)
                X, labels = load_features_labels(
                    os.path.join(base_dir, "features", f"{name}.pkl"),
                    os.path.join(base_dir, "labels", f"{name}.parquet"),
                    split_path,
                )

                splits = split_and_scale_survival(X, labels, cfg["tte_col"])

                results, model = train_and_evaluate_deephit(
                    splits,
                    truncation=args.truncation,
                    model_name=f"{dataset}_{name}_compressed",
                    save_path=save_dir,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                )

            except Exception as e:
                print(f"[ERROR] {dataset} | {name}: {e}")
                import traceback; traceback.print_exc()
