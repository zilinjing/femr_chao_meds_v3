# =========================
# Imports & utilities
# =========================
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import torchtuples as tt
from pycox.models import DeepHitSingle, CoxPH
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score


# ========================= 
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# IO helpers
# =========================
def save_metrics(metrics, path):
    with open(path, "wb") as f:
        pickle.dump(metrics, f)


def load_metrics(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize(metrics: dict):
    for k, v in metrics.items():
        v = np.asarray(v)
        print(f"{k}: {v.mean():.3f} ({v.std():.3f})")


# =========================
# Preprocessing
# =========================
def remove_duplicates(data: pd.DataFrame, labels: pd.DataFrame):
    return data[~data.index.duplicated(keep="last")], labels[~labels.index.duplicated(keep="last")]


def scale_split(covariates, split_mask):
    scaler = StandardScaler().fit(covariates[split_mask["train"]])
    return {
        k: scaler.transform(covariates[v])
        for k, v in split_mask.items()
    }


# =========================
# Survival pipeline
# =========================
def split_survival(data, labels, tte_col):
    split = labels["split_name"].values
    tte = labels[tte_col].values
    event = labels["tte_label"].values.astype(bool)

    mask = {
        "train": split == "train",
        "val": split == "tuning",
        "hyper": split == "hyper",
        "test": split == "test",
    }

    X = scale_split(data.values, mask)

    return (
        X["train"], (tte[mask["train"]], event[mask["train"]]),
        X["val"],   (tte[mask["val"]],   event[mask["val"]]),
        X["hyper"], (tte[mask["hyper"]], event[mask["hyper"]]),
        X["test"],  (tte[mask["test"]],  event[mask["test"]]),
    )


def evaluate_survival(surv_df, y_train, y_test, horizon_days=5*365, n_boot=100):
    t_train, e_train = y_train
    t_test, e_test = y_test

    print(t_train, t_test)
    print(t_train.max(), t_test.max())
    print(e_train.mean(), e_test.mean())

    et_train = np.array(list(zip(e_train, t_train)), dtype=[("e", bool), ("t", float)])
    et_test = np.array(list(zip(e_test, t_test)), dtype=[("e", bool), ("t", float)])

    times = surv_df.index.values
    tau_idx = np.argmin(np.abs(times - horizon_days))
    risk = 1.0 - surv_df.iloc[tau_idx].values

    mask = (times <= horizon_days) & (times >= t_test.min())
    times_eval = times[mask]

    results = {"C-Index": [], "Int-Brier": []}

    for _ in tqdm(range(n_boot)):
        idx = np.random.choice(len(et_test), len(et_test), replace=True)

        c = concordance_index_ipcw(
            et_train, et_test[idx], risk[idx], tau=horizon_days
        )[0]

        surv_eval = surv_df.iloc[mask, idx].T.values
        ibs = integrated_brier_score(
            et_train, et_test[idx], surv_eval, times_eval
        )

        results["C-Index"].append(c)
        results["Int-Brier"].append(ibs)

    summarize(results)
    return results


def train_survival(x_train, y_train, x_val, y_val, x_hyper, y_hyper, x_test, y_test, model_type):
    set_seed()

    in_features = x_train.shape[1]
    out_features = 1
    labtrans = None

    if model_type == "deephit":
        labtrans = LabTransDiscreteTime(100)
        y_train_norm = labtrans.fit_transform(*y_train)
        y_val = labtrans.transform(*y_val)
        y_hyper = labtrans.transform(*y_hyper)
        out_features = labtrans.out_features
    else:
        y_train_norm = y_train

    grid = ParameterGrid({
        "lr": [1e-4],
        "batch": [1024],
        "do": [0, 0.1],
        "layers": [[50] * i for i in range(3)],
    })

    best_model, best_loss = None, np.inf

    for p in grid:
        net = tt.practical.MLPVanilla(
            in_features, p["layers"], out_features, batch_norm=True, dropout=p['do']
        )

        model = (
            DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
            if model_type == "deephit"
            else CoxPH(net, tt.optim.Adam)
        )

        model.optimizer.set_lr(p["lr"])
        model.fit(
            x_train, y_train_norm, p["batch"], 100,
            callbacks=[tt.callbacks.EarlyStopping()],
            val_data=(x_hyper, y_hyper),
        )

        loss = model.score_in_batches(x_val, y_val)["loss"]
        if loss < best_loss:
            best_loss, best_model = loss, model

    if model_type == "cox":
        best_model.compute_baseline_hazards()

    surv_df = best_model.predict_surv_df(x_test)
    return evaluate_survival(surv_df, y_train, y_test)


# =========================
# Binary pipeline
# =========================
def split_binary(data, labels):
    split = labels["split_name"].values
    y = labels["boolean_value"].values

    mask = {
        "train": split == "train",
        "val": split == "tuning",
        "hyper": split == "hyper",
        "test": split == "test",
    }

    X = scale_split(data.values, mask)

    return X["train"], y[mask["train"]], X["val"], y[mask["val"]], X["hyper"], y[mask["hyper"]], X["test"], y[mask["test"]], labels.index[mask["test"]]


def evaluate_binary(pred, y, n_boot=100):
    results = {"AUROC": []}

    for _ in tqdm(range(n_boot)):
        idx = np.random.choice(len(y), len(y), replace=True)
        results["AUROC"].append(roc_auc_score(y[idx], pred[idx]))

    summarize(results)
    return results


def train_binary(x_train, y_train, x_val, y_val, x_hyper, y_hyper, x_test, y_test, test_index, n_jobs=-1):
    set_seed()

    grid = ParameterGrid({
        "learning_rate": [0.01],
        "max_depth": [3, 10],
        "n_estimators": [200],
    })

    best_model, best_auc = None, -np.inf

    for p in grid:
        print(f"p is {p}")
        model = xgb.XGBClassifier(
            **p,
            tree_method="hist",
            n_jobs=n_jobs,
            eval_metric="auc",
        )
        model.fit(x_train, y_train, eval_set=[(x_hyper, y_hyper)], verbose=False)

        val_pred = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)

        if auc > best_auc:
            best_auc, best_model = auc, model

    test_pred = best_model.predict_proba(x_test)[:, 1]
    metrics = evaluate_binary(test_pred, y_test)

    pred_df = pd.DataFrame({
        "subject_id": test_index.get_level_values("subject_id"),
        "prediction_time": test_index.get_level_values("prediction_time"),
        "predicted_boolean_probability": test_pred,
        "boolean_value": y_test,
    })

    return metrics, pred_df