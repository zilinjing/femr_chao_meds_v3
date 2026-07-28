# =========================
# Imports & utilities
# =========================
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import xgboost as xgb
import torchtuples as tt
from pycox.models import DeepHitSingle, CoxPH
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

import polars as pl
from meds_evaluation.evaluate import evaluate_bootstrapped_binary_classification

# The same helpers finetune_motor.py uses, so bootstrap indices are byte-identical
# across arms and `paired_bootstrap_test` can be applied to these results directly.
from femr.omop_meds_tutorial.evaluation.p_test.tte.utils import (
    bootstrap_cindex_parallel,
    generate_bootstrap_indices,
    make_structured_array,
    save_survival_results,
)

# Report on 100 resamples (paper tables); persist 10,000 for later paired tests.
N_REPORT = 100
N_SAVE = 10000


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
        if k.startswith("_"):
            # bookkeeping (e.g. the selected hyperparameters), not a bootstrap metric
            print(f"{k}: {v}")
            continue
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


def split_sparse(covariates, split_mask):
    """Row-slice a sparse matrix per split, without scaling.

    Gradient-boosted trees are invariant to monotone per-feature transforms, so
    the StandardScaler in `scale_split` is unnecessary for XGBoost -- and it
    cannot run on a sparse matrix with `with_mean=True` anyway. Keeping the
    matrix sparse is what makes the wider value-aware feature set tractable.
    """
    return {k: covariates[np.flatnonzero(v)] for k, v in split_mask.items()}


def densify_for_mlp(covariates, split_mask, artifacts=None):
    """Densify + scale for DeepHit, dropping columns that are constant in train.

    DeepHit is a torchtuples MLP and needs dense float32 input. All-zero (or
    constant) training columns carry no information and only inflate the dense
    matrix, so they are dropped first. Prints the final shape: if this is still
    too wide for memory, raise --min_support at extraction time rather than
    dropping a window.
    """
    train_rows = covariates[np.flatnonzero(split_mask["train"])]
    nnz_per_col = np.asarray((train_rows != 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(nnz_per_col > 0)
    print(f"densify_for_mlp: keeping {len(keep)}/{covariates.shape[1]} columns")

    # float32 end to end. sklearn preserves the input dtype, and torch consumes float32
    # anyway, so going through float64 only doubled both the memory traffic and the peak
    # RSS (106 GB -> ~55 GB on the AMI TTE cohort) for no numerical benefit.
    scaler = StandardScaler()
    scaler.fit(np.asarray(train_rows[:, keep].todense(), dtype=np.float32))

    if artifacts is not None:
        # Everything needed to turn a raw feature matrix back into model input.
        # Without these the saved net is unusable: the column subset and the
        # standardisation are both fitted on this task's training split.
        artifacts["keep_columns"] = keep
        artifacts["n_columns_in"] = int(covariates.shape[1])
        artifacts["scaler_mean"] = scaler.mean_
        artifacts["scaler_scale"] = scaler.scale_

    return {
        k: scaler.transform(
            np.asarray(covariates[np.flatnonzero(v)][:, keep].todense(), dtype=np.float32)
        ).astype("float32")
        for k, v in split_mask.items()
    }


# =========================
# Survival pipeline
# =========================
def split_survival(data, labels, tte_col, artifacts=None):
    split = labels["split_name"].values
    tte = labels[tte_col].values
    event = labels["tte_label"].values.astype(bool)

    mask = {
        "train": split == "train",
        "val": split == "tuning",
        "hyper": split == "hyper",
        "test": split == "test",
    }

    # DeepHit needs dense, scaled input; see densify_for_mlp.
    X = densify_for_mlp(data, mask, artifacts=artifacts)

    return (
        X["train"], (tte[mask["train"]], event[mask["train"]]),
        X["val"],   (tte[mask["val"]],   event[mask["val"]]),
        X["hyper"], (tte[mask["hyper"]], event[mask["hyper"]]),
        X["test"],  (tte[mask["test"]],  event[mask["test"]]),
    )


def evaluate_survival(surv_df, y_train, y_test, truncation=5 * 365, n_report=N_REPORT,
                      n_save=N_SAVE, test_order=None, save_path=None,
                      model_name="meds_tab_deephit", best_params=None, model=None,
                      n_jobs=16):
    """C-index / integrated Brier score, following finetune_motor.py exactly.

    Two bootstraps, matching the foundation-model arms:

    * **reported** -- `n_report` (100) resamples drawn with `np.random.default_rng(42)`,
      the same generator and call pattern as `finetune_motor.tte_evaluation`'s `ibs_rng`.
      These give the mean (std) in the paper tables.
    * **saved** -- `n_save` (10,000) resamples from `generate_bootstrap_indices(n, 10000,
      seed=42)`, the shared helper. Because that helper is deterministic in `n_test`, the
      indices are identical across every model evaluated on the same test set, which is
      what makes `p_test.tte.utils.paired_bootstrap_test` valid against the FM results.

    `paired_bootstrap_test` asserts equality of `test_order`, so `test_order` must be the
    list of (subject_id, prediction_time) pairs in test-row order.
    """
    t_train, e_train = y_train
    t_test, e_test = y_test

    et_train = make_structured_array(t_train, e_train)
    et_test = make_structured_array(t_test, e_test)

    times = surv_df.index.to_numpy()

    # ---- saved: 10,000 shared-index resamples for later paired testing ----
    bootstrap_indices = generate_bootstrap_indices(len(et_test), n_save, seed=42)
    c_indices_saved = bootstrap_cindex_parallel(
        surv_df, et_train, et_test, truncation, bootstrap_indices, n_jobs
    )

    if save_path is not None:
        saved = {
            "C-Index": c_indices_saved,
            "test_order": test_order,
            "model_name": model_name,
            "best_params": best_params,
        }
        if model is not None:
            # Same filenames and payload as the FM arms (tte_results_/tte_model_/tte_metrics_).
            save_survival_results(saved, model, save_path, truncation)
        else:
            # save_survival_results calls model.save_net(); without a model, write the
            # results pickle and metrics json itself, keeping the identical layout.
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"tte_results_{truncation}.pkl"), "wb") as f:
                pickle.dump(saved, f)
            with open(os.path.join(save_path, f"tte_metrics_{truncation}.json"), "w") as f:
                json.dump(
                    {
                        "mean": float(np.mean(c_indices_saved)),
                        "std": float(np.std(c_indices_saved)),
                        "truncation": float(truncation),
                        "n_bootstrap": len(c_indices_saved),
                        "model_name": model_name,
                        "best_params": best_params,
                    },
                    f, indent=2, default=str,
                )

    # ---- reported: 100 resamples, identical procedure to finetune_motor ----
    tau_idx = np.argmin(np.abs(times - truncation))
    eval_mask = (times >= et_test["t"].min()) & (times <= et_test["t"].max()) & (times <= truncation)
    times_eval = times[eval_mask]

    results = {"C-Index": [], "Int-Brier": []}
    rng = np.random.default_rng(42)
    for _ in tqdm(range(n_report), desc="bootstrap (reported)"):
        sample = rng.choice(len(et_test), len(et_test))
        risk = 1.0 - surv_df.iloc[tau_idx, sample].values
        results["C-Index"].append(
            concordance_index_ipcw(et_train, et_test[sample], risk, tau=truncation)[0]
        )
        results["Int-Brier"].append(
            integrated_brier_score(
                et_train, et_test[sample], surv_df.iloc[eval_mask, sample].T.values, times_eval
            )
        )

    summarize(results)
    results["_saved_bootstrap"] = {
        "n_save": len(c_indices_saved),
        "mean": float(np.mean(c_indices_saved)),
        "std": float(np.std(c_indices_saved)),
        "path": str(save_path) if save_path else None,
    }
    print(f"saved {len(c_indices_saved)} C-index resamples for paired testing "
          f"(mean {np.mean(c_indices_saved):.4f})")
    return results


def train_survival(x_train, y_train, x_val, y_val, x_hyper, y_hyper, x_test, y_test, model_type,
                   test_order=None, save_path=None, model_name="meds_tab_deephit",
                   grid_mode="fast", patience=5, n_jobs=16, artifacts=None,
                   wd_values=None):
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

    # Tuning budget is a strict SUPERSET of the DeepHit head used for the
    # foundation-model probes (tte_evaluation/tte_grid_search_filter_new_phenotype.py:79-97),
    # so the tabular baseline cannot be called under-tuned relative to them:
    #
    #   FM probe:  dropout {0, 0.1} x lr {1e-3, 1e-4} x batch {1024, 2048}, 8 configs,
    #              1000 epochs, architecture hard-coded to []  (linear -- it is a probe)
    #   here:      the same three axes, plus [50] and [50,50] as options, 24 configs,
    #              the same 1000-epoch budget.
    #
    # The submitted baseline used a single lr (1e-4), a single batch (1024) and only
    # 100 epochs -- 10x less training than the probes it is compared against. That was
    # the real handicap; the architecture grid already included non-linear options.
    #
    # `wd` (decoupled weight decay, via tt.optim.AdamW) is the one axis the FM probe
    # does not have, and it is added because this arm needs it: a linear DeepHit over
    # ~80k standardised features with 100 duration bins is ~8M parameters against ~95k
    # training rows, and without an L2 penalty the validation loss rises from the first
    # epoch (train 4.04 -> 2.45 while val 6.76 -> 8.03 on a 49k-feature subset).
    # The FM probe sees 768 features (~77k parameters) and does not have this problem.
    # Regularising a *baseline* can only strengthen it, so this cannot be called unfair.
    # Architecture options are bottleneck *widths*, not depths. At ~74k inputs and 100
    # duration bins the first Linear is >99% of the parameters, so depth is nearly free
    # and nearly inert ([50] -> [50,50] adds 2,650 of 3.7M parameters) while width moves
    # capacity by millions:  [64] 4.8M  <  [] 7.4M  <  [128] 9.5M.
    # `[]` (no hidden layer, no nonlinearity) is the FM probe's architecture and must
    # stay: it is what makes this grid a superset of the probe's search space.
    #   fast -> 32 configs, full -> 48 configs.
    layer_options = [[], [50]] if grid_mode == "fast" else [[], [64], [128]]
    grid = ParameterGrid({
        "lr": [1e-3, 1e-4],
        "batch": [1024, 2048],
        "do": [0, 0.1],
        "layers": layer_options,
        # Two values, not three: this is where the 72 -> 48 config cut was taken,
        # because weight decay is the only axis the FM probe does not also search.
        # Keeping lr/batch/dropout at full resolution preserves the parity argument.
        # 1e-4 was dropped rather than 1e-2: at 4.8-9.5M parameters it is mild enough
        # to be near-indistinguishable from 0.0, so it would have spent half the grid
        # measuring the same regime twice.
        "wd": wd_values if wd_values is not None else [0.0, 1e-2],
    })
    print(f"DeepHit grid: {len(grid)} configs (mode={grid_mode})")
    # Matches `epochs = 1000` in the FM probe. EarlyStopping still cuts most
    # configurations short; torchtuples places the model on GPU automatically when one
    # is available, which is what makes this budget affordable at ~80k input features.
    epochs = 1000

    best_model, best_loss, best_params = None, np.inf, None

    for p in grid:
        net = tt.practical.MLPVanilla(
            in_features, p["layers"], out_features, batch_norm=True, dropout=p['do']
        )

        # AdamW with decoupled weight decay; wd=0.0 reproduces plain Adam.
        optimizer = tt.optim.AdamW(lr=p["lr"], decoupled_weight_decay=p["wd"])
        model = (
            DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
            if model_type == "deephit"
            else CoxPH(net, optimizer)
        )

        model.optimizer.set_lr(p["lr"])
        # `x_hyper` early-stops and `x_val` selects, which is the reverse of the FM
        # probe's naming. Immaterial: both are disjoint random 10% slices of train of
        # equal size, so the two conventions are the same experiment.
        model.fit(
            x_train, y_train_norm, p["batch"], epochs,
            # patience 5 rather than the torchtuples default of 10. At this width the
            # validation loss turns within a couple of epochs (train 4.04 -> 2.45 while
            # val 6.76 -> 8.03 unregularised), so 10 epochs of patience buys nothing but
            # spends ~2x the wall-clock per configuration.
            callbacks=[tt.callbacks.EarlyStopping(patience=patience)],
            val_data=(x_hyper, y_hyper),
        )

        loss = model.score_in_batches(x_val, y_val)["loss"]
        if loss < best_loss:
            best_loss, best_model, best_params = loss, model, p

    if model_type == "cox":
        best_model.compute_baseline_hazards()

    # Record which architecture won. If `layers == []` is selected, the tabular
    # DeepHit is linear -- identical in form to the FM probe head -- and the
    # "is the head allowed hidden layers?" fairness question is moot for that task.
    print(f"best DeepHit config: {best_params} (val NLL {best_loss:.4f})")

    # ---- reproducibility bundle -------------------------------------------------
    # save_survival_results already writes the net (tte_model_<tau>.pt). On its own
    # that is not enough to score new data: the column subset, the standardisation
    # and the discrete-time cuts are all fitted here. Persist them together.
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        bundle = dict(artifacts or {})
        bundle.update(
            {
                "best_params": {**best_params, "layers": list(best_params["layers"])},
                "val_nll": float(best_loss),
                "labtrans_cuts": getattr(labtrans, "cuts", None),
                "num_durations": int(out_features),
                "in_features": int(in_features),
                "model_type": model_type,
                "model_name": model_name,
                "grid_mode": grid_mode,
                "patience": patience,
                "n_configs": len(grid),
                "seed": 42,
                "net_file": f"tte_model_{5 * 365}.pt",
            }
        )
        bundle_path = os.path.join(save_path, "reproducibility.pkl")
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"saved reproducibility bundle -> {bundle_path} "
              f"(keys: {sorted(bundle.keys())})")

    surv_df = best_model.predict_surv_df(x_test)
    metrics = evaluate_survival(
        surv_df, y_train, y_test, test_order=test_order, save_path=save_path,
        model_name=model_name, model=best_model, n_jobs=n_jobs,
        best_params={**best_params, "layers": list(best_params["layers"])},
    )
    metrics["_best_params"] = [
        {**best_params, "layers": list(best_params["layers"]), "val_nll": float(best_loss)}
    ]
    return metrics


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

    # XGBoost consumes the CSR matrix directly -- no densify, no scaling.
    X = split_sparse(data, mask)

    return X["train"], y[mask["train"]], X["val"], y[mask["val"]], X["hyper"], y[mask["hyper"]], X["test"], y[mask["test"]], labels.index[mask["test"]]


def evaluate_binary_meds(pred_df: pd.DataFrame, bootstrapping: int = 100) -> dict:
    """Bootstrapped binary-classification metrics via the meds_evaluation package.

    This is the reported number: `mean_roc_auc_score` +- `std_roc_auc_score` over
    `bootstrapping` resamples. meds_evaluation resamples through its own `_resample`
    helper rather than a plain row bootstrap, so use it for every arm (foundation
    models and this baseline) or the error bars are not comparable.

    Note: meds_evaluation covers **binary classification only**. There is no
    survival/C-index entry point in the package, so time-to-event tasks keep the
    `concordance_index_ipcw` bootstrap in `evaluate_survival`.
    """
    df = pl.DataFrame(
        {
            "subject_id": pred_df["subject_id"].to_numpy().astype("int64"),
            "prediction_time": pd.to_datetime(pred_df["prediction_time"]).to_numpy(),
            "boolean_value": pred_df["boolean_value"].to_numpy().astype(bool),
            "predicted_boolean_value": pred_df["predicted_boolean_value"].to_numpy().astype(bool),
            "predicted_boolean_probability": pred_df["predicted_boolean_probability"]
            .to_numpy()
            .astype("float64"),
        }
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime))

    res = evaluate_bootstrapped_binary_classification(df, bootstrapping=bootstrapping)
    print(f"MEDS AUROC: {res['mean_roc_auc_score']:.4f} +- {res['std_roc_auc_score']:.4f}"
          f"  (n={bootstrapping} bootstraps)")
    return res


def evaluate_binary(pred, y, n_report=N_REPORT, n_save=N_SAVE, test_order=None,
                    save_path=None, model_name="meds_tab_xgboost", best_params=None):
    """AUROC / AUPRC bootstraps, following finetune_motor.logistic_regression exactly.

    * **reported** -- `n_report` (100) resamples with `np.random.default_rng(0)` and
      `rng.integers(0, n, size=n)`, skipping degenerate single-class draws, std with
      `ddof=1`. Byte-identical procedure to the FM arms.
    * **saved** -- `n_save` (10,000) resamples from the shared
      `generate_bootstrap_indices(n, 10000, seed=42)` helper, written to
      `binary_results_{model_name}.pkl` alongside `test_order`, so binary tasks can be
      paired-tested the same way `p_test.tte.utils.paired_bootstrap_test` handles TTE.

    NOTE: `finetune_motor.logistic_regression` currently computes only the 100 reported
    resamples and discards them. To paired-test binary tasks against the FM arms, the
    same `generate_bootstrap_indices` save needs to be added there too -- otherwise the
    FM side has no per-resample AUROC to pair against.
    """
    y_true = np.asarray(y, dtype=np.int64)
    y_pred = np.asarray(pred, dtype=float)
    n = len(y_true)

    def _auroc_auprc(idx):
        yb, pb = y_true[idx], y_pred[idx]
        if np.unique(yb).size < 2:
            return None
        precision, recall, _ = precision_recall_curve(yb, pb)
        return roc_auc_score(yb, pb), auc(recall, precision)

    # ---- reported: 100 resamples, matching finetune_motor ----
    results = {"AUROC": [], "AUPRC": []}
    rng = np.random.default_rng(0)
    for _ in range(n_report):
        got = _auroc_auprc(rng.integers(0, n, size=n))
        if got is None:
            continue
        results["AUROC"].append(got[0])
        results["AUPRC"].append(got[1])

    print(f"AUROC: {np.mean(results['AUROC']):.4f} "
          f"({np.std(results['AUROC'], ddof=1):.4f})  [n={len(results['AUROC'])} of {n_report}]")
    print(f"AUPRC: {np.mean(results['AUPRC']):.4f} "
          f"({np.std(results['AUPRC'], ddof=1):.4f})")

    # ---- saved: 10,000 shared-index resamples for later paired testing ----
    saved_auroc, saved_auprc = [], []
    for idx in tqdm(generate_bootstrap_indices(n, n_save, seed=42), desc="bootstrap (saved)"):
        got = _auroc_auprc(idx)
        if got is None:
            continue
        saved_auroc.append(got[0])
        saved_auprc.append(got[1])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"binary_results_{model_name}.pkl")
        with open(out, "wb") as f:
            pickle.dump(
                {
                    "AUROC": saved_auroc,
                    "AUPRC": saved_auprc,
                    "test_order": test_order,
                    "model_name": model_name,
                    "best_params": best_params,
                },
                f,
            )
        with open(os.path.join(save_path, f"binary_metrics_{model_name}.json"), "w") as f:
            json.dump(
                {
                    "mean_auroc": float(np.mean(saved_auroc)),
                    "std_auroc": float(np.std(saved_auroc)),
                    "n_bootstrap": len(saved_auroc),
                    "model_name": model_name,
                    "best_params": best_params,
                },
                f, indent=2, default=str,
            )
        print(f"saved {len(saved_auroc)} AUROC resamples to {out}")

    results["_saved_bootstrap"] = {
        "n_save": len(saved_auroc),
        "mean_auroc": float(np.mean(saved_auroc)),
        "std_auroc": float(np.std(saved_auroc)),
        "path": str(save_path) if save_path else None,
    }
    return results


def train_binary(x_train, y_train, x_val, y_val, x_hyper, y_hyper, x_test, y_test, test_index,
                 n_jobs=-1, save_path=None, model_name="meds_tab_xgboost", device="cpu"):
    set_seed()

    # Widened from the submitted baseline (2 configurations) toward the search
    # budget the MEDS-Tab recipe uses (100 Optuna trials). 24 configurations with
    # early stopping on the held-out `hyper` split.
    grid = ParameterGrid({
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6, 10],
        "n_estimators": [200, 1000],
        "subsample": [0.8],
        "colsample_bytree": [0.5, 1.0],
    })

    best_model, best_auc, best_params = None, -np.inf, None

    for p in grid:
        print(f"p is {p}")
        model = xgb.XGBClassifier(
            **p,
            tree_method="hist",
            # device="cuda" is ~9x faster than 32 CPU threads on this matrix
            # (13.9s vs 126.1s per config at 124,734 x 80,707, 217M nnz) with
            # equivalent AUROC (0.8392 vs 0.8368; they differ only in where early
            # stopping fired). Same algorithm -- `hist` either way.
            device=device,
            n_jobs=n_jobs,
            eval_metric="auc",
            # Constructor arg requires xgboost >= 1.6; passing it to .fit() was
            # removed in xgboost 2.0.
            early_stopping_rounds=20,
        )
        model.fit(x_train, y_train, eval_set=[(x_hyper, y_hyper)], verbose=False)

        val_pred = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)

        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, p

    print(f"best XGBoost config: {best_params} (val AUROC {best_auc:.4f})")
    test_pred = best_model.predict_proba(x_test)[:, 1]
    # test_order must match the FM arms for paired testing (see evaluate_binary).
    test_order = [
        (int(s_), pd.Timestamp(t_).isoformat())
        for s_, t_ in zip(
            test_index.get_level_values("subject_id"),
            test_index.get_level_values("prediction_time"),
        )
    ]
    metrics = evaluate_binary(
        test_pred, y_test, test_order=test_order, save_path=save_path,
        model_name=model_name, best_params=best_params,
    )
    metrics["_best_params"] = [{**best_params, "val_auroc": float(best_auc)}]

    pred_df = pd.DataFrame({
        "subject_id": test_index.get_level_values("subject_id"),
        "prediction_time": test_index.get_level_values("prediction_time"),
        "predicted_boolean_probability": test_pred,
        # meds_evaluation requires a thresholded prediction alongside the probability.
        "predicted_boolean_value": test_pred >= 0.5,
        "boolean_value": y_test,
    })

    # The reported metric: meds_evaluation bootstrapped mean +- std.
    metrics["_meds_evaluation"] = evaluate_binary_meds(pred_df)

    return metrics, pred_df