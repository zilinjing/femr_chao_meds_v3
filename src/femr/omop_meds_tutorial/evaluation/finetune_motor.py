"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""
import json
import meds_reader
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
import femr.splits
import argparse
from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser
from .generate_features import get_model_name
from .eval_utils import _metric_mean_std, _update_metric_table
from .p_test.tte.utils import (
    bootstrap_cindex_parallel,
    generate_bootstrap_indices,
    make_structured_array,
    save_survival_results,
)

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt
from sklearn.model_selection import ParameterGrid
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sksurv.metrics import concordance_index_ipcw, brier_score

np.random.seed(42)
torch.manual_seed(42)


def create_arg_parser():
    # args = create_omop_meds_tutorial_arg_parser()
    args = argparse.ArgumentParser(description="Arguments for linear probing")
    args.add_argument(
        "--meds_reader",
        dest="meds_reader",
        default=None,
    )
    args.add_argument(
        "--cohort_label",
        dest="cohort_label",
        default=None,
    )
    args.add_argument(
        "--observation_window",
        dest="observation_window",
        type=int,
        default=None,
        help="The observation window for extracting features",
    )

    args.add_argument(
        "--main_split_path",
        dest="main_split_path", 
        default=None, 
        help="The path to the main split file",
    )

    args.add_argument(
        "--tte_cohort",
        dest="tte_cohort", 
        default=None, 
        help="The path to tte cohort",
    )

    args.add_argument(
        "--truncation",
        dest="truncation", 
        type = float,
        default=365, 
        help="The path to tte cohort",
    )

    args.add_argument(
        "--cohort_type",
        dest="cohort_type", 
        default=None, 
        help="The type of the cohort, either classification or phenotype",
    )

    args.add_argument(
        "--model_name",
        dest="model_name",
        required=True,
        help="The model name",
    )
    args.add_argument(
        "--model_path",
        dest="model_path",
        required=True,
        help="The model path"
    )
    args.add_argument(
        "--output_root",
        default=None,
        help="Directory where results/features_with_label will be written. Defaults to --pretraining_data.",
    )

    return args

def logistic_regression(train_set,test_data,label_output_dir):
    model = LogisticRegressionCV(scoring='roc_auc')
    model.fit(train_set['features'].to_list(), train_set['boolean_value'])

    y_pred = model.predict_proba(test_data['features'])[:, 1]

    # Convert predictions to a Polars DataFrame
    logistic_predictions = pl.DataFrame({
        "subject_id": test_data["subject_ids"].tolist(),
        "prediction_time": test_data["prediction_times"].tolist(),
        "predicted_boolean_probability": y_pred.tolist(),
        "predicted_boolean_value": None,
        "boolean_value": test_data["boolean_values"].astype(bool).tolist()
    })

    logistic_predictions = logistic_predictions.with_columns(
        pl.col("predicted_boolean_value").cast(pl.Boolean())
    )

    # Create output directory
    logistic_test_predictions = label_output_dir / "test_predictions"
    logistic_test_predictions.mkdir(exist_ok=True, parents=True)
    # Write to parquet
    logistic_predictions.write_parquet(logistic_test_predictions / "predictions.parquet")

    # roc_auc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], y_pred)
    # precision, recall, _ = sklearn.metrics.precision_recall_curve(test_data['boolean_values'], y_pred)
    # pr_auc = sklearn.metrics.auc(recall, precision)
    rng = np.random.default_rng(0)
    aurocs, auprcs = [], []

    y_true = np.asarray(test_data["boolean_values"], dtype=np.int64)
    n = len(y_true)
    for bi in range(100):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        yb = y_true[idx]
        pb = y_pred[idx]

        # AUROC needs both classes present; if degenerate, skip
        if np.unique(yb).size < 2:
            continue

        aurocs.append(sklearn.metrics.roc_auc_score(yb, pb))
        p, r, _ = sklearn.metrics.precision_recall_curve(yb, pb)
        auprcs.append(sklearn.metrics.auc(r, p))

    auroc_mean, auroc_std = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
    auprc_mean, auprc_std = float(np.mean(auprcs)), float(np.std(auprcs, ddof=1))

    return (auroc_mean, auroc_std), (auprc_mean, auprc_std)



def tte_evaluation(
    train_set_init,
    test_set_init,
    truncation,
    tte_cohort,
    save_path=None,
    model_name="model",
    n_bootstrap=10000,
    seed=42,
    n_jobs=16,
):

    #need to filter train_set and test_set from tte_cohort
    tte_cohort =  tte_cohort.rename(columns={"time_to_event_days": "time_to_event"}, errors="ignore")
    keys = tte_cohort[["subject_id", "prediction_time","time_to_event", "tte_label"]].drop_duplicates()
    train_set = train_set_init.merge(
        keys, on=["subject_id", "prediction_time"], how="inner"
    ).sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    test_set = test_set_init.merge(
        keys, on=["subject_id", "prediction_time"], how="inner"
    ).sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    print(f"before and after the train_set is {len(train_set_init)} and {len(train_set)}")
    print(f"before and after the test_set is {len(test_set_init)} and {len(test_set)}")

    test_order = list(zip(
        test_set["subject_id"].values.tolist(),
        test_set["prediction_time"].astype(str).values.tolist(),
    ))

    # Extract validation set to stop loss
    val = train_set.sample(frac=0.2, replace=False, random_state=42)
    train_set = train_set.drop(val.index)

    # Create hyper parameter selection set
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    val = val.drop(hyper.index)

    # Transform labels
    num_durations = 100
    # map continuous time to discrete bins
    labtrans = DeepHitSingle.label_transform(num_durations)

    # extract (time, event) tuples from dataframe
    get_target = lambda df: (df['time_to_event'].values, df['tte_label'].values)

    # x_train: (n_samples, n_features)
    x_train = np.stack(train_set.features.values).astype(np.float32)

    # y_train: (discrete_time_indices, event_indicators)
    y_train = labtrans.fit_transform(*get_target(train_set))

    durations_train, events_train = get_target(train_set)

    x_val = np.stack(val.features.values).astype(np.float32)
    y_val = labtrans.transform(*get_target(val))
    val = (x_val, y_val)

    x_hyper = np.stack(hyper.features.values) v
    y_hyper = labtrans.transform(*get_target(hyper))

    x_test = np.stack(test_set.features.values).astype(np.float32)
    durations_test, events_test = get_target(test_set)

    # Define one layer NN for Deephit
    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    grid = ParameterGrid({
        'dropout': [0, 0.1],
        'lr': [0.001, 0.0001],
        'batch': [1024, 2048]
    })

    # Hyperparameter selection
    best_nll = np.inf
    best_model = None
    best_params = None
    for param in grid:
        net = tt.practical.MLPVanilla(in_features, [], out_features, True, param['dropout'])

        # Train Deephit
        model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
        model.optimizer.set_lr(param['lr'])
        epochs = 1000
        _ = model.fit(x_train, y_train, param['batch'], epochs, [tt.callbacks.EarlyStopping()],
                    val_data=val)
        
        nll = model.score_in_batches(x_hyper, y_hyper)['loss']
        if nll < best_nll:
            best_model = model
            best_nll = nll
            best_params = param

    print(f"Best params: {best_params}, NLL: {best_nll:.4f}")
    

    # Evaluation best model
    surv = best_model.predict_surv_df(x_test)

    # times represent all discretized intervals
    times = surv.index.to_numpy()

    # the index closest to truncation days
    print(f"Truncate the cohort into {truncation} days")
    index_truncation = np.argmin(np.abs(times - truncation))

    # print(f"times_5yr is {times_5yr}")
    results = {'C-Index': [], 'Int-Brier': []}

    # Create structured arrays with filtered data
    # Create format for sksurv
    et_train = make_structured_array(durations_train, events_train)
    et_test = make_structured_array(durations_test, events_test)

    bootstrap_indices = generate_bootstrap_indices(len(et_test), n_bootstrap, seed)
    c_indices = bootstrap_cindex_parallel(
        surv,
        et_train,
        et_test,
        truncation,
        bootstrap_indices,
        n_jobs,
    )

    saved_results = {
        "C-Index": c_indices,
        "test_order": test_order,
        "model_name": model_name,
        "best_params": best_params,
    }

    if save_path:
        save_survival_results(saved_results, best_model, save_path, truncation)
    
    # Now all patients have outcomes within 5 years
    # So we can use the full 5-year time grid for all bootstrap samples
    min_time_sample = et_test['t'].min()
    max_time_sample = et_test['t'].max()
    
    # Filter time grid to be within BOTH:
    # 1. The observed data range [min_time, max_time]
    # 2. The 5-year truncation window [0, 1825]
    eval_mask = (times >= min_time_sample) & \
                (times <= max_time_sample) & \
                (times <= truncation)
    times_eval = times[eval_mask]
    
    for i in tqdm(range(100)):
        sample = np.random.choice(len(et_test), len(et_test))
        surv_at_5yr = surv.iloc[index_truncation, sample]
        risk_at_5yr = 1 - surv_at_5yr
        
        c_index = concordance_index_ipcw(
            et_train,
            et_test[sample],
            risk_at_5yr,
            tau=truncation
        )
        results['C-Index'].append(c_index[0])         
        
        # Safety check: need at least 2 time points for integration
        if len(times_eval) < 2:
            print(f"Warning: Bootstrap iteration {i} has insufficient time points "
                  f"(min={min_time_sample:.1f}, max={max_time_sample:.1f}), using fallback...")
        
        # Extract survival curves for the valid time points
        surv_eval = surv.iloc[eval_mask, sample].values.T
        
        # Compute integrated Brier score
        ibs = integrated_brier_score(
            et_train,
            et_test[sample],
            surv_eval,
            times_eval
        )
        results['Int-Brier'].append(ibs)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"Bootstrap Results ({len(results['C-Index'])} iterations for C-Index, "
          f"{len(results['Int-Brier'])} for Int-Brier):")
    for metric in results:
        if len(results[metric]) > 0:
            mean_val = np.mean(results[metric])
            std_val = np.std(results[metric])
            print(f'{metric}: {mean_val:.4f} (±{std_val:.4f})')
        else:
            print(f'{metric}: No valid results')
    print(f"{'='*70}\n")
    
    return results

    
def main():
    args = create_arg_parser().parse_args()
    # pretraining_data = pathlib.Path(args.pretraining_data)
    label_name = args.cohort_label
    output_root = Path(args.output_root)
    if args.cohort_label is not None:
        label_path = output_root /args.model_name/ "labels" / (args.cohort_label + '.parquet')
        if label_path.exists():
            print(f"Using the user defined label at: {label_path}")
        else:
            raise RuntimeError(f"The user provided label does not exist at {label_path}")

    # labels = ["in_hospital_mortality","readmission","long_los"]
    output_dir = Path(output_root) / "results"
    summary_dir = output_dir / "summary_tables"
    model_col = f"{args.model_name}"
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=32) as database:
        if args.observation_window:
            label_output_dir = output_dir / label_name/ f"{args.model_name}_{args.observation_window}"
        else:
            label_output_dir = output_dir / label_name/ f"{args.model_name}"
        label_output_dir.mkdir(exist_ok=True, parents=True)
        test_result_file = label_output_dir / 'metrics.json'
        features_label_data = label_output_dir / 'features_with_label'
        features_label_data.mkdir(exist_ok=True, parents=True)
        # if test_result_file.exists():
        #     print(f"The result already existed for {label_name} at {test_result_file}, it will be skipped!")
        #     continue
        labels = pd.read_parquet(output_root /args.model_name/ "labels" / (label_name + '.parquet'))

        motor_features_name = get_model_name(label_name, args.model_name, args.observation_window)
        features_path = output_root / args.model_name /"features"
        with open(features_path / f"{motor_features_name}.pkl", 'rb') as f:
            features = pickle.load(f)

        # Find labels that have no features
        labels_no_features = labels[~labels.subject_id.isin(features["subject_ids"])]
        if len(labels_no_features) > 0:
            print(f"{len(labels_no_features)} features are not included")
            # labels_no_features.to_parquet(label_output_dir / "labels_no_features.parquet")

        # Remove the labels that do not have features generated
        labels = labels[labels.subject_id.isin(features["subject_ids"])]
        labels = labels.sort_values(["subject_id", "prediction_time"])
        # labels = labels.sample(n=len(labels), random_state=42, replace=False)
        labeled_features = femr.featurizers.join_labels(features, labels,label_name)
        print(f"label features length {len(labeled_features['times'])}")

        main_split = femr.splits.SubjectSplit.load_from_csv(args.main_split_path)

        # print(main_split)
        train_mask = np.isin(labeled_features['subject_ids'], main_split.train_subject_ids)
        test_mask = np.isin(labeled_features['subject_ids'], main_split.test_subject_ids)

        # print()

        def apply_mask(values, mask):
            def apply(k, v):
                if len(v.shape) == 1:
                    return v[mask]
                elif len(v.shape) == 2:
                    return v[mask, :]
                else:
                    assert False, f"Cannot handle {k} {v.shape}"

            return {k: apply(k, v) for k, v in values.items()}

        train_data = apply_mask(labeled_features, train_mask)
        test_data = apply_mask(labeled_features, test_mask)

        # print(f"train_data length is {len(train_data)}, {train_data}")
        # print(f"test_data length is {len(test_data)},{test_data}")

        print("Saving features and labels to parquet")
        train_features_list = [feature for feature in train_data["features"]]
        test_features_list = [feature for feature in test_data["features"]]

        train_set = pd.DataFrame({
            "subject_id" : train_data["subject_ids"],
            "prediction_time" : train_data["prediction_times"],
            "boolean_value": train_data["boolean_values"],
            "features" : train_features_list
        }).sample(
            frac=1.0,
            random_state=42
        )
        train_set.to_parquet(features_label_data / "train.parquet", index=False)

        test_set = pd.DataFrame({
            "subject_id" : test_data["subject_ids"],
            "prediction_time" : test_data["prediction_times"],
            "boolean_value": test_data["boolean_values"],
            "features" : test_features_list
        })
        test_set.to_parquet(features_label_data / "test.parquet", index=False)


        print(f"Total labels: {len(labels)}")
        print(f"Total train features labels: {len(train_data['features'])}")
        print(f"Total test features labels: {len(test_data['features'])}")

        metrics = {}
        roc_auc,pr_auc = logistic_regression(train_set,test_data,label_output_dir)
        # print(label_name, roc_auc)
        metrics = {
            "auroc": roc_auc,
            "auprc": pr_auc
        }
        if args.cohort_type == "phenotype":
            tte_label_path = Path(args.tte_cohort)/label_name/f"{label_name}.parquet"
            if tte_label_path.exists():
                tte_cohort = pd.read_parquet(tte_label_path)
            else:
                tte_label_dir = Path(args.tte_cohort)/label_name
                files = sorted(tte_label_dir.glob("*.parquet"))
                dfs = [pd.read_parquet(f) for f in files]
                tte_cohort = pd.concat(dfs, ignore_index=True)

            tte_results = tte_evaluation(
                train_set,
                test_set,
                args.truncation,
                tte_cohort,
                save_path=label_output_dir,
                model_name=model_col,
            )
            metrics['C-Index'] = (float(np.mean(tte_results['C-Index'])), float(np.std(tte_results['C-Index'])))
            metrics['Int-Brier'] = (float(np.mean(tte_results['Int-Brier'])), float(np.std(tte_results['Int-Brier'])))
        # tte_result_file = label_output_dir / 'tte_metrics.json'
        # with open(tte_result_file, "w") as f:
        #     json.dump(metrics, f, indent=4)
        with open(test_result_file, "w") as f:
            json.dump(metrics, f, indent=4)

        auroc_mean, auroc_std = _metric_mean_std(metrics, "auroc")
        auprc_mean, auprc_std = _metric_mean_std(metrics, "auprc")
        cidx_mean, cidx_std   = _metric_mean_std(metrics, "C-Index")
        ibr_mean,  ibr_std    = _metric_mean_std(metrics, "Int-Brier")

        if auroc_mean is not None:
            _update_metric_table(summary_dir, "auroc", label_name, model_col, auroc_mean, auroc_std)
        if auprc_mean is not None:
            _update_metric_table(summary_dir, "auprc", label_name, model_col, auprc_mean, auprc_std)

        if cidx_mean is not None:
            _update_metric_table(summary_dir, "cindex", label_name, model_col, cidx_mean, cidx_std)
        if ibr_mean is not None:
            _update_metric_table(summary_dir, "int_brier", label_name, model_col, ibr_mean, ibr_std)



if __name__ == "__main__":
    main()

