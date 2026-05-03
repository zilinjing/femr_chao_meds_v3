from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import os
import argparse
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from femr.omop_meds_tutorial.evaluation.eval_utils import _metric_mean_std, _update_metric_table
from femr.omop_meds_tutorial.evaluation.p_test.tte.utils import (
    bootstrap_cindex_parallel,
    generate_bootstrap_indices,
    make_structured_array,
    save_survival_results,
)

np.random.seed(42)
torch.manual_seed(42)

def tte_evaluation(train_set, test_set, task, model_name, save_path, truncation=365, tte_cohort_path=None):
    # === [unchanged] cohort loading + merge ===
    tte_cohort_path = Path(tte_cohort_path) / task / f"{task}.parquet"
    print(f"reading cohort from {tte_cohort_path}")
    if tte_cohort_path.exists():
        tte_cohort = pd.read_parquet(tte_cohort_path)
    else:
        task_dir = tte_cohort_path.parent
        parquet_files = list(task_dir.glob("*.parquet"))
        tte_cohort = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    tte_cohort = tte_cohort.rename(columns={"time_to_event_days": "time_to_event"}, errors="ignore")
    keys = tte_cohort[["subject_id", "prediction_time", "time_to_event", "tte_label"]].drop_duplicates()
    train_set = train_set.merge(keys, on=["subject_id", "prediction_time"], how="inner") \
        .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    test_set = test_set.merge(keys, on=["subject_id", "prediction_time"], how="inner") \
        .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    test_order = list(zip(
        test_set["subject_id"].values.tolist(),
        test_set["prediction_time"].astype(str).values.tolist(),
    ))

    print(f"truncation is {truncation}")
    val = train_set.sample(frac=0.2, replace=False, random_state=42)
    train_set = train_set.drop(val.index)
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    val = val.drop(hyper.index)

    # === [unchanged] label transform + feature extraction ===
    num_durations = 100
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time_to_event'].values, df['tte_label'].values)

    x_train = np.stack(train_set.features.values).astype(np.float32)
    y_train = labtrans.fit_transform(*get_target(train_set))
    durations_train, events_train = get_target(train_set)

    x_val = np.stack(val.features.values).astype(np.float32)
    y_val = labtrans.transform(*get_target(val))
    val = (x_val, y_val)

    x_hyper = np.stack(hyper.features.values).astype(np.float32)
    y_hyper = labtrans.transform(*get_target(hyper))

    x_test = np.stack(test_set.features.values).astype(np.float32)
    durations_test, events_test = get_target(test_set)

    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    # Structured arrays (needed in both cached and uncached paths)
    et_train = make_structured_array(durations_train, events_train)
    et_test = make_structured_array(durations_test, events_test)

    # ===================================================================
    # CACHE CHECK: skip grid search + parallel C-Index bootstrap if both
    # tte_metrics_{truncation}.json and tte_model_{truncation}.pt exist
    # ===================================================================
    save_path = Path(save_path)
    metrics_path = save_path / f"tte_metrics_{truncation}.json"
    model_path = save_path / f"tte_model_{truncation}.pt"
    cache_exists = metrics_path.exists() and model_path.exists()

    if cache_exists:
        print(f"Cache found at {save_path}, reloading model and skipping grid search")
        with open(metrics_path) as f:
            cached_metrics = json.load(f)
        best_params = cached_metrics["best_params"]

        # Reconstruct network with cached best_params and load weights
        net = tt.practical.MLPVanilla(in_features, [], out_features, True, best_params['dropout'])
        best_model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
        best_model.load_net(str(model_path))
    else:
        # === [unchanged] grid search ===
        grid = ParameterGrid({
            'dropout': [0, 0.1],
            'lr': [0.001, 0.0001],
            'batch': [1024, 2048]
        })

        best_nll = np.inf
        best_model = None
        best_params = None
        for param in grid:
            net = tt.practical.MLPVanilla(in_features, [], out_features, True, param['dropout'])
            model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
            model.optimizer.set_lr(param['lr'])
            epochs = 1000
            _ = model.fit(x_train, y_train, param['batch'], epochs,
                          [tt.callbacks.EarlyStopping()], val_data=val)
            nll = model.score_in_batches(x_hyper, y_hyper)['loss']
            if nll < best_nll:
                best_model = model
                best_nll = nll
                best_params = param
        print(f"Best params: {best_params}, NLL: {best_nll:.4f}")

    # === Compute survival predictions on test set (needed in both paths) ===
    surv = best_model.predict_surv_df(x_test)
    times = surv.index.to_numpy()
    print(f"Truncate the cohort into {truncation} days")
    index_truncation = np.argmin(np.abs(times - truncation))

    # Only run the 10k parallel C-Index bootstrap and save if cache is missing
    if not cache_exists:
        bootstrap_indices = generate_bootstrap_indices(len(et_test), 10000, 42)
        c_indices = bootstrap_cindex_parallel(
            surv, et_train, et_test, truncation, bootstrap_indices, 16,
        )
        saved_results = {
            "C-Index": c_indices,
            "test_order": test_order,
            "model_name": model_name,
            "best_params": best_params,
        }
        if save_path:
            save_survival_results(saved_results, best_model, save_path, truncation)

    # ===================================================================
    # IBS bootstrap loop (always runs — IBS is not cached)
    # ===================================================================
    results = {'C-Index': [], 'Int-Brier': []}

    # FIX 1: local RNG isolated from global state — sample sequence is
    # reproducible regardless of what pycox/torch did before this point
    ibs_rng = np.random.default_rng(42)

    for i in tqdm(range(100)):
        sample = ibs_rng.choice(len(et_test), len(et_test))
        et_test_sample = et_test[sample]

        # C-Index at truncation
        surv_at_tau = surv.iloc[index_truncation, sample]
        risk_at_tau = 1 - surv_at_tau
        c_index = concordance_index_ipcw(et_train, et_test_sample, risk_at_tau, tau=truncation)
        results['C-Index'].append(c_index[0])

        # FIX 2: per-sample valid time grid — integrated_brier_score requires
        # every time to lie in [min, max] of THIS bootstrap sample, not the full test set
        sample_min = et_test_sample['t'].min()
        sample_max = et_test_sample['t'].max()
        eval_mask = (times >= sample_min) & (times <= sample_max) & (times <= truncation)
        times_eval = times[eval_mask]

        if len(times_eval) < 2:
            print(f"Warning: bootstrap iter {i} has <2 valid time points, skipping IBS")
            continue

        surv_eval = surv.iloc[eval_mask, sample].values.T
        ibs = integrated_brier_score(et_train, et_test_sample, surv_eval, times_eval)
        results['Int-Brier'].append(ibs)

    # === [unchanged] display ===
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

def run(tasks, old_phenotype_path_str,truncation,tte_cohort,model_list):
    old_phenotype_dir = Path(old_phenotype_path_str)
    # phenotype_label_dir = Path("/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_tte")
    # key_cols = ["subject_id", "prediction_time", "tte_label", "time_to_event"]
    if model_list is not None:
        model_list = model_list.split(",")
        print(f"Skipping model list is {model_list}")
    for task in tasks:
        old_phenotype_path = old_phenotype_dir / task
        # old_phenotype_path = old_phenotype_dir / task
        if not old_phenotype_path.exists():
            print(f"Skip task {task}: {old_phenotype_path} not found.")
            continue
        # if tte_cohort is not None:
        #     tte_cohort = Path(tte_cohort)/task/f"{task}.parquet"
        # the subdir includes pregenerated embeddings for all models: motor, tpp, mtpp
        for subdir in old_phenotype_path.iterdir():
            if model_list is not None and subdir.name in model_list:
                print(f"Skipping model {subdir.name}, continue")
                continue
            if not subdir.is_dir():
                continue
            train_path = subdir / "features_with_label" / "train.parquet"
            test_path  = subdir / "features_with_label" / "test.parquet"
            if not (train_path.exists() and test_path.exists()):
                print(f"Missing train/test parquet under {subdir}, skipping.")
                continue
            train_parquet = pd.read_parquet(train_path)
            test_parquet  = pd.read_parquet(test_path)
            model_name = subdir.name
            save_dir = old_phenotype_dir / task / model_name

            tte_results = tte_evaluation(train_parquet, test_parquet,task,model_name, save_dir, truncation,tte_cohort)

            metrics = {}
            metrics['C-Index']  = (float(np.mean(tte_results['C-Index'])),  float(np.std(tte_results['C-Index'])))
            metrics['Int-Brier'] = (float(np.mean(tte_results['Int-Brier'])), float(np.std(tte_results['Int-Brier'])))

            # os.makedirs(save_dir, exist_ok=True)
            save_file = save_dir / f"tte_{truncation}_days.json"
            with open(save_file, "w") as f:
                json.dump(metrics, f)
            print(f"Saved metrics to {save_file}")

            summary_dir = old_phenotype_dir / f"summary_table_{truncation}"
            os.makedirs(summary_dir,exist_ok=True)

            cidx_mean, cidx_std   = _metric_mean_std(metrics, "C-Index")
            ibr_mean,  ibr_std    = _metric_mean_std(metrics, "Int-Brier")

            if cidx_mean is not None:
                _update_metric_table(summary_dir, "cindex", task, str(subdir.name), cidx_mean, cidx_std)
            if ibr_mean is not None:
                _update_metric_table(summary_dir, "int_brier", task, str(subdir.name), ibr_mean, ibr_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated list of tasks, e.g., 'AMI,Celiac,MASLD,Ischemic_Stroke'")
    parser.add_argument("--old_phenotype_path", type=str, required=True,
                        help="Base directory path containing 'results/<task>/*/features_with_label/*.parquet'")
    parser.add_argument("--truncation", type=float, required=False, default="365",
                        help="truncation days")
    parser.add_argument("--tte_cohort", type=str, required=False, default=None,
                        help="path to tte cohort")
    parser.add_argument("--model_name", type=str, required=False, default=None,
                        help="model")
    args = parser.parse_args()

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(tasks_list)
    run(tasks_list, args.old_phenotype_path,args.truncation,args.tte_cohort,args.model_name)

'''
python tte_grid_search_new_time_motor_new.py \
    --tasks ami,masld,stroke \
    --truncation 300 \
    --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task/results \
    --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/tte \
    --model_name mamba_mtpp_2e-4_730,mamba_tpp_2e-4_730,transformer_ntp_730,transformer_tpp_730

python tte_grid_search_new_time_motor_new.py \
    --tasks AMI \
    --truncation 365 \
    --old_phenotype_path /data/results/zj2398/femr/cumc/phenotype/results \
    --tte_cohort /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_tte/tte/
Celiac,Ischemic_Stroke,MASLD
'''