import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from pycox.models import DeepHitSingle
import torchtuples as tt
from utils import (
    generate_bootstrap_indices, bootstrap_cindex_parallel,
    make_structured_array, save_survival_results,
)
from pathlib import Path
import os
import argparse
import sys

import numpy as np, torch, random                                                                                 
np.random.seed(42)                                                                                              
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)   

# def load_features_labels(feature_path, label_path, split_path):
#     data = pickle.load(open(feature_path, "rb"))
#     labels = (
#         pd.read_parquet(label_path)
#         .set_index(["subject_id", "prediction_time"])
#     )

#     X = pd.DataFrame(
#         data["features"].todense(),
#         index=pd.MultiIndex.from_arrays(
#             [data["subject_ids"], data["feature_times"]],
#             names=["subject_id", "prediction_time"],
#         ),
#     )

#     X = X.loc[labels.index]
#     X, labels = remove_duplicates(X, labels)

#     split = pd.read_csv(split_path, index_col='subject_id')
#     labels = labels.join(split)

#     val = labels[labels.split_name == 'train'].index.to_series().sample(frac=0.2, replace=False, random_state=42)

def check_cohort(task,old_phenotype_path,mtpp,motor,llama,tte_cohort_path):
    tte_cohort = pd.read_parquet(tte_cohort_path).rename(columns={"time_to_event_days": "time_to_event"}, errors="ignore")
    keys = tte_cohort[["subject_id", "prediction_time", "time_to_event", "tte_label"]].drop_duplicates()

    mtpp_train = pd.read_parquet(old_phenotype_path/mtpp/"features_with_label"/"train.parquet").merge(keys, on=["subject_id", "prediction_time"], how="inner").sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    mtpp_test = pd.read_parquet(old_phenotype_path/mtpp/"features_with_label"/"test.parquet").merge(keys, on=["subject_id", "prediction_time"], how="inner").sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    llama_train = pd.read_parquet(old_phenotype_path/llama/"features_with_label"/"train.parquet").merge(keys, on=["subject_id", "prediction_time"], how="inner").sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    llama_test = pd.read_parquet(old_phenotype_path/llama/"features_with_label"/"test.parquet").merge(keys, on=["subject_id", "prediction_time"], how="inner").sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    motor_train = pd.read_parquet(old_phenotype_path/motor/"features_with_label"/"train.parquet").merge(keys, on=["subject_id", "prediction_time"], how="inner").sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    motor_test = pd.read_parquet(old_phenotype_path/motor/"features_with_label"/"test.parquet").merge(keys, on=["subject_id", "prediction_time"], how="inner").sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    # deephit = pd.read_parquet(f"/shared/share_mala/zj2398/mimic/task/phenotype_task/deephit/tte/{task}/features")
    key_cols = ["subject_id", "prediction_time"]
    assert mtpp_train[key_cols].equals(llama_train[key_cols]), f"[{task}] mtpp_train != llama_train"
    assert mtpp_train[key_cols].equals(motor_train[key_cols]), f"[{task}] mtpp_train != motor_train"
    assert mtpp_test[key_cols].equals(llama_test[key_cols]), f"[{task}] mtpp_test != llama_test"
    assert mtpp_test[key_cols].equals(motor_test[key_cols]), f"[{task}] mtpp_test != motor_test"
    print(f"[{task}] Cohort check passed: train={len(mtpp_train)}, test={len(mtpp_test)}")


# tte_evaluation(train_parquet, test_parquet,task,truncation,tte_cohort_task)

def tte_evaluation(train_set_init, test_set_init, truncation, tte_cohort_path,
                   save_path=None,model_name="model", 
                   n_bootstrap=10000, seed=42, n_jobs=16):
    """
    Returns: (results_dict, best_model)
    results_dict contains 'C-Index', 'test_order' for paired comparison.
    """
    tte_cohort = pd.read_parquet(tte_cohort_path).rename(columns={"time_to_event_days": "time_to_event"}, errors="ignore")
    keys = tte_cohort[["subject_id", "prediction_time", "time_to_event", "tte_label"]].drop_duplicates()

    train_set = train_set_init.merge(keys, on=["subject_id", "prediction_time"], how="inner")
    test_set = test_set_init.merge(keys, on=["subject_id", "prediction_time"], how="inner")

    # === CRITICAL: deterministic sort so bootstrap indices align across models ===
    train_set = train_set.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    test_set = test_set.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    # print(f"Train: {len(train_set_init)} -> {len(train_set)}, Test: {len(test_set_init)} -> {len(test_set)}")
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")

    # Record test order for alignment assertion later
    test_order = list(zip(
        test_set["subject_id"].values.tolist(),
        test_set["prediction_time"].astype(str).values.tolist(),
    ))

    val = train_set.sample(frac=0.2, replace=False, random_state=42)
    train_set = train_set.drop(val.index)
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    val = val.drop(hyper.index)

    num_durations = 100
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df["time_to_event"].values, df["tte_label"].values)

    x_train = np.stack(train_set.features.values).astype(np.float32)          
    y_train = labtrans.fit_transform(*get_target(train_set))
    durations_train, events_train = get_target(train_set)

    x_val = np.stack(val.features.values).astype(np.float32)          
    y_val = labtrans.transform(*get_target(val))

    x_hyper = np.stack(hyper.features.values).astype(np.float32)          
    y_hyper = labtrans.transform(*get_target(hyper))

    x_test = np.stack(test_set.features.values).astype(np.float32)          
    durations_test, events_test = get_target(test_set)

    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    grid = ParameterGrid({
        "dropout": [0, 0.1],
        "lr": [0.001, 0.0001],
        "batch": [1024, 2048],
    })

    best_nll = np.inf
    best_model = None
    best_params = None
    for param in grid:
        net = tt.practical.MLPVanilla(in_features, [], out_features, True, param["dropout"])
        model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
        model.optimizer.set_lr(param["lr"])
        _ = model.fit(x_train, y_train, param["batch"], 1000,
                      [tt.callbacks.EarlyStopping()], val_data=(x_val, y_val))
        nll = model.score_in_batches(x_hyper, y_hyper)["loss"]
        if nll < best_nll:
            best_model, best_nll, best_params = model, nll, param

    print(f"Best params: {best_params}, NLL: {best_nll:.4f}")

    surv = best_model.predict_surv_df(x_test)
    et_train = make_structured_array(durations_train, events_train)
    et_test = make_structured_array(durations_test, events_test)

    bootstrap_indices = generate_bootstrap_indices(len(et_test), n_bootstrap, seed)
    c_indices = bootstrap_cindex_parallel(surv, et_train, et_test, truncation,
                                          bootstrap_indices, n_jobs)

    results = {
        "C-Index": c_indices,
        "test_order": test_order,
        "model_name": model_name,
        "best_params": best_params,
    }

    mean_val, std_val = np.mean(c_indices), np.std(c_indices)
    print(f"[{model_name}] C-Index: {mean_val:.4f} (±{std_val:.4f})")

    if save_path:
        save_survival_results(results, best_model, save_path,truncation)

    return results, best_model


def run(tasks, old_phenotype_path_str,truncation,model_name,tte_cohort_path):
    old_phenotype_dir = Path(old_phenotype_path_str)
  
    # for task in tasks:
    #     tte_cohort_task = Path(tte_cohort_path)/task/f"{task}.parquet" 
    #     old_phenotype_path = old_phenotype_dir / task
    #     check_cohort(task,old_phenotype_path,"transformer_mtpp_730","motor_730","cc-llama",tte_cohort_task)
    # return 
    for task in tasks:
        tte_cohort_task = Path(tte_cohort_path)/task/f"{task}.parquet" 
        # old_phenotype_path = old_phenotype_dir / "results" / task
        old_phenotype_path = old_phenotype_dir / task
        if not old_phenotype_path.exists():
            print(f"Skip task {task}: {old_phenotype_path} not found.")
            continue
        save_dir = old_phenotype_path / model_name
        train_path = save_dir  / "features_with_label" / "train.parquet"
        test_path  = save_dir  / "features_with_label" / "test.parquet"
        if not (train_path.exists() and test_path.exists()):
            print(f"Missing train/test parquet under {old_phenotype_path}, skipping.")
            continue
        train_parquet = pd.read_parquet(train_path)
        test_parquet  = pd.read_parquet(test_path)

        tte_results,best_model = tte_evaluation(train_parquet, test_parquet,truncation,tte_cohort_task,save_path=save_dir)

        # metrics = {}
        # metrics['C-Index']  = (float(np.mean(tte_results['C-Index'])),  float(np.std(tte_results['C-Index'])))
        # metrics['Int-Brier'] = (float(np.mean(tte_results['Int-Brier'])), float(np.std(tte_results['Int-Brier'])))

        # os.makedirs(save_dir, exist_ok=True)

        # save_file = save_dir / f"tte_{args.output_name}_{truncation}_days.json"
        # with open(save_file, "w") as f:
        #     json.dump(metrics, f)
        # print(f"Saved metrics to {save_file}")

            # summary_dir = old_phenotype_dir / "results" / f"summary_table_{args.output_name}"
            # os.makedirs(summary_dir,exist_ok=True)

            # cidx_mean, cidx_std   = _metric_mean_std(metrics, "C-Index")

            # if cidx_mean is not None:
            #     _update_metric_table(summary_dir, "cindex", task, str(subdir.name), cidx_mean, cidx_std)
            # if ibr_mean is not None:
            #     _update_metric_table(summary_dir, "int_brier", task, str(subdir.name), ibr_mean, ibr_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated list of tasks, e.g., 'AMI,Celiac,MASLD,Ischemic_Stroke'")
    parser.add_argument("--old_phenotype_path", type=str, required=True,
                        help="Base directory path containing 'results/<task>/*/features_with_label/*.parquet'")
    parser.add_argument("--truncation", type=float, required=False, default=365,
                        help="truncation days")
    parser.add_argument("--model_name", type=str, required=False, default="",
                        help="model_name")
    parser.add_argument("--tte_cohort", type=str, required=False, default=None,
                        help="path to tte cohort")
    args = parser.parse_args()

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(tasks_list)
    run(tasks_list, args.old_phenotype_path,args.truncation,args.model_name,args.tte_cohort)

# export CUDA_VISIBLE_DEVICES=1

# python tte_test.py \
#     --tasks ami,masld,stroke  \
#     --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task/results \
#     --truncation 300 \
#     --model_name transformer_mtpp_730 \
#     --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/tte

# python tte_test.py \
#     --tasks ami,masld,stroke  \
#     --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task/results \
#     --truncation 300 \
#     --model_name transformer_ntp_730 \
#     --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/tte

# python tte_test.py \
#     --tasks ami,masld,stroke  \
#     --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task/results \
#     --truncation 300 \
#     --model_name motor_730 \
#     --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/tte

# python tte_test.py \
#     --tasks ami,masld,stroke  \
#     --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task/results \
#     --truncation 300 \
#     --model_name cc-llama \
#     --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/tte