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

np.random.seed(42)
torch.manual_seed(42)

# this is debugged codes 
def tte_evaluation(train_set, test_set,task,truncation=365,tte_cohort_path=None):
    # Extract validation set to stop loss
    if tte_cohort_path is not None:
        tte_cohort_path = Path(tte_cohort_path)/task/f"{task}.parquet"
        print(f"reading cohort from {tte_cohort_path}")
        if tte_cohort_path.exists():
            tte_cohort = pd.read_parquet(tte_cohort_path)
        else:
            task_dir = tte_cohort_path.parent
            parquet_files = list(task_dir.glob("*.parquet"))
            tte_cohort = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        tte_cohort =  tte_cohort.rename(columns={"time_to_event_days": "time_to_event"}, errors="ignore")
        print(tte_cohort)
        keys = tte_cohort[["subject_id", "prediction_time",'time_to_event',"tte_label"]].drop_duplicates()
        print(f"before train set {len(train_set)}, test_set {len(test_set)}")
        train_set = train_set.merge(
            keys, on=["subject_id", "prediction_time"], how="inner"
        )

        test_set = test_set.merge(
            keys, on=["subject_id", "prediction_time"], how="inner"
        )
        print(f"after train set {len(train_set)}, test_set {len(test_set)}")
    print(f"truncation is {truncation}")
    print(train_set)
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

    x_hyper = np.stack(hyper.features.values).astype(np.float32) 
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
    # grid = ParameterGrid({
    #     'dropout': [0],
    #     'lr': [0.001],
    #     'batch': [1024]
    # })


    # Hyperparameter selection
    best_nll = np.inf
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
    

    # Evaluation best model
    surv = best_model.predict_surv_df(x_test)
    # truncation = 5 * 365

    # times represent all discretized intervals
    times = surv.index.to_numpy()

    # the index closest to truncation days
    index_truncation = np.argmin(np.abs(times - truncation))

    # Create mask for time points within 5 years (for integrated Brier score)
    # mask_5yr = times <= truncation
    # times_5yr = times[mask_5yr]  # Time grid from 0 to ~5 years
    # print(f"times_5yr is {times_5yr}")
    results = {'C-Index': [], 'Int-Brier': []}

    # Create structured arrays with filtered data
    # Create format for sksurv
    et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                    dtype=[('e', bool), ('t', float)])
    et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                    dtype=[('e', bool), ('t', float)])
    
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
        
        # =======================================================================
        # C-INDEX: Use risk scores at 5 years
        # =======================================================================
        surv_at_5yr = surv.iloc[index_truncation, sample]
        risk_at_5yr = 1 - surv_at_5yr
        
        c_index = concordance_index_ipcw(
            et_train,
            et_test[sample],
            risk_at_5yr,
            tau=truncation
        )
        results['C-Index'].append(c_index[0])
        
        # =======================================================================
        # INTEGRATED BRIER SCORE: Filter time points to valid range
        # =======================================================================
        
        # Get the observed time range for this bootstrap sample
        # print(f"times_eval is {times_eval}")
         
        
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

def run(tasks, old_phenotype_path_str,output_name,truncation,tte_cohort):
    old_phenotype_dir = Path(old_phenotype_path_str)
    # phenotype_label_dir = Path("/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_tte")
    # key_cols = ["subject_id", "prediction_time", "tte_label", "time_to_event"]
    for subdir in old_phenotype_dir.iterdir():
        if not subdir.is_dir():
            continue       
        for task in tasks:
            # old_phenotype_path = old_phenotype_dir / "results" / task
            old_phenotype_path = subdir / task
            if not old_phenotype_path.exists():
                print(f"Skip task {task}: {old_phenotype_path} not found.")
                continue
            # if tte_cohort is not None:
            #     tte_cohort = Path(tte_cohort)/task/f"{task}.parquet"
            # the subdir includes pregenerated embeddings for all models: motor, tpp, mtpp

            # for subdir in old_phenotype_path.iterdir():
                # if subdir.name not in ["ami", "masld", "stroke"]:
                #     print(f"not correct model {subdir.name}, continue")
                #     continue
            # if not subdir.is_dir():
            #     continue
            train_path = old_phenotype_path / "features_with_label" / "train.parquet"
            test_path  = old_phenotype_path / "features_with_label" / "test.parquet"
            if not (train_path.exists() and test_path.exists()):
                print(f"Missing train/test parquet under {old_phenotype_path}, skipping.")
                continue
            train_parquet = pd.read_parquet(train_path)
            test_parquet  = pd.read_parquet(test_path)

            tte_results = tte_evaluation(train_parquet, test_parquet,task,truncation,tte_cohort)

            metrics = {}
            metrics['C-Index']  = (float(np.mean(tte_results['C-Index'])),  float(np.std(tte_results['C-Index'])))
            metrics['Int-Brier'] = (float(np.mean(tte_results['Int-Brier'])), float(np.std(tte_results['Int-Brier'])))

            save_dir = old_phenotype_dir / "results" / task 
            os.makedirs(save_dir, exist_ok=True)
            save_file = save_dir / f"tte_{args.output_name}_{truncation}_days.json"
            with open(save_file, "w") as f:
                json.dump(metrics, f)
            print(f"Saved metrics to {save_file}")

            summary_dir = old_phenotype_dir / "results" / f"summary_table_{args.output_name}"
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
    parser.add_argument("--output_name", type=str, required=False, default="",
                        help="output_name")
    parser.add_argument("--tte_cohort", type=str, required=False, default=None,
                        help="path to tte cohort")
    args = parser.parse_args()

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(tasks_list)
    run(tasks_list, args.old_phenotype_path,args.output_name,args.truncation,args.tte_cohort)

'''
command:
python tte_grid_search_filter.py \
    --tasks stroke \
    --truncation 330 \
    --output_name first_event \
    --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task/ \
    --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype/tte_first

    
python tte_grid_search_filter.py \
    --tasks Celiac,Ischemic_Stroke \
    --old_phenotype_path /data/results/zj2398/femr/cumc_0.1/phenotype

fomoh

python tte_grid_search_filter_fomoh.py \
    --tasks AMI,Celiac,Ischemic_Stroke,MASLD \
    --truncation 365 \
    --output_name "" \
    --old_phenotype_path /data/results/zj2398/fomoh/cumc/linear_probing/ \
    --tte_cohort /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_tte/tte/

'''