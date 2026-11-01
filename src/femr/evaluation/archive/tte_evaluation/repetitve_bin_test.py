from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import os
import argparse

np.random.seed(42)
torch.manual_seed(42)

def tte_evaluation(train_set, test_set, task, model_name):
    # codes for using the first year labels
    print(f"task is {task}, model is {model_name} initial length {len(train_set)},{len(test_set)}")
    train_set = train_set[train_set["time_to_event"] <= 330]
    test_set  = test_set [test_set ["time_to_event"] <= 330]
    print(f"after length {len(train_set)},{len(test_set)}")

    val = train_set.sample(frac=0.1, replace=False, random_state=42)
    train_set = train_set.drop(val.index)

    # Transform labels
    num_durations = 100
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time_to_event'].values, df['tte_label'].values)

    # Extract features and labels
    x_train = np.stack(train_set.features.values)
    y_train = labtrans.fit_transform(*get_target(train_set))
    # train = (x_train, y_train)

    cuts = labtrans.cuts  # bin edges used for discretization

    print("\n[Discretization bins]")
    print(f"Number of cuts: {len(cuts)}")

    # Print each bin interval
    for i in range(len(cuts) - 1):
        left = cuts[i]
        right = cuts[i + 1]
        print(f"Bin {i:03d}: [{left:.3f}, {right:.3f})")

    # Check for repeated bin edges
    unique_cuts, counts = np.unique(cuts, return_counts=True)
    dup_edges = unique_cuts[counts > 1]

    if len(dup_edges) > 0:
        print("\n[Warning] Found repeated bin edges:")
        for edge in dup_edges:
            print(f"  edge = {edge}")
    else:
        print("\nAll bin edges are unique.")

    # Optional: check strict monotonicity
    if np.all(np.diff(cuts) > 0):
        print("Bin edges are strictly increasing.")
    else:
        print("Bin edges are NOT strictly increasing!")

    # x_val = np.stack(val.features.values)
    # y_val = labtrans.transform(*get_target(val))
    
    # val = (x_val, y_val)



    # x_test = np.stack(test_set.features.values)
    # durations_test, events_test = get_target(test_set)

    # # Define one layer NN for Deephit
    # in_features = x_train.shape[1]
    # out_features = labtrans.out_features
    # net = tt.practical.MLPVanilla(in_features, [], out_features)

    # # Train Deephit
    # model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
    # epochs, batch = 100, 2048
    # _ = model.fit(x_train, y_train, batch, epochs, [tt.callbacks.EarlyStopping()],
    #               val_data=val)

    # # Predict on test set
    # surv = model.predict_surv_df(x_test)

    # # Compute bootstrapped performance
    # results = {'C-Index': [], 'Int-Brier': []}
    # time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    # for _ in tqdm(range(100)):
    #     sample = np.random.choice(len(x_test), len(x_test))
    #     # surv has times as rows and individuals as columns; sample columns
    #     ev = EvalSurv(surv.iloc[:, sample], durations_test[sample], events_test[sample], censor_surv='km')
    #     results['C-Index'].append(ev.concordance_td('antolini'))
    #     results['Int-Brier'].append(ev.integrated_brier_score(time_grid))

    # # Display
    # for metric in results:
    #     print('{}: {:.3f} ({:.3f})'.format(metric, np.mean(results[metric]), np.std(results[metric])))
    # return results

    return None


def run(tasks, old_phenotype_path_str):
    old_phenotype_dir = Path(old_phenotype_path_str)

    for task in tasks:
        old_phenotype_path = old_phenotype_dir / "results" / task
        if not old_phenotype_path.exists():
            print(f"Skip task {task}: {old_phenotype_path} not found.")
            continue
        
        # the subdir includes pregenerated embeddings for all models: motor, tpp, mtpp
        for subdir in old_phenotype_path.iterdir():
            if not subdir.is_dir():
                continue
            train_path = subdir / "features_with_label" / "train.parquet"
            test_path  = subdir / "features_with_label" / "test.parquet"
            if not (train_path.exists() and test_path.exists()):
                print(f"Missing train/test parquet under {subdir}, skipping.")
                continue

            train_parquet = pd.read_parquet(train_path)
            test_parquet  = pd.read_parquet(test_path)

            tte_results = tte_evaluation(train_parquet, test_parquet, task, subdir.name)
            break

            # metrics = {}
            # metrics['C-Index']  = (float(np.mean(tte_results['C-Index'])),  float(np.std(tte_results['C-Index'])))
            # metrics['Int-Brier'] = (float(np.mean(tte_results['Int-Brier'])), float(np.std(tte_results['Int-Brier'])))

            # save_dir = old_phenotype_dir / "debug_results" / task / subdir.name
            # os.makedirs(save_dir, exist_ok=True)
            # save_file = save_dir / "tte.json"
            # with open(save_file, "w") as f:
            #     json.dump(metrics, f)
            # print(f"Saved metrics to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated list of tasks, e.g., 'AMI,Celiac,MASLD,Ischemic_Stroke'")
    parser.add_argument("--old_phenotype_path", type=str, required=True,
                        help="Base directory path containing 'results/<task>/*/features_with_label/*.parquet'")
    args = parser.parse_args()

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run(tasks_list, args.old_phenotype_path)

'''
command:
python 1st_y_evaluation.py \
    --tasks celiac,masld,stroke \
    --old_phenotype_path /shared/share_mala/zj2398/mimic/task/old_phenotype

python 1st_y_evaluation.py \
    --tasks ami,masld,stroke \
    --old_phenotype_path /shared/share_mala/zj2398/mimic/task/phenotype_task
'''