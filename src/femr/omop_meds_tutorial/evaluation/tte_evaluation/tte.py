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

def tte_evaluation(train_set, test_set):
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
    x_train = np.stack(train_set.features.values)

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
    # print(f"surv is {surv}")
    truncation = 365

    # times represent all discretized intervals
    times = surv.index.to_numpy()

    # the index closest to 5 yeats
    index_truncation = np.argmin(np.abs(times - truncation))

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