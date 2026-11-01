"""Shared bootstrap, parallel C-index, and paired hypothesis testing utilities."""

import json
import numpy as np
import pickle
from joblib import Parallel, delayed
from sksurv.metrics import concordance_index_ipcw


def generate_bootstrap_indices(n_test, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    return np.array([rng.choice(n_test, n_test) for _ in range(n_bootstrap)])


def _compute_cindex_batch(indices_batch, surv, tau_idx, et_train, et_test, tau):
    results = []
    for sample in indices_batch:
        risk = 1.0 - surv.iloc[tau_idx, sample].values
        c = concordance_index_ipcw(et_train, et_test[sample], risk, tau=tau)[0]
        results.append(c)
    return results


def bootstrap_cindex_parallel(surv_df, et_train, et_test, truncation,
                              bootstrap_indices, n_jobs=16):
    times = surv_df.index.to_numpy()
    tau_idx = np.argmin(np.abs(times - truncation))

    chunks = np.array_split(bootstrap_indices, n_jobs)
    chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_cindex_batch)(chunk, surv_df, tau_idx, et_train, et_test, truncation)
        for chunk in chunks
    )
    return [c for chunk in chunk_results for c in chunk]


def make_structured_array(durations, events):
    return np.array(
        list(zip(events.astype(bool), durations.astype(float))),
        dtype=[("e", bool), ("t", float)],
    )


def save_survival_results(results, model, dir,truncation):
    """
    results dict should contain:
      - 'C-Index': list of bootstrap c-indices
      - 'test_order': list of (subject_id, prediction_time) tuples
      - 'model_name': str
    """
    from pathlib import Path as _Path
    dir = _Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    results_path = dir / f"tte_results_{truncation}.pkl"
    model_path = dir / f"tte_model_{truncation}.pt"
    metrics_path = dir / f"tte_metrics_{truncation}.json"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    model.save_net(str(model_path))
    c_indices = np.asarray(results["C-Index"])
    metrics = {
        "mean": float(np.mean(c_indices)),
        "std": float(np.std(c_indices)),
        "truncation": float(truncation),
        "n_bootstrap": int(len(c_indices)),
        "model_name": results.get("model_name"),
        "best_params": results.get("best_params"),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved results to {results_path}, model to {model_path}, metrics to {metrics_path}")


def load_survival_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def paired_bootstrap_test(results_a, results_b, name_a="A", name_b="B"):
    """One-sided test: H1: C_B > C_A. Both must share bootstrap indices."""
    # Assert same patient ordering
    assert results_a["test_order"] == results_b["test_order"], \
        "Test set ordering mismatch — cannot do paired comparison"

    ca = np.array(results_a["C-Index"])
    cb = np.array(results_b["C-Index"])
    delta = cb - ca

    mean_delta = np.mean(delta)
    ci_lo, ci_hi = np.percentile(delta, [2.5, 97.5])
    p_value = np.mean(delta <= 0)

    print(f"\n{name_b} vs {name_a}:")
    print(f"  ΔC-index (B−A): {mean_delta:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  p-value (one-sided, H1: B>A): {p_value:.4f}")
    return {"mean_delta": mean_delta, "ci": (ci_lo, ci_hi), "p_value": p_value}