"""Shared paths / cohort definitions for the history-length stratified analysis.

Reviewer response: ORA's Eq. 2 supervises later events more often than earlier ones.
We stratify held-out prediction points by how many events precede them and check whether
ORA's gain over each baseline is uniform across strata.
"""

import glob
import os

MEDS_READER = "/shared/share_mala/zj2398/mimic/data/meds_v0.6_reader"
MEDS_PARQUET = "/shared/share_mala/zj2398/mimic/data/mimic-3.1-meds/MEDS_cohort/data/held_out"

OUT_ROOT = "/shared/share_mala/zj2398/mimic/task/stratified_history"
COUNT_DIR = os.path.join(OUT_ROOT, "history_counts")

OUTCOME_ROOT = "/shared/share_mala/zj2398/mimic/task/patient_outcome_task"
PHENOTYPE_ROOT = "/shared/share_mala/zj2398/mimic/task/phenotype_task"
REGRESSION_ROOT = "/shared/share_mala/zj2398/mimic/task/regression"

OUTCOME_TASKS = ["in_hospital_mortality", "long_los", "readmission"]
PHENOTYPE_TASKS = ["ami", "masld", "stroke"]
REGRESSION_TASKS = ["creatinine", "glucose", "pao2", "platelets"]

# Only events within this window before the prediction time are counted for phenotype
# cohorts, matching the 730-day observation window those models were run with.
PHENOTYPE_WINDOW_DAYS = 730

TTE_TRUNCATION = 300.0

# Display name -> per-family directory name. `transformer_mtpp` is ORA.
ORA = "transformer_mtpp"
MODELS = ["cc_llama", "icarefm", "motor", "transformer_survivehr", "transformer_mtpp"]
BASELINES = [m for m in MODELS if m != ORA]


def outcome_predictions(task: str, model: str) -> str:
    """Held-out predictions for the binary patient-outcome tasks."""
    base = f"{OUTCOME_ROOT}/results/{task}"
    if model == "cc_llama":
        # The cc-llama run subdirectory differs per task (llama_298453, llama_298449, ...).
        hits = sorted(glob.glob(f"{base}/cc-llama/*/test_predictions/predictions.parquet"))
        assert len(hits) == 1, f"expected exactly one cc-llama run for {task}, got {hits}"
        return hits[0]
    return f"{base}/{model}/test_predictions/predictions.parquet"


def phenotype_dir(task: str, model: str) -> str:
    """Result directory for a phenotype model (binary predictions and TTE artifacts)."""
    base = f"{PHENOTYPE_ROOT}/results/{task}"
    if model == "cc_llama":
        hits = sorted(d for d in glob.glob(f"{base}/cc-llama/*") if os.path.isdir(d))
        hits = [d for d in hits if os.path.basename(d).startswith("llama_")]
        assert len(hits) == 1, f"expected exactly one cc-llama run for {task}, got {hits}"
        return hits[0]
    return f"{base}/{model}_730"


def phenotype_predictions(task: str, model: str) -> str:
    return f"{phenotype_dir(task, model)}/test_predictions/predictions.parquet"


def phenotype_tte_dir(task: str, model: str) -> str:
    """TTE artifacts live one level up from the cc-llama run subdirectory."""
    if model == "cc_llama":
        return os.path.dirname(phenotype_dir(task, model))
    return phenotype_dir(task, model)


def regression_predictions(task: str, model: str) -> str:
    """The 50k-row evaluation set.

    `test_predictions_50k.parquet` exists only for motor and transformer_mtpp; for the
    other models the plain file already *is* that evaluation set. Preferring the _50k
    file when present makes all five models share identical keys per task and reproduces
    the published R^2 in regression/summary_dir/r2.csv.
    """
    d = f"{REGRESSION_ROOT}/{model}/results/{task}/{model}"
    fifty = f"{d}/test_predictions_50k.parquet"
    return fifty if os.path.exists(fifty) else f"{d}/test_predictions.parquet"


# family -> (tasks, prediction-file resolver, observation window in days or None)
FAMILIES = {
    "outcome": (OUTCOME_TASKS, outcome_predictions, None),
    "phenotype_binary": (PHENOTYPE_TASKS, phenotype_predictions, PHENOTYPE_WINDOW_DAYS),
    "phenotype_tte": (PHENOTYPE_TASKS, None, PHENOTYPE_WINDOW_DAYS),
    "regression": (REGRESSION_TASKS, regression_predictions, None),
}


def cohort_keys_path(family: str, task: str) -> str:
    """Cohort file that defines the held-out prediction points for a family/task."""
    if family == "outcome":
        return f"{OUTCOME_ROOT}/cohort/{task}/held_out.parquet"
    if family == "phenotype_binary":
        return f"{PHENOTYPE_ROOT}/cohort/binary/{task}/held_out.parquet"
    if family == "phenotype_tte":
        return f"{PHENOTYPE_ROOT}/cohort/tte/{task}/held_out.parquet"
    if family == "regression":
        # The regression cohort parquet holds every split, so take the keys from the
        # evaluation file itself (identical across all five models).
        return regression_predictions(task, "cc_llama")
    raise ValueError(family)


def counts_path(family: str, task: str) -> str:
    return os.path.join(COUNT_DIR, family, f"{task}.parquet")
