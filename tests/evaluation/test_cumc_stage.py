"""The CUMC batch stage and baseline options, and that they leave the MIMIC commands unchanged."""
import datetime
import hashlib
import os
import pathlib
import pickle
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import meds
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import femr.models.architecture.embedding as embedding
from femr.evaluation.baseline.meds_tab.generate_tabular_features import sort_labels
from femr.evaluation.baseline.meds_tab.meds_tab_featurizer import MedsTabNumericFeaturizer
from femr.evaluation.common import preflight

SCRIPT = pathlib.Path(embedding.__file__).parents[2] / "evaluation" / "script"
RUN_ALL = SCRIPT.parent / "baseline" / "meds_tab" / "run_all.py"


def bash(env_name: str, body: str, tmp_path, **env) -> str:
    """Load lib.sh with one env file and run `body` in it; returns what it prints."""
    code = f"source {SCRIPT}/lib.sh {env_name}\n{body}\n"
    result = subprocess.run(["bash", "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path,
                            env={**os.environ, "RESULT_ROOT": str(tmp_path / "results"), **env})
    return result.stdout


def test_batch_dir_keeps_the_key_of_existing_batches(tmp_path):
    """A different key would silently orphan the batches already saved for MIMIC."""
    (tmp_path / "tok").mkdir()
    (tmp_path / "tok" / "dictionary.msgpack").write_bytes(b"tokenizer")
    (tmp_path / "ontology.pkl").write_bytes(b"ontology")
    cohort = tmp_path / "masld"
    cohort.mkdir()
    (cohort / "held_out.parquet").write_bytes(b"test rows")
    (cohort / "train.parquet").write_bytes(b"train rows")

    out = bash("mimic_gsb",
               f'ARCH=transformer MODEL_PATH={tmp_path}/tok\nbatch_dir_of phenotype masld {cohort} 730',
               tmp_path, MEDS_READER="/db", META_DIR=str(tmp_path), HF_CACHE="/cache")

    content = b"/db transformer 65536 8 730\n" + b"tokenizer" + b"ontology" + b"test rows" + b"train rows"
    assert out.strip() == f"/cache/eval_batches/phenotype_masld_{hashlib.md5(content).hexdigest()[:16]}"


# python is replaced by a function that prints its arguments, so nothing is trained.
PRINT_RUN_ALL = 'python() { echo "$*"; }\ncollect() { :; }\nbaseline_main phenotype AMI'


def test_mimic_baseline_command_has_no_cumc_options(tmp_path):
    out = bash("mimic_gsb", PRINT_RUN_ALL, tmp_path)
    assert "run_all.py" in out
    for option in ("--deephit_grid", "--wd", "--tte_col"):
        assert option not in out


def test_cumc_baseline_command_selects_72_configs_and_its_time_column(tmp_path):
    out = bash("cumc_cumc", PRINT_RUN_ALL, tmp_path)
    assert "--deephit_grid full --wd 0,1e-4,1e-2 --tte_col time_to_event" in out


@pytest.mark.parametrize("args, column", [([], "time_to_event_days"),
                                          (["--tte_col", "time_to_event"], "time_to_event")])
def test_run_all_time_column(args, column, tmp_path, monkeypatch):
    # With empty task lists run_all.py only parses its arguments.
    argv = [str(RUN_ALL), "--phenotypes", "--binary_only", "--results_dir", str(tmp_path), *args]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.syspath_prepend(str(RUN_ALL.parent))
    assert runpy.run_path(str(RUN_ALL), run_name="__main__")["TTE_COL"] == column


def test_load_features_labels_sorts_the_columns_of_each_row(tmp_path, monkeypatch):
    """xgboost 3.2 on the GPU silently trains bad models on rows with unsorted columns."""
    monkeypatch.setattr(sys, "argv", [str(RUN_ALL), "--phenotypes", "--binary_only", "--results_dir", str(tmp_path)])
    monkeypatch.syspath_prepend(str(RUN_ALL.parent))
    load_features_labels = runpy.run_path(str(RUN_ALL), run_name="__main__")["load_features_labels"]

    # Every row stores column 2 before column 0, as CountFeaturizer does: first seen comes first.
    n = 10
    matrix = scipy.sparse.csr_matrix((np.tile([5.0, 7.0], n), np.tile([2, 0], n), np.arange(0, 2 * n + 1, 2)),
                                     shape=(n, 3))
    subjects, times = np.arange(n), pd.date_range("2020-01-01", periods=n)
    with open(tmp_path / "features.pkl", "wb") as f:
        pickle.dump({"features": matrix, "subject_ids": subjects, "feature_times": times.to_numpy()}, f)
    pd.DataFrame({"subject_id": subjects, "prediction_time": times, "boolean_value": True}).to_parquet(
        tmp_path / "labels.parquet")
    pd.DataFrame({"subject_id": subjects, "split_name": "train"}).to_csv(tmp_path / "split.csv", index=False)

    X, _ = load_features_labels(str(tmp_path / "features.pkl"), str(tmp_path / "labels.parquet"),
                                str(tmp_path / "split.csv"))

    assert X.has_sorted_indices
    assert X.toarray().tolist() == [[7.0, 0.0, 5.0]] * n


def test_build_batches_passes_the_workers_and_only_the_wanted_tasks(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "dictionary.msgpack").write_bytes(b"tokenizer")
    (tmp_path / "ontology.pkl").write_bytes(b"ontology")
    for task in ("AMI", "regression/glucose", "regression/bilirubin"):
        (tmp_path / task).mkdir(parents=True)
        (tmp_path / task / "labels.parquet").write_bytes(task.encode())
    env = dict(MEDS_READER="/db", META_DIR=str(tmp_path), HF_CACHE=str(tmp_path / "hf"), BATCH_WORKERS="3",
               PHENOTYPE_BINARY_COHORT=str(tmp_path), REGRESSION_COHORT=str(tmp_path / "regression"),
               REGRESSION_TASKS="glucose")
    stub = f'python() {{ echo "$*"; }}\nARCH=transformer MODEL_PATH={tmp_path}/tokenizer\n'

    phenotype = bash("cumc_cumc", stub + "build_batches phenotype AMI", tmp_path, **env)
    assert "--batches_only" in phenotype and "--num_proc 12" in phenotype
    assert "--task_type binary" in phenotype and "--observation_window 730" in phenotype

    regression = bash("cumc_cumc", stub + "build_batches regression", tmp_path, **env)
    assert "regression/glucose/" in regression and "bilirubin" not in regression
    assert "--task_type regression" in regression and "--observation_window" not in regression


def test_build_batches_skips_a_task_that_is_built(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "dictionary.msgpack").write_bytes(b"tokenizer")
    (tmp_path / "ontology.pkl").write_bytes(b"ontology")
    (tmp_path / "AMI").mkdir()
    (tmp_path / "AMI" / "labels.parquet").write_bytes(b"labels")
    env = dict(MEDS_READER="/db", META_DIR=str(tmp_path), HF_CACHE=str(tmp_path / "hf"),
               PHENOTYPE_BINARY_COHORT=str(tmp_path))
    setup = f"ARCH=transformer MODEL_PATH={tmp_path}/tokenizer\n"

    built = bash("cumc_cumc", setup + f'batch_dir_of phenotype AMI {tmp_path}/AMI 730', tmp_path, **env).strip()
    os.makedirs(built)
    out = bash("cumc_cumc", setup + 'python() { echo CALLED; }\nbuild_batches phenotype AMI', tmp_path, **env)
    assert "already built" in out and "CALLED" not in out


def run_preflight(root, monkeypatch, tte_col="time_to_event"):
    """Run preflight on the folders of write_cohorts; returns (exit code, problems)."""
    preflight.problems.clear()
    paths = dict(meta_dir="meta", models_root="models", result_root="results", hf_cache="hf",
                 meds_tab_dir="meds_tab", outcome_cohort="outcome", binary_cohort="binary",
                 tte_cohort="tte", regression_cohort="regression")
    argv = ["preflight", "--regression_tasks", "glucose", "--tte_col", tte_col]
    for name, folder in paths.items():
        argv += [f"--{name}", str(root / folder)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        preflight.main()
    return stopped.value.code, list(preflight.problems)


def write_cohorts(root, time_col="time_to_event"):
    """The smallest folders that look like CUMC: subject 1 is in train, subject 2 in test."""
    labels = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "prediction_time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
        "boolean_value": [True, False, False],
        "tte_label": [True, False, False],
        time_col: [10.0, 20.0, 30.0],
        "numerical_value": [1.0, 2.0, 3.0],
    })
    for folder, name in (("outcome/death", "train"), ("binary/AMI", "train"), ("tte/AMI", "train"),
                         ("regression/glucose", "glucose")):
        (root / folder).mkdir(parents=True)
        labels.to_parquet(root / folder / f"{name}.parquet")
    (root / "meta" / "tokenizer").mkdir(parents=True)
    (root / "meta" / "tokenizer" / "dictionary.msgpack").write_bytes(b"tokenizer")
    (root / "meta" / "ontology.pkl").write_bytes(b"ontology")
    pd.DataFrame({"subject_id": [1, 2], "split_name": ["train", "test"]}).to_csv(
        root / "meta" / "main_split.csv", index=False)
    for folder in ("models", "results", "meds_tab"):
        (root / folder).mkdir()


def test_preflight_accepts_cumc_labels(tmp_path, monkeypatch):
    write_cohorts(tmp_path)
    assert run_preflight(tmp_path, monkeypatch) == (0, [])


def test_preflight_accepts_labels_in_any_order(tmp_path, monkeypatch):
    """Regression cohorts are randomly sampled, and every pipeline handles the order itself."""
    write_cohorts(tmp_path)
    for path in ("regression/glucose/glucose.parquet", "tte/AMI/train.parquet", "binary/AMI/train.parquet"):
        shuffled = pd.read_parquet(tmp_path / path).iloc[[1, 2, 0]]
        shuffled.to_parquet(tmp_path / path)
    assert run_preflight(tmp_path, monkeypatch) == (0, [])


def test_meds_tab_labels_see_no_future_events():
    """Out of order, the Jan 3 label also counted the events of Jan 5 and Jan 9, with no error."""
    day = lambda d: datetime.datetime(2020, 1, d)
    events = [SimpleNamespace(time=day(d), code="glucose", numeric_value=100.0) for d in (1, 5, 9, 13)]
    subject = SimpleNamespace(subject_id=1, events=events)
    featurizer = MedsTabNumericFeaturizer(windows_days=(None,), min_support=0)
    featurizer.encorperate_prepreprocessed_data([{"glucose": 4}])
    labels = pd.DataFrame({"subject_id": [2, 1, 1], "prediction_time": [day(1), day(11), day(3)]})

    ordered = sort_labels(labels)

    assert list(ordered.index) == [0, 1, 2]  # map_with_data looks rows up by index label
    assert ordered["subject_id"].tolist() == [1, 1, 2]
    rows = featurizer.featurize(subject, list(ordered[ordered.subject_id == 1].itertuples(index=False)))
    assert [row[0].value for row in rows] == [1, 3]  # code/count at Jan 3 and Jan 11


def test_preflight_reports_a_missing_time_column_instead_of_crashing(tmp_path, monkeypatch):
    write_cohorts(tmp_path, time_col="time_to_event_days")
    code, problems = run_preflight(tmp_path, monkeypatch)
    assert code == 1 and len(problems) == 1 and "no column time_to_event" in problems[0]


def test_preflight_reports_labels_that_would_be_read_twice(tmp_path, monkeypatch):
    write_cohorts(tmp_path)
    shutil.copy(tmp_path / "binary/AMI/train.parquet", tmp_path / "binary/AMI/AMI.parquet")
    code, problems = run_preflight(tmp_path, monkeypatch)
    assert code == 1 and "read twice" in problems[0]


def test_preflight_reports_a_split_without_test_rows(tmp_path, monkeypatch):
    write_cohorts(tmp_path)
    pd.DataFrame({"subject_id": [1, 2], "split_name": ["train", "held_out"]}).to_csv(
        tmp_path / "meta" / "main_split.csv", index=False)
    code, problems = run_preflight(tmp_path, monkeypatch)
    assert code == 1 and "split_name must be train/test" in problems[0]


def test_batches_only_builds_batches_without_a_model(tmp_path):
    label = meds.Label(subject_id=1, prediction_time=datetime.datetime(2020, 1, 1), boolean_value=True)
    batch_dir = tmp_path / "batches"
    processor = mock.MagicMock()
    processor.convert_dataset.return_value.save_to_disk.side_effect = os.mkdir
    no_model = mock.Mock(side_effect=AssertionError("the model must not be loaded"))

    with mock.patch.object(embedding, "load_femr_model_with_compat", no_model), \
            mock.patch("femr.models.tokenizer.Tokenizer.from_pretrained"), \
            mock.patch("femr.models.processor.FEMRBatchProcessor", return_value=processor):
        result = embedding.compute_features(
            db=mock.MagicMock(), model_path="tokenizer_folder", labels=[label], device="cuda:0",
            batch_dir=str(batch_dir), batches_only=True)

    assert result is None
    processor.convert_dataset.assert_called_once()
    assert batch_dir.is_dir()
