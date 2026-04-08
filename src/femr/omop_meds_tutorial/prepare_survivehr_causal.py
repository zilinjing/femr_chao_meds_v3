from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import meds_reader
import pandas as pd

import femr.splits
from femr.models.processor import FEMRBatchProcessor
from femr.models.tasks.survivehr_causal import SurvivEHRCausalTask
from femr.models.tokenizer import HierarchicalTokenizer


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Prepare FEMR batches for SurvivEHR-style causal pretraining.")
    parser.add_argument("--pretraining_data", required=True, type=str)
    parser.add_argument("--meds_reader", required=True, type=str)
    parser.add_argument("--mtpp_task_path", required=True, type=str)
    parser.add_argument("--model", choices=["transformer", "mamba"], default="transformer")
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--tokens_per_batch", type=int, default=8192)
    parser.add_argument("--mamba_per_sample_max_tokens", type=int, default=8192)
    parser.add_argument("--hf_cache_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--surv_weight", type=float, default=1.0)
    parser.add_argument("--value_weight", type=float, default=0.1)
    parser.add_argument("--survival_hidden_dim", type=int, default=32)
    parser.add_argument("--value_hidden_dim", type=int, default=32)
    parser.add_argument("--quadrature_n", type=int, default=15)
    parser.add_argument("--time_scale_days", type=float, default=365.25)
    parser.add_argument("--concurrent_strategy", type=str, default=None)
    return parser


def _load_train_val_databases(database, pretraining_data_path: pathlib.Path, meds_reader_path: pathlib.Path):
    main_split_path = pretraining_data_path / "main_split.csv"
    if main_split_path.exists():
        main_split = femr.splits.SubjectSplit.load_from_csv(str(main_split_path))
    else:
        subject_ids = [_ for _ in database]
        subject_splits = pd.read_parquet(meds_reader_path / "metadata/subject_splits.parquet")
        subject_splits = subject_splits[subject_splits.subject_id.isin(subject_ids)]
        train_tuning_split = subject_splits[~subject_splits.split.isin(["held_out"])].subject_id.tolist()
        test_split = subject_splits[subject_splits.split.isin(["held_out"])].subject_id.tolist()
        main_split = femr.splits.SubjectSplit(train_tuning_split, test_split)
        main_split.save_to_csv(str(main_split_path))

    train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.05)
    train_database = database.filter(train_split.train_subject_ids)
    val_database = database.filter(train_split.test_subject_ids)
    return train_database, val_database


def main():
    args = create_arg_parser().parse_args()
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)

    ontology_path = pretraining_data_path / "ontology.pkl"
    tokenizer_path = pretraining_data_path / "tokenizer"
    if not ontology_path.exists():
        raise FileNotFoundError(f"Expected ontology at {ontology_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Expected tokenizer at {tokenizer_path}")

    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)
    tokenizer = HierarchicalTokenizer.from_pretrained(tokenizer_path, ontology=ontology)

    with open(args.mtpp_task_path, "rb") as f:
        mtpp_task = pickle.load(f)

    task = SurvivEHRCausalTask.from_mtpp_task(
        mtpp_task,
        surv_weight=args.surv_weight,
        value_weight=args.value_weight,
        survival_hidden_dim=args.survival_hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        quadrature_n=args.quadrature_n,
        time_scale_days=args.time_scale_days,
        concurrent_strategy=args.concurrent_strategy,
    )

    task_path = pretraining_data_path / "survivehr_causal_task.pkl"
    if args.overwrite or not task_path.exists():
        with open(task_path, "wb") as f:
            pickle.dump(task, f)

    task.bind_tokenizer(tokenizer)
    processor = FEMRBatchProcessor(tokenizer, task, args.hf_cache_path)

    timing = {}
    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=args.num_threads) as database:
        train_database, val_database = _load_train_val_databases(database, pretraining_data_path, meds_reader_path)

        if args.model == "mamba":
            train_batches_path = pretraining_data_path / "train_batches_survivehr_causal_mamba"
            val_batches_path = pretraining_data_path / "val_batches_survivehr_causal_mamba"
            convert_train = processor.convert_dataset_mamba
            convert_kwargs = dict(
                tokens_per_batch=args.tokens_per_batch,
                per_sample_max_tokens=args.mamba_per_sample_max_tokens,
                num_proc=args.num_threads,
            )
        else:
            train_batches_path = pretraining_data_path / "train_batches_survivehr_causal"
            val_batches_path = pretraining_data_path / "val_batches_survivehr_causal"
            convert_train = processor.convert_dataset
            convert_kwargs = dict(tokens_per_batch=args.tokens_per_batch, num_proc=args.num_threads)

        if args.overwrite and train_batches_path.exists():
            import shutil

            shutil.rmtree(train_batches_path)
        if args.overwrite and val_batches_path.exists():
            import shutil

            shutil.rmtree(val_batches_path)

        if not train_batches_path.exists():
            t0 = time.perf_counter()
            train_batches = convert_train(train_database, **convert_kwargs)
            timing["train_batches_time_seconds"] = time.perf_counter() - t0
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        if not val_batches_path.exists():
            t0 = time.perf_counter()
            val_batches = convert_train(val_database, **convert_kwargs)
            timing["val_batches_time_seconds"] = time.perf_counter() - t0
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)

    if timing:
        timing_path = pretraining_data_path / "prepare_survivehr_causal_timing.json"
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()


'''
  PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src 

  python -m femr.omop_meds_tutorial.prepare_survivehr_causal \
    --pretraining_data /user/zj2398/cache/mtpp_8k_survivehr \
    --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --mtpp_task_path /user/zj2398/cache/mtpp_8k_survivehr/mtpp_task.pkl \
    --model transformer \
    --tokens_per_batch 8192 \
    --num_threads 32

  Train:

  PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src 
  python -m femr.omop_meds_tutorial.pretrain_survivehr_causal \
    --pretraining_data /user/zj2398/cache/mtpp_8k_survivehr \
    --output_dir /user/zj2398/cache/mtpp_8k_survivehr/survivehr_causal_transformer \
    --model transformer \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --hf_run_name survivehr_causal_transformer \
    --bf16
export CUDA_VISIBLE_DEVICES=0,1
'''