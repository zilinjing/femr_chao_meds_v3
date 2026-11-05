import femr.ontology
import pathlib
import meds_reader
import pickle
import femr.splits
import filecmp
from femr.models.tokenizer import Tokenizer
import femr.models.tasks
import femr.models.processor
import pandas as pd
import polars as pl
import logging
import time
import json
import contextlib
import os
import sys
import threading
import traceback
from femr.models.tasks.motor import MOTOR_Task as MOTOR_Task
from femr.models.tasks.ntp import CLMBRTask as CLMBRTask
from femr.models.tasks.tpp import TPP_Task as TPP_Task
from femr.models.tasks.mtpp import MTPP_Task as MTPP_Task
from femr.models.tasks.mtppvalue import MTPPValue_Task as MTPPValue_Task
from femr.models.tasks.survivehr_task import SurvivEHRTask
from femr.models.tasks.icarefm_task import ICareFMTask


@contextlib.contextmanager
def fail_fast(database, poll_seconds=10):
    """Exit with an error instead of hanging when meds_reader multiprocessing fails.

    meds_reader never notices a dead worker (the parent waits for its result forever), and on an
    exception its shutdown joins workers that may be blocked sending results nobody reads. So: a
    watchdog thread stops the run when a worker exits with an error, and an exception in the parent
    is printed and ends the run before meds_reader's shutdown. Both kill the workers and exit with 1.
    """
    workers = getattr(database, "_processes", None) or []

    def stop(message):
        print(message, file=sys.stderr, flush=True)
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        sys.stdout.flush()
        os._exit(1)

    def watch():
        while True:
            for worker in workers:
                if worker.exitcode not in (None, 0):
                    stop(f"meds_reader worker {worker.pid} died (exit code {worker.exitcode}); stopping prepare.py.")
            time.sleep(poll_seconds)

    threading.Thread(target=watch, daemon=True).start()
    try:
        yield
    except BaseException:
        traceback.print_exc()
        stop("prepare.py failed; meds_reader workers stopped.")


def main(args):
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)
    subject_splits_path = meds_reader_path / "metadata/subject_splits.parquet"
    code_metadata_path = meds_reader_path / "metadata/codes.parquet"
    num_threads = args.num_threads
    codes_to_skip = []
    if args.motor_codes_to_skip:
        codes_to_skip = pl.read_parquet(args.motor_codes_to_skip)
        codes_to_skip = codes_to_skip["code"].to_list()


    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=args.num_threads) as database, fail_fast(database):
        subject_ids = [_ for _ in database]
        print(f"subject_ids length: {len(subject_ids)}")
        ontology_path = pathlib.Path(args.ontology_path)
        print(f"ontology_path: {ontology_path}")
        if not ontology_path.exists():
            print("Creating ontology")
            ontology = femr.ontology.Ontology(args.athena_path, code_metadata_path=str(code_metadata_path))
            print("Pruning the ontology")
            ontology.prune_to_dataset(
                database,
                prune_all_descriptions=True,
                remove_ontologies={'SPL', 'HemOnc'}
                # remove_ontologies={'SPL', 'HemOnc', 'LOINC'}
            )

            with open(ontology_path, 'wb') as f:
                pickle.dump(ontology, f)
        else:
            print("Loading ontology")
            with open(ontology_path, 'rb') as f:
                ontology = pickle.load(f)

        subject_splits = pd.read_parquet(subject_splits_path)
        subject_splits = subject_splits[subject_splits.subject_id.isin(subject_ids)]
        train_tuning_split = subject_splits[~subject_splits.split.isin(["held_out"])].subject_id.tolist()
        test_split = subject_splits[subject_splits.split.isin(["held_out"])].subject_id.tolist()
        main_split = femr.splits.SubjectSplit(train_tuning_split, test_split)
        main_split.save_to_csv(str(pretraining_data_path / 'main_split.csv'))

        train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.05)

        main_database = database.filter(main_split.train_subject_ids)
        train_database = main_database.filter(train_split.train_subject_ids)
        val_database = main_database.filter(train_split.test_subject_ids)

        # All models share one tokenizer: it is trained once at --tokenizer_path and reused.
        local_tokenizer_path = pretraining_data_path / 'tokenizer'
        tokenizer_path = pathlib.Path(args.tokenizer_path) if args.tokenizer_path else local_tokenizer_path
        if not tokenizer_path.exists():
            print(f"Train tokenizer at {tokenizer_path}")
            Tokenizer.train(
                main_database, vocab_size=args.vocab_size, ontology=ontology,
                banned_properties=set(args.banned_properties),
            ).save_pretrained(tokenizer_path)
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_pretrained(tokenizer_path, ontology=ontology)

        # pretrain.py and saved checkpoints read the tokenizer from the pretraining directory,
        # so keep a copy there and make sure an existing copy is the shared one.
        if local_tokenizer_path.resolve() != tokenizer_path.resolve():
            if not local_tokenizer_path.exists():
                tokenizer.save_pretrained(local_tokenizer_path)
            elif not filecmp.cmp(local_tokenizer_path / 'dictionary.msgpack', tokenizer_path / 'dictionary.msgpack', shallow=False):
                raise ValueError(f"{local_tokenizer_path} differs from the shared tokenizer {tokenizer_path}")

        task_loss_type = "mtpp" if args.loss_type == "mtpp_shared" else args.loss_type
        task_path = pretraining_data_path / f'{task_loss_type}_task.pkl'

        if not task_path.exists():
            # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
            print(f"Train {args.loss_type} task")

            # MOTOR/TPP/MTPP start from every code token in the vocabulary (no candidate cap),
            # and keep every task that passes their filters unless --num_final_tasks is given.
            num_code_tokens = sum(1 for entry in tokenizer.dictionary["vocab"] if entry["type"] == "code")
            num_final_tasks = args.num_final_tasks or num_code_tokens

            if args.loss_type == "motor":
                print("motor choosen")
                task = MOTOR_Task.fit_pretraining_task_info(
                    main_database, tokenizer,
                    num_tasks=num_code_tokens,
                    num_bins=args.time_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip
                )
            elif args.loss_type == "tpp":
                print("tpp choosen")
                task = TPP_Task.fit_pretraining_task_info(
                    main_database, tokenizer,
                    num_tasks=num_code_tokens,
                    num_bins=args.time_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip
                )
            elif args.loss_type == "mtpp" or args.loss_type == "mtpp_shared":
                print("mtpp choosen")
                task = MTPP_Task.fit_pretraining_task_info(
                    main_database, tokenizer,
                    num_tasks=num_code_tokens,
                    num_final_tasks=num_final_tasks,
                    num_bins=args.time_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip,
                    num_value_bins=args.value_bins,
                    value_count_threshold=args.value_count_threshold,
                    count_label_positions=args.count_label_positions,
                )
            elif args.loss_type == "mtpp_value":
                print("mtpp_value choosen")
                task = MTPPValue_Task.fit_pretraining_task_info(
                    main_database,
                    tokenizer,
                    num_tasks=num_code_tokens,
                    num_final_tasks=num_final_tasks,
                    num_bins=args.time_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip,
                    num_value_bins=args.value_bins,
                    value_count_threshold=args.value_count_threshold,
                )
            elif args.loss_type == "ntp":
                print("ntp choosen")
                task = CLMBRTask(clmbr_vocab_size=len(tokenizer.original_code_to_id))
            elif args.loss_type == "survivehr":
                print("survivehr choosen")
                task = SurvivEHRTask.fit_pretraining_task_info(
                    main_database,
                    tokenizer,
                    value_count_threshold=args.value_count_threshold,
                )
            elif args.loss_type == "icarefm":
                print("icarefm choosen")
                task = ICareFMTask.fit_pretraining_task_info(
                    main_database,
                    tokenizer,
                    value_count_threshold=args.value_count_threshold,
                )
            else:
                print("no task matched")

            with open(task_path, 'wb') as f:
                pickle.dump(task, f)

        else:
            with open(task_path, 'rb') as f:
                task = pickle.load(f)
            # Bins are fixed when the task is fit; an existing task must match the requested bins.
            if isinstance(task, MOTOR_Task):
                fitted = {"time_bins": len(task.time_bins) - 1}
            elif isinstance(task, TPP_Task):
                fitted = {"time_bins": task.time_bins.shape[1] - 1}
            elif isinstance(task, MTPP_Task):
                fitted = {"time_bins": task.non_numerical_time_bins.shape[1] - 1, "value_bins": task.num_value_bins}
            else:
                fitted = {}
            for name, value in fitted.items():
                if value != getattr(args, name):
                    raise ValueError(f"{task_path} was fit with {name}={value}, but --{name} {getattr(args, name)} was given. "
                                     f"Use another --pretraining_data directory or move the task file away to refit.")

        if args.tasks_only:
            print(f"--tasks_only: stop after {task_path}")
            return

        # print(f"Motor task length: {len(motor_task.pretraining_task_codes)}")
        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task, args.hf_cache_path)

        # example_subject_id = list(train_database)[0]
        # example_subject = train_database[example_subject_id]

        # We can do this one subject at a time
        # print("Convert a single subject")
        # example_batch = processor.collate([processor.convert_subject(example_subject, tensor_type='pt')])
        # print(f"example subject_id {example_subject_id} , example_subject: {example_subject}")
        # print(f"example_batch keys: {example_batch['batch'].keys()}")
        # for key in example_batch['batch'].keys():
        #     print(f"example_batch[{key}], shape: {example_batch['batch'][key].shape}, values: {example_batch['batch'][key]}")

        # Generate only one style of batches based on model type
        prepare_timing = {}

        if args.model in ("transformer", None):
            train_batches_path = pretraining_data_path / 'train_batches'
            if not train_batches_path.exists():
                print(f"Convert batches (transformer-style), saving to {train_batches_path}")
                t0 = time.perf_counter()
                train_batches = processor.convert_dataset(
                    train_database, tokens_per_batch=args.tokens_per_batch,
                    min_subjects_per_batch=args.min_subjects_per_batch, num_proc=num_threads
                )
                prepare_timing["train_batches_time"] = time.perf_counter() - t0
                print(f"Train batches conversion took {prepare_timing['train_batches_time']:.2f}s")
                print("Convert batches to pytorch")
                train_batches.set_format("pt")
                train_batches.save_to_disk(train_batches_path)

            val_batches_path = pretraining_data_path / 'val_batches'
            if not val_batches_path.exists():
                print(f"Convert val batches (transformer-style), saving to {val_batches_path}")
                t0 = time.perf_counter()
                val_batches = processor.convert_dataset(
                    val_database, tokens_per_batch=args.tokens_per_batch,
                    min_subjects_per_batch=args.min_subjects_per_batch, num_proc=num_threads
                )
                prepare_timing["val_batches_time"] = time.perf_counter() - t0
                print(f"Val batches conversion took {prepare_timing['val_batches_time']:.2f}s")
                val_batches.set_format("pt")
                val_batches.save_to_disk(val_batches_path)
        elif args.model == "mamba":
            # Mamba-optimized packing: ≤mamba_per_sample_max_tokens per sample, ≤tokens_per_batch per batch
            train_batches_mamba_path = pretraining_data_path / 'train_batches_mamba'
            if not train_batches_mamba_path.exists():
                print(f"Convert batches (mamba-optimized), saving to {train_batches_mamba_path}")
                t0 = time.perf_counter()
                train_batches_mamba = processor.convert_dataset_mamba(
                    train_database,
                    tokens_per_batch=args.tokens_per_batch,
                    per_sample_max_tokens=getattr(args, 'mamba_per_sample_max_tokens', 8192),
                    num_proc=num_threads,
                )
                prepare_timing["train_batches_time"] = time.perf_counter() - t0
                print(f"Train batches (mamba) conversion took {prepare_timing['train_batches_time']:.2f}s")
                train_batches_mamba.set_format("pt")
                train_batches_mamba.save_to_disk(train_batches_mamba_path)

            val_batches_mamba_path = pretraining_data_path / 'val_batches_mamba'
            if not val_batches_mamba_path.exists():
                print(f"Convert val batches (mamba-optimized), saving to {val_batches_mamba_path}")
                t0 = time.perf_counter()
                val_batches_mamba = processor.convert_dataset_mamba(
                    val_database,
                    tokens_per_batch=args.tokens_per_batch,
                    per_sample_max_tokens=getattr(args, 'mamba_per_sample_max_tokens', 8192),
                    num_proc=num_threads,
                )
                prepare_timing["val_batches_time"] = time.perf_counter() - t0
                print(f"Val batches (mamba) conversion took {prepare_timing['val_batches_time']:.2f}s")
                val_batches_mamba.set_format("pt")
                val_batches_mamba.save_to_disk(val_batches_mamba_path)
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        if prepare_timing:
            timing_path = pretraining_data_path / ('prepare_timing.json' if args.model in ("transformer", None) else f'prepare_timing_{args.model}.json')
            with open(timing_path, 'w') as f:
                json.dump(prepare_timing, f)
            print(f"Prepare timing saved to {timing_path}")


def create_omop_meds_tutorial_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for preparing Motor")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--ontology_path",
        dest="ontology_path",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--athena_path",
        dest="athena_path",
        action="store",
        required=False,
        default="",
    )
    parser.add_argument(
        "--motor_codes_to_skip",
        dest="motor_codes_to_skip",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--num_threads",
        dest="num_threads",
        action="store",
        required=False,
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        action="store",
        required=False,
        type=int,
        # this is decided based on the 99% percentile of the number of tokens
        default=8192,
    )
    parser.add_argument(
        "--mamba_per_sample_max_tokens",
        dest="mamba_per_sample_max_tokens",
        action="store",
        required=False,
        type=int,
        default=8192,
        help="Max tokens per sample/window for Mamba batching (truncate if longer)",
    )
    parser.add_argument(
        "--time_bins",
        dest="time_bins",
        action="store",
        required=False,
        type=int,
        default=8,
    )

    parser.add_argument(
        "--vocab_size",
        dest="vocab_size",
        action="store",
        required=False,
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--num_final_tasks",
        dest="num_final_tasks",
        action="store",
        required=False,
        type=int,
        default=None,
        help="Cap on MOTOR/TPP/MTPP tasks after filtering (default: no cap)",
    )

    parser.add_argument(
        "--count_label_positions",
        dest="count_label_positions",
        action="store_true",
        help="MTPP: apply --value_count_threshold to label-position value samples instead of raw values",
    )

    parser.add_argument(
        "--tokenizer_path",
        dest="tokenizer_path",
        action="store",
        required=False,
        help="Shared tokenizer directory; trained there if missing (default: <pretraining_data>/tokenizer)",
    )

    parser.add_argument(
        "--min_subjects_per_batch",
        dest="min_subjects_per_batch",
        action="store",
        required=False,
        type=int,
        default=1,
        help="Transformer windows are at most tokens_per_batch // min_subjects_per_batch events",
    )

    parser.add_argument(
        "--banned_properties",
        dest="banned_properties",
        nargs="*",
        default=[],
        help="Columns that get no text/numeric tokens when the tokenizer is trained (e.g. hadm_id icustay_id)",
    )

    parser.add_argument(
        "--tasks_only",
        dest="tasks_only",
        action="store_true",
        help="Stop after the tokenizer and the task pickle (do not build batches)",
    )

    parser.add_argument(
        "--hf_cache_path",
        dest="hf_cache_path",
        action="store",
        required=False,
    )

    parser.add_argument(
        "--value_bins",
        dest="value_bins",
        action="store",
        required=False,
        type=int,
        default=10,
    )

    parser.add_argument(
        "--value_count_threshold",
        dest="value_count_threshold",
        action="store",
        required=False,
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--loss_type",
        dest="loss_type",
        action="store",
        required=True,
        choices=["motor", "tpp", "mtpp", "mtpp_shared", "mtpp_value", "ntp", "survivehr", "icarefm"],
        type=str,
        default=8,
    )

    # Choose how to build pretraining batches (affects efficiency for different backbones)
    # Support both --model for convenience
    parser.add_argument(
        "--model",
        dest="model",
        action="store",
        required=False,
        choices=["transformer", "mamba"],
        default="transformer",
        help="Batching target: transformer (original) or mamba (length-packed)",
    )
    

    return parser


if __name__ == "__main__":
    main(create_omop_meds_tutorial_argparser().parse_args())

'''
mimic
gsb
python prepare.py \
  --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  --ontology_path /user/zj2398/cache/mtpp_8k_num_bin5/ontology.pkl \
  --time_bins 8 \
  --num_threads 64 \
  --value_bins 5 \
  --num_final_tasks 6100 \
  --tokens_per_batch 16384 \
  --model mamba \
  --loss_type mtpp \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader

python prepare.py \
  --pretraining_data /user/zj2398/cache/mtpp_8k_T_8_V_5_all_nonnume \
  --ontology_path /user/zj2398/cache/mtpp_8k_T_8_V_5_all_nonnume/ontology.pkl \
  --time_bins 8 \
  --num_threads 32 \
  --value_bins 5 \
  --num_final_tasks 6100 \
  --tokens_per_batch 8192 \
  --model mamba \
  --loss_type mtpp \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader

python prepare.py \
  --pretraining_data /user/zj2398/cache/survivehr \
  --ontology_path /user/zj2398/cache/survivehr/ontology.pkl \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --num_threads 32 \
  --tokens_per_batch 16384 \
  --value_count_threshold 1000 \
  --model mamba \
  --loss_type survivehr \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets

python prepare.py \
  --pretraining_data /user/zj2398/cache/icarefm \
  --ontology_path /user/zj2398/cache/icarefm/ontology.pkl \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --num_threads 32 \
  --tokens_per_batch 16384 \
  --value_count_threshold 1000 \
  --model mamba \
  --loss_type icarefm \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets

kuvira mimic

python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/mimic/mtpp \
  --ontology_path /data2/processed_datasets/zj2398/femr/ontology.pkl \
  --time_bins 8 \
  --num_threads 80 \
  --value_bins 5 \
  --num_final_tasks 6100 \
  --tokens_per_batch 16384 \
  --model mamba \
  --loss_type mtpp \
  --hf_cache_path /data2/huggingface/zj2398/ \
  --meds_reader  /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader


kuvira cumc

python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/mimic/mtpp \
  --ontology_path /data2/processed_datasets/zj2398/femr/ontology.pkl \
  --time_bins 8 \
  --num_threads 80 \
  --value_bins 5 \
  --num_final_tasks 6100 \
  --tokens_per_batch 16384 \
  --model mamba \
  --loss_type mtpp \
  --hf_cache_path /data2/huggingface/zj2398/ \
  --meds_reader  /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader

python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/icarefm \
  --ontology_path /data2/processed_datasets/zj2398/femr/cumc/icarefm/ontology.pkl \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_threads 24 \
  --tokens_per_batch 16384 \
  --value_count_threshold 1000 \
  --model transformer \
  --loss_type icarefm \
  --hf_cache_path /data2/huggingface/zj2398/



    
'''
