import numpy as np
import transformers
import pathlib
import torch
import sys
import time
import json
import femr.models.architecture.transformer
import femr.models.architecture.embedding
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from femr.omop_meds_tutorial.evaluation.generate_labels import create_omop_meds_tutorial_arg_parser
import torch.nn as nn
from transformers import TrainerCallback
from torch.profiler import ProfilerActivity, profile
import wandb
import datasets
datasets.disable_caching()

from femr.models.tasks.motor import MOTOR_Task as MOTOR_Task
from femr.models.tasks.tpp import TPP_Task as TPP_Task
from femr.models.tasks.mtpp import MTPP_Task as MTPP_Task

class CustomEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) / state.best_metric
                > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


def create_arg_parser():
    arg_parser = create_omop_meds_tutorial_arg_parser()
    arg_parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default=None
    )

    arg_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None
    )
    arg_parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-5
    )

    arg_parser.add_argument(
        "--max_grad_norm",
        dest="max_grad_norm",
        type=float,
        default=1.0
    )

    arg_parser.add_argument(
        "--hf_run_name",
        dest="hf_run_name",
        type=str,
        default="new_run"
    )
    arg_parser.add_argument(
        "--n_layers",
        dest="n_layers",
        type=int,
        default=11
    )
    arg_parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=20
    )
    arg_parser.add_argument(
        "--per_device_train_batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--model",
        dest="model",
        type=str,
        choices=["transformer", "mamba"],
        default="transformer",
        required=True,
    )
    arg_parser.add_argument(
        "--loss_type",
        dest="loss_type",
        type=str,
        choices=["motor", "tpp","mtpp","mtpp_shared","ntp"],
        default="motor",
        required=True,
    )
    arg_parser.add_argument(
        "--linear_interpolation",
        dest="linear_interpolation",
        type=bool,
        default=False
    )
    return arg_parser

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class WandbTrainLossCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        log_data = {}
        
        # Train loss vs step
        if "loss" in logs:
            log_data["train/loss_step"] = logs["loss"]
        
        # Eval loss vs step
        if "eval_loss" in logs:
            log_data["eval/loss_step"] = logs["eval_loss"]
        
        # Add epoch-based logging
        if "loss" in logs:
            log_data["train/loss_epoch"] = logs["loss"]
        if "eval_loss" in logs:
            log_data["eval/loss_epoch"] = logs["eval_loss"]
        
        # Record step and epoch for W&B axes
        log_data["global_step"] = state.global_step
        log_data["epoch"] = state.epoch
        
        wandb.log(log_data)


class ProfilingTrainer(transformers.Trainer):
    """Trainer that profiles FLOPs (epoch 1) and clean wall time (epoch 2).

    Epoch 1: profiler ON  -> accurate FLOPs, inflated time (profiler overhead).
    Epoch 2+: profiler OFF -> accurate wall time, no overhead.

    Set n_epochs=2 to get both accurate FLOPs and accurate per-epoch time.
    Then multiply each by your real number of epochs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiled_step_flops = []    # FLOPs per step (epoch 1 only)
        self.clean_step_times = []       # wall time per step (epoch 2 only)

    def training_step(self, *args, **kwargs):
        use_cuda = self.args.device.type == 'cuda'

        if self.state.epoch < 1.0:
            # Epoch 1: profile for FLOPs (time is inflated by profiler)
            if use_cuda:
                activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            else:
                activities = [ProfilerActivity.CPU]

            with profile(activities=activities, with_flops=True) as prof:
                loss = super().training_step(*args, **kwargs)

            step_flops = sum(
                evt.flops for evt in prof.key_averages()
                if evt.flops is not None and evt.flops > 0
            )
            self.profiled_step_flops.append(step_flops)
            return loss

        elif self.state.epoch < 2.0:
            # Epoch 2: no profiler, measure clean wall time
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            loss = super().training_step(*args, **kwargs)

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.clean_step_times.append(t1 - t0)
            return loss

        return super().training_step(*args, **kwargs)


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    if args.loss_type == "ntp":
        tokenizer = femr.models.tokenizer.CLMBRTokenizer.from_pretrained(
            tokenizer_path, ontology=ontology
        )
    else:
        tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
            tokenizer_path, ontology=ontology
        )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


    # # <-- critical alias for pickle backward-compatibility
    # import femr.models.tasks as _tasks_mod
    # _tasks_mod.MOTORTask = _MOTORImpl

    model_type = args.loss_type.split("_")[0]
    task_path = pretraining_data / f'{model_type}_task.pkl'
    with open(task_path, 'rb') as f:
        task = pickle.load(f)
    # print(f"{args.loss_type} task: {task}")
    # print(f"{args.loss_type} task length: {len(task.pretraining_task_codes)}")
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task)

    # Choose dataset packaging based on model
    if args.model == "mamba":
        train_batches_path = pretraining_data / 'train_batches_mamba'
        val_batches_path = pretraining_data / 'val_batches_mamba'
    else:
        train_batches_path = pretraining_data / 'train_batches'
        val_batches_path = pretraining_data / 'val_batches'

    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Use Hydra to load model hyperparameters and keep tokenizer-derived fields runtime-bound
    conf_dir = pathlib.Path(__file__).parent / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        hydra_cfg = compose(config_name="config", overrides=[f"model={args.model}"])

    model_cfg_dict = OmegaConf.to_container(hydra_cfg.model, resolve=True)

    # Build architecture config from Hydra, then override vocab_size and is_hierarchical from tokenizer
    if args.model == "transformer":
        print("using transformer as the model")
        # Apply optional CLI override for n_layers
        if args.n_layers is not None:
            model_cfg_dict["n_layers"] = args.n_layers
        model_config = femr.models.config.FEMRTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            is_hierarchical=(isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer) or isinstance(tokenizer, femr.models.tokenizer.CLMBRTokenizer)),
            hidden_size=model_cfg_dict.get("hidden_size"),
            intermediate_size=model_cfg_dict.get("intermediate_size"),
            n_heads=model_cfg_dict.get("n_heads"),
            n_layers=model_cfg_dict.get("n_layers"),
            attention_width=model_cfg_dict.get("attention_width"),
            use_normed_ages=model_cfg_dict.get("use_normed_ages"),
            use_bias=model_cfg_dict.get("use_bias"),
            hidden_act=model_cfg_dict.get("hidden_act"),
        )
    else:  # mamba
        # Apply optional CLI override for n_layers
        print("using mamba as the model")
        if args.n_layers is not None:
            model_cfg_dict["n_layers"] = args.n_layers
        model_config = femr.models.config.FEMRMambaConfig(
            vocab_size=tokenizer.vocab_size,
            is_hierarchical=(isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer) or isinstance(tokenizer, femr.models.tokenizer.CLMBRTokenizer)),
            hf_name=model_cfg_dict.get("hf_name"),
            hidden_size=model_cfg_dict.get("hidden_size"),
            intermediate_size=model_cfg_dict.get("intermediate_size"),
            n_layers=model_cfg_dict.get("n_layers"),
            d_state=model_cfg_dict.get("d_state"),
            use_normed_ages=model_cfg_dict.get("use_normed_ages"),
            use_bias=model_cfg_dict.get("use_bias"),
            config_kwargs=model_cfg_dict.get("config_kwargs") or {},
        )

    config = femr.models.config.FEMRModelConfig.from_task_configs(model_config, task.get_task_config())
    # print(f"The config is {config}")

    # Unified model wrapper supports both architectures
    model = femr.models.architecture.embedding.FEMRModel(config, loss_type=args.loss_type,linear_interpolation=args.linear_interpolation)
    # model = model.to(torch.device("cuda:0"))


    print(f"Model param count: {count_parameters(model)}")
    # print(f"model config is {model_config}") 

    learning_rate = args.learning_rate
    # output_dir = 'tmp_trainer_' + sys.argv[1]
    output_dir = args.output_dir
    # print(args.per_device_train_batch_size)
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        learning_rate=learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,
        # lr_scheduler_type="cosine",

        weight_decay=0.05,
        adam_beta2=0.95,
        # report_to="none",
        report_to=["wandb"],
        max_grad_norm=args.max_grad_norm,
        run_name=args.hf_run_name,
        # run_name="motor_pretrain_mimic",
        num_train_epochs=args.n_epochs,
        ddp_find_unused_parameters=False,

        warmup_steps=1000,

        logging_strategy='epoch',
        # logging_strategy='steps',
        logging_steps=10,

        save_strategy='epoch',
        eval_strategy='epoch',
        # evaluation_strategy='epoch',

        # prediction_loss_only=True,
        # dataloader_num_workers=1,
        dataloader_num_workers=40,

        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    print(f"CONFIRMATION: The actual per_device_train_batch_size is {trainer_config.per_device_train_batch_size}")


    trainer = ProfilingTrainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Save FLOPs and time metrics
    # Epoch 1 gives accurate FLOPs, epoch 2 gives accurate time.
    # Multiply each by your real num_epochs for total training cost.
    per_epoch_flops = sum(trainer.profiled_step_flops)
    per_epoch_time = sum(trainer.clean_step_times)
    training_metrics = {
        "per_epoch_flops": per_epoch_flops,       # exact FLOPs for 1 epoch (from epoch 1 profiling)
        "per_epoch_time": per_epoch_time,          # clean wall time for 1 epoch in seconds (from epoch 2, no profiler)
        "steps_per_epoch": len(trainer.profiled_step_flops),
        "n_epochs_run": args.n_epochs,             # how many epochs this run used (should be 2 for profiling)
    }
    metrics_path = pathlib.Path(output_dir) / "training_flops_time.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Training FLOPs/time metrics saved to {metrics_path}")
    print(f"  per_epoch_flops: {per_epoch_flops:.4e} ({per_epoch_flops / 1e12:.4f} TFLOPs)")
    print(f"  per_epoch_time:  {per_epoch_time:.2f}s")
    print(f"  To estimate full training: multiply each by your real num_epochs")


if __name__ == "__main__":
    main()


# python -u -m femr.omop_meds_tutorial.pretrain_motor \
#   --pretraining_data $PRETRAINING_DATA \
#   --meds_reader $OMOP_MEDS_READER

'''

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

kuvira

export CUDA_VISIBLE_DEVICES=1,5,6,7

accelerate launch \
  --num_processes 4 \
  pretrain.py \
    --pretraining_data  /data2/processed_datasets/zj2398/femr/mimic/mtpp  \
    --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --output_dir /data2/processed_datasets/zj2398/femr/mimic/mtpp/mamba  \
    --hf_run_name mimic_mamba_revised_mtpp_6100 \
    --max_grad_norm 1.0 \
    --model mamba \
    --n_layers 28 \
    --loss_type mtpp_shared

accelerate launch \
  --num_processes 4 \
  pretrain.py  \
    --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5   \
    --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader  \
    --per_device_train_batch_size 1   \
    --learning_rate 8e-6 \
    --max_grad_norm 4.0 \
    --output_dir /user/zj2398/cache/mtpp_all_shared_6100_bin_5/mamba_8e-6_4.0_cosine \
    --model mamba  \
    --n_layers 28  \
    --hf_run_name mimic_mamba_mtpp_shared_6100_8e-6_4.0_cosine  \
    --loss_type mtpp_shared

  python pretrain.py \
    --pretraining_data  /data2/processed_datasets/zj2398/femr/mimic/mtpp  \
    --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-6 \
    --output_dir /data2/processed_datasets/zj2398/femr/cumc/mtpp/mamba  \
    --hf_run_name mimic_mamba_revised_mtpp_6100 \
    --model mamba \
    --loss_type mtpp_shared


mimic 
gsb

export CUDA_VISIBLE_DEVICES=3,4,5,6,7

accelerate launch \
  --num_processes 5 \
  pretrain.py \
  --pretraining_data /user/zj2398/cache/mtpp_8k_T_16_V_5 \
  --checkpoint_dir /user/zj2398/cache/mtpp_8k_T_16_V_5/transformer/checkpoint-64392 \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/mtpp_8k_T_16_V_5/transformer \
  --model transformer \
  --hf_run_name mimic_transformer_mtpp_6100_T_16_V_5_continue \
  --loss_type mtpp_shared 

#bash run_mimic_new.sh 16 10 /user/zj2398/cache/mtpp_8k_T_16_V_5

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --num_processes 4 \
  --gpu_ids "4,5,6,7" \
    pretrain.py \
    --pretraining_data /user/zj2398/cache/mtpp_8k \
    --meds_reader /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
    --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_mean_all_shared \
    --model transformer \
    --hf_run_name mimic_mtpp_shared \
    --loss_type mtpp_shared

    
python pretrain_motor.py \
  --pretraining_data /user/zj2398/cache/mtpp_8k \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --checkpoint_dir /user/zj2398/cache/mtpp_8k/output_mamba_store/checkpoint-53660 \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/mtpp_8k/output_mamba \
  --model mamba \
  --loss_type mtpp \
  --learning_rate e-6 \
  --n_layers 12

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --num_processes 4 \
  --gpu_ids "4,5,6,7" \
  pretrain.py \
  --pretraining_data /shared/share_mala/zj2398/fomoh/motor \
  --meds_reader /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /shared/share_mala/zj2398/fomoh/motor/test \
  --model transformer \
  --loss_type motor







'''
