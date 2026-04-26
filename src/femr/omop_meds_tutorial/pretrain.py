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
import wandb
import datasets
datasets.disable_caching()

from femr.models.tasks.motor import MOTOR_Task as MOTOR_Task
from femr.models.tasks.tpp import TPP_Task as TPP_Task
from femr.models.tasks.mtpp import MTPP_Task as MTPP_Task
from femr.models.tasks.mtppvalue import MTPPValue_Task as MTPPValue_Task
from femr.models.tasks.survivehr_task import SurvivEHRTask as SurvivEHRTask

class CustomEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) / max(abs(state.best_metric), 1e-12)
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
        choices=["motor", "tpp", "mtpp", "mtpp_shared", "mtpp_value", "ntp", "survivehr"],
        default="motor",
        required=True,
    )
    arg_parser.add_argument(
        "--linear_interpolation",
        dest="linear_interpolation",
        type=bool,
        default=False
    )
    arg_parser.add_argument(
        "--survivehr_time_scale_days",
        dest="survivehr_time_scale_days",
        type=float,
        default=1.0,
        help="Scale raw SurviveHR day deltas before the ODE head. Use 1.0 for days.",
    )
    arg_parser.add_argument(
        "--logging_steps",
        dest="logging_steps",
        type=int,
        default=100,
        help="Log training metrics every N optimizer steps.",
    )
    return arg_parser

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


class SurviveHRMetricsCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        if not getattr(state, "is_world_process_zero", True):
            return
        if model is None or wandb.run is None:
            return

        task_model = getattr(_unwrap_model(model), "task_model", None)
        metrics = getattr(task_model, "last_metrics", None)
        if not metrics:
            return

        log_data = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                log_data[key] = value.detach().float().cpu().item()
            else:
                log_data[key] = float(value)

        if log_data:
            log_data["train/global_step"] = state.global_step
            wandb.log(log_data)


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    if args.loss_type in ("ntp", "survivehr"):
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

    task_loss_type = "mtpp" if args.loss_type == "mtpp_shared" else args.loss_type
    task_path = pretraining_data / f'{task_loss_type}_task.pkl'
    with open(task_path, 'rb') as f:
        task = pickle.load(f)
    if args.loss_type == "survivehr":
        task.time_scale_days = args.survivehr_time_scale_days
        print(f"Using SurviveHR time_scale_days={task.time_scale_days}")
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


    print(f"tokenizer vocab is {tokenizer.vocab_size}")
    print(f"embedding bag {model.backbone.embed_bag}")
    # sys.exit(0)

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
        report_to="wandb",
        max_grad_norm=args.max_grad_norm,
        run_name=args.hf_run_name,
        # run_name="motor_pretrain_mimic",
        num_train_epochs=args.n_epochs,
        ddp_find_unused_parameters=(args.loss_type == "survivehr"),

        warmup_steps=1000,

        logging_strategy='steps',
        logging_steps=args.logging_steps,

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


    callbacks = [CustomEarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
    if args.loss_type == "survivehr":
        callbacks.append(SurviveHRMetricsCallback())

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


# python -u -m femr.omop_meds_tutorial.pretrain_motor \
#   --pretraining_data $PRETRAINING_DATA \
#   --meds_reader $OMOP_MEDS_READER

'''

export CUDA_VISIBLE_DEVICES=2

kuvira

export CUDA_VISIBLE_DEVICES=4,5,6,7

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
  --num_processes 2 \
  pretrain.py \
    --pretraining_data  /data2/processed_datasets/zj2398/femr/mimic/motor  \
    --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
    --per_device_train_batch_size 1 \
    --output_dir /data2/processed_datasets/zj2398/femr/mimic/motor/transformer  \
    --hf_run_name mimic_motor_test \
    --model transformer \
    --loss_type motor

    
accelerate launch \
    --num_processes 2 \
    pretrain.py \
    --pretraining_data /user/zj2398/cache/survivehr \
    --meds_reader /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
    --output_dir /user/zj2398/cache/survivehr/transformer \
    --model transformer \
    --loss_type survivehr \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --hf_run_name mimic_survivehr_transformer_value_1000



accelerate launch \
  --num_processes 2 \
    pretrain.py \
    --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/survivehr \
    --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
    --output_dir /data2/processed_datasets/zj2398/femr/cumc/survivehr/transformer  \
    --model transformer \
    --loss_type survivehr \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --hf_run_name cumc_survivehr_transformer_value_1000

mimic 
gsb


accelerate launch \
    --num_processes 3 \
    pretrain.py \
    --pretraining_data /user/zj2398/cache/survivehr \
    --meds_reader /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
    --output_dir /user/zj2398/cache/survivehr/transformer \
    --model transformer \
    --loss_type survivehr \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --hf_run_name mimic_survivehr_transformer_value_1000


    
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



'''
