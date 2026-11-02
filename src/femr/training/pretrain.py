import numpy as np
import transformers
import pathlib
import torch
import sys
import time
import json
import copy
import femr.models.architecture.embedding
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from femr.evaluation.common.generate_labels import create_omop_meds_tutorial_arg_parser
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
import wandb
import datasets
import logging


datasets.disable_caching()

from femr.models.tasks.motor import MOTOR_Task as MOTOR_Task
from femr.models.tasks.tpp import TPP_Task as TPP_Task
from femr.models.tasks.mtpp import MTPP_Task as MTPP_Task
from femr.models.tasks.mtppvalue import MTPPValue_Task as MTPPValue_Task
from femr.models.tasks.survivehr_task import SurvivEHRTask as SurvivEHRTask
from femr.models.tasks.icarefm_task import ICareFMTask as ICareFMTask

logger = logging.getLogger(__name__)

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
        "--best_model_dir",
        dest="best_model_dir",
        type=str,
        default=None,
        help="Where to keep the lowest-eval_loss model. Defaults to <output_dir>/best_model.",
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
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of microbatches to accumulate before each optimizer step.",
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
        choices=["motor", "tpp", "mtpp_shared", "mtpp_factorized", "mtpp_value", "ntp", "survivehr", "icarefm"],
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
    arg_parser.add_argument(
        "--icarefm_eval_seed",
        dest="icarefm_eval_seed",
        type=int,
        default=42,
        help="Deterministic dynamic-threshold seed for ICareFM validation collation.",
    )
    arg_parser.add_argument(
        "--profile_steps",
        dest="profile_steps",
        type=int,
        default=0,
        help="Run the torch profiler over the first N training steps to measure FLOPs/token. "
             "0 (default) disables profiling entirely. Counted in microbatches, which equal "
             "optimizer steps (and so --max_steps) only when --gradient_accumulation_steps is 1.",
    )
    arg_parser.add_argument(
        "--time_steps",
        dest="time_steps",
        type=int,
        default=0,
        help="Time the N steps that follow --profile_steps with the profiler off, to measure "
             "seconds/token. 0 (default) skips the timing phase.",
    )
    arg_parser.add_argument(
        "--max_steps",
        dest="max_steps",
        type=int,
        default=-1,
        help="Stop after N optimizer steps instead of running --n_epochs. Any value > 0 marks the "
             "run as profiling-only: eval, checkpointing and early stopping are turned off, since "
             "the run never reaches an epoch boundary. -1 (default) trains normally.",
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


class SaveBestModelCallback(transformers.TrainerCallback):
    """Keep a standalone copy of the lowest-eval_loss model on disk *while* training runs.

    load_best_model_at_end only restores the best weights after trainer.train() returns, so a
    crashed or killed job leaves nothing but the rotating checkpoint-* dirs (which also lack the
    tokenizer that compute_features expects next to the weights). This rewrites <best_dir> every
    time eval_loss improves, so the best model so far is always on disk.
    """

    def __init__(self, best_dir, tokenizer=None, metric_name="eval_loss", greater_is_better=False):
        self.best_dir = pathlib.Path(best_dir)
        self.tokenizer = tokenizer
        self.metric_name = metric_name
        self.operator = np.greater if greater_is_better else np.less
        self.best_metric = None
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Resuming from a checkpoint: don't overwrite a better best_model with a worse first eval.
        if self.best_metric is None and state.best_metric is not None:
            self.best_metric = state.best_metric

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.trainer is None or not metrics:
            return
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return
        if self.best_metric is not None and not self.operator(metric_value, self.best_metric):
            return
        self.best_metric = metric_value
        self.save(state, reason="new best")

    def save(self, state, reason=""):
        # save_model() writes weights + config only (no optimizer state) and is a no-op off rank 0.
        self.trainer.save_model(str(self.best_dir))
        if self.trainer.args.should_save:
            self.best_dir.mkdir(parents=True, exist_ok=True)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(self.best_dir))
            with open(self.best_dir / "best_model_info.json", "w") as f:
                json.dump(
                    {
                        "metric": self.metric_name,
                        "value": self.best_metric,
                        "epoch": state.epoch,
                        "global_step": state.global_step,
                        "source_checkpoint": state.best_model_checkpoint,
                    },
                    f,
                    indent=2,
                )
        logger.info(
            "Saved best model (%s=%s, epoch=%s, step=%s)%s -> %s",
            self.metric_name, self.best_metric, state.epoch, state.global_step,
            f" [{reason}]" if reason else "", self.best_dir,
        )


class SeparateEvalCollatorTrainer(transformers.Trainer):
    """Trainer with an optional eval-only collator and optional FLOPs/time profiling.

    Profiling samples the first `profile_steps + time_steps` training steps:

        steps [0, profile_steps)                     profiler ON  -> FLOPs per token
        steps [profile_steps, +time_steps)           profiler OFF -> seconds per token
        every step after that                        untouched, zero overhead

    Splitting the two phases matters because the profiler adds enough per-step overhead to
    distort wall time, so FLOPs and time cannot be read off the same steps. Both counts
    default to 0, in which case training_step is the stock implementation.

    Rates are per token rather than per step so variable-length batches don't add noise;
    multiply them by the epoch's token count to extrapolate (see `profiling_summary`).
    """

    def __init__(self, *args, eval_data_collator=None, profile_steps=0, time_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self.profile_steps = profile_steps
        self.time_steps = time_steps
        self.sampled_steps = 0     # training steps seen so far while sampling
        self.profiled_flops = 0    # FLOPs summed over the profiled steps
        self.profiled_tokens = 0   # tokens summed over the profiled steps
        self.timed_seconds = 0.0   # wall time summed over the timed steps
        self.timed_tokens = 0      # tokens summed over the timed steps

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)

        train_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = train_collator

    @staticmethod
    def _num_tokens(inputs) -> int:
        """Total tokens in a collated batch.

        The "transformer" key holds the backbone input for Mamba runs too (processor.py),
        so this works for both architectures.
        """
        lengths = inputs["batch"]["transformer"]["subject_lengths"]
        if isinstance(lengths, torch.Tensor):
            return int(lengths.sum().item())
        return int(np.sum(lengths))

    def training_step(self, model, inputs, *args, **kwargs):
        step = self.sampled_steps
        if step >= self.profile_steps + self.time_steps:
            # Sampling is over (or was never enabled): stock training step, no overhead.
            return super().training_step(model, inputs, *args, **kwargs)

        self.sampled_steps += 1
        num_tokens = self._num_tokens(inputs)
        on_cuda = self.args.device.type == "cuda"

        if step < self.profile_steps:
            activities = [ProfilerActivity.CPU]
            if on_cuda:
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, with_flops=True) as prof:
                loss = super().training_step(model, inputs, *args, **kwargs)
            self.profiled_flops += sum(
                evt.flops for evt in prof.key_averages()
                if evt.flops is not None and evt.flops > 0
            )
            self.profiled_tokens += num_tokens
            return loss

        # Timing phase. Sync around the step so async CUDA work lands inside the interval.
        if on_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss = super().training_step(model, inputs, *args, **kwargs)
        if on_cuda:
            torch.cuda.synchronize()
        self.timed_seconds += time.perf_counter() - start
        self.timed_tokens += num_tokens
        return loss

    def profiling_summary(self, total_epoch_tokens: int) -> dict:
        """Extrapolate the sampled rates to a full epoch.

        Under DDP the ranks split one epoch between them, so `total_epoch_tokens` is the sum
        of what every rank sees. FLOPs/token is the same on each rank, so multiplying gives
        the epoch total across all ranks. Wall time is not additive -- the ranks run
        concurrently -- so that one is divided by world_size.
        """
        flops_per_token = self.profiled_flops / self.profiled_tokens if self.profiled_tokens else 0.0
        seconds_per_token = self.timed_seconds / self.timed_tokens if self.timed_tokens else 0.0
        world_size = max(1, self.args.world_size)
        return {
            "per_epoch_flops": flops_per_token * total_epoch_tokens,
            "per_epoch_time_seconds": seconds_per_token * total_epoch_tokens / world_size,
            "flops_per_token": flops_per_token,
            "time_per_token_seconds": seconds_per_token,
            "total_epoch_tokens": total_epoch_tokens,
            "world_size": world_size,
            "profiled_steps": min(self.sampled_steps, self.profile_steps),
            "timed_steps": max(0, self.sampled_steps - self.profile_steps),
            "profiled_flops": self.profiled_flops,
            "profiled_tokens": self.profiled_tokens,
            "timed_seconds": self.timed_seconds,
            "timed_tokens": self.timed_tokens,
        }


def count_epoch_tokens(train_batches, num_proc: int = 32) -> int:
    """Total tokens in one pass over the training set, used to extrapolate the sampled rates."""
    # The saved batches carry set_format("pt") from prepare.py, under which subject_lengths is
    # a torch tensor and np.sum() rejects it; strip the format so map sees plain python lists.
    counts = train_batches.with_format(None).map(
        lambda example: {"n_tokens": int(np.sum(example["transformer"]["subject_lengths"]))},
        num_proc=num_proc,
        remove_columns=train_batches.column_names,
        desc="Counting tokens",
    )
    return int(sum(counts["n_tokens"]))


def write_profiling_metrics(trainer, total_epoch_tokens: int, output_dir: str) -> None:
    """Write per-epoch FLOPs/time to <output_dir>/training_flops_time.json.

    Reported per epoch only. Runs end on early stopping rather than at --n_epochs, so the
    epoch count of a given run is a property of that run, not of this measurement -- multiply
    per_epoch_flops / per_epoch_time_seconds by the epochs the real run actually reached.
    """
    summary = trainer.profiling_summary(total_epoch_tokens)

    if summary["profiled_steps"] < trainer.profile_steps:
        logger.warning(
            "Only %s/%s profiling steps ran -- FLOPs/token is based on a short sample.",
            summary["profiled_steps"], trainer.profile_steps,
        )
    if summary["timed_steps"] < trainer.time_steps:
        logger.warning(
            "Only %s/%s timing steps ran -- seconds/token is based on a short sample. "
            "Raise --max_steps to at least %s.",
            summary["timed_steps"], trainer.time_steps, trainer.profile_steps + trainer.time_steps,
        )
    # DataParallel: n_gpu > 1 without a launcher means the step was spread over several GPUs
    # while world_size stayed 1, so neither rate describes the single-GPU setup it looks like.
    if trainer.args.n_gpu > 1:
        logger.warning(
            "Measured under DataParallel across %s GPUs (world_size=1). Set CUDA_VISIBLE_DEVICES "
            "to a single GPU, or launch with accelerate, for a meaningful rate.",
            trainer.args.n_gpu,
        )
    summary["n_gpu"] = trainer.args.n_gpu

    if trainer.args.should_save:  # rank 0 only, so the ranks don't race on the same file
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = pathlib.Path(output_dir) / "training_flops_time.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote FLOPs/time metrics to %s", metrics_path)

    logger.info(
        "Per epoch: %.4e FLOPs (%.2f TFLOPs), %.1fs (%.2fh) over %s tokens",
        summary["per_epoch_flops"], summary["per_epoch_flops"] / 1e12,
        summary["per_epoch_time_seconds"], summary["per_epoch_time_seconds"] / 3600,
        f"{total_epoch_tokens:,}",
    )


def main():
    args = create_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    pretraining_data = pathlib.Path(args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.Tokenizer.from_pretrained(tokenizer_path, ontology=ontology)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


    # # <-- critical alias for pickle backward-compatibility
    # import femr.models.tasks as _tasks_mod
    # _tasks_mod.MOTORTask = _MOTORImpl

    # mtpp_factorized is the ORA-Factorized ablation: same task/batches as ORA, different head.
    task_loss_type = "mtpp" if args.loss_type in ("mtpp_shared", "mtpp_factorized") else args.loss_type
    task_path = pretraining_data / f'{task_loss_type}_task.pkl'
    with open(task_path, 'rb') as f:
        task = pickle.load(f)
    if args.loss_type == "survivehr":
        task.time_scale_days = args.survivehr_time_scale_days
        print(f"Using SurviveHR time_scale_days={task.time_scale_days}")
    # print(f"{args.loss_type} task: {task}")
    # print(f"{args.loss_type} task length: {len(task.pretraining_task_codes)}")
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task)
    eval_processor = processor
    if args.loss_type == "icarefm":
        eval_task = copy.deepcopy(task)
        eval_task.threshold_seed = args.icarefm_eval_seed
        eval_processor = femr.models.processor.FEMRBatchProcessor(tokenizer, eval_task)
        print(f"Using ICareFM eval threshold seed={args.icarefm_eval_seed}")

    # Choose dataset packaging based on model
    if args.model == "mamba":
        train_batches_path = pretraining_data / 'train_batches_mamba'
        val_batches_path = pretraining_data / 'val_batches_mamba'
    else:
        train_batches_path = pretraining_data / 'train_batches'
        val_batches_path = pretraining_data / 'val_batches'

    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    profiling = args.profile_steps > 0 or args.time_steps > 0
    # --max_steps marks a run whose only purpose is collecting FLOPs/time. It stops mid-epoch,
    # so epoch-based eval/checkpointing/early-stopping have nothing to act on.
    profile_only = args.max_steps > 0
    # Only paid for when profiling: this is a full pass over the training set.
    total_epoch_tokens = count_epoch_tokens(train_batches) if profiling else 0
    if profiling:
        print(f"Total tokens per epoch: {total_epoch_tokens:,} across {len(train_batches)} batches")

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

    # print(f"Model param count: {count_parameters(model)}")
    param_count = count_parameters(model)
    backbone_count = count_parameters(model.backbone)
    head_count = count_parameters(model.task_model)
    other_count = param_count - backbone_count - head_count
    logger.info(
        "Model param count: total=%s backbone=%s head=%s other=%s",
        param_count, backbone_count, head_count, other_count,
    )
    if wandb.run is not None:
        wandb.run.summary["model/param_count"] = param_count
        wandb.run.summary["model/backbone_param_count"] = backbone_count
        wandb.run.summary["model/head_param_count"] = head_count

    learning_rate = args.learning_rate
    # output_dir = 'tmp_trainer_' + sys.argv[1]
    output_dir = args.output_dir
    # print(args.per_device_train_batch_size)
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,
        # lr_scheduler_type="cosine",

        weight_decay=0.05,
        adam_beta2=0.95,
        # report_to="none",
        report_to="none" if profile_only else "wandb",
        max_grad_norm=args.max_grad_norm,
        run_name=args.hf_run_name,
        # run_name="motor_pretrain_mimic",
        num_train_epochs=args.n_epochs,
        max_steps=args.max_steps,
        ddp_find_unused_parameters=(args.loss_type == "survivehr"),

        warmup_steps=1000,

        logging_strategy='steps',
        logging_steps=args.logging_steps,

        save_strategy='no' if profile_only else 'epoch',
        eval_strategy='no' if profile_only else 'epoch',
        # evaluation_strategy='epoch',

        # prediction_loss_only=True,
        # dataloader_num_workers=1,
        dataloader_num_workers=40,

        save_total_limit=10,
        load_best_model_at_end=not profile_only,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    print(f"CONFIRMATION: The actual per_device_train_batch_size is {trainer_config.per_device_train_batch_size}")
    print(f"CONFIRMATION: The actual gradient_accumulation_steps is {trainer_config.gradient_accumulation_steps}")
    if profile_only:
        print(f"PROFILE-ONLY MODE: stopping after {args.max_steps} steps, no eval/save/early-stopping")

    callbacks = []
    save_best_callback = None
    if not profile_only:
        if args.best_model_dir:
            best_model_dir = pathlib.Path(args.best_model_dir)
        else:
            best_model_dir = pathlib.Path(output_dir) / "best_model"
        # Same normalisation the Trainer applies to metric_for_best_model before looking it up.
        best_metric_name = trainer_config.metric_for_best_model
        if not best_metric_name.startswith("eval_"):
            best_metric_name = f"eval_{best_metric_name}"
        save_best_callback = SaveBestModelCallback(
            best_model_dir,
            tokenizer=tokenizer,
            metric_name=best_metric_name,
            greater_is_better=trainer_config.greater_is_better,
        )
        callbacks = [
            CustomEarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
            save_best_callback,
        ]
    if args.loss_type in ("survivehr", "icarefm"):
        callbacks.append(SurviveHRMetricsCallback())

    trainer = SeparateEvalCollatorTrainer(
        model=model,
        data_collator=processor.collate,
        eval_data_collator=eval_processor.collate if args.loss_type == "icarefm" else None,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=callbacks,
        profile_steps=args.profile_steps,
        time_steps=args.time_steps,
    )
    if save_best_callback is not None:
        save_best_callback.trainer = trainer

    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)

    if save_best_callback is not None:
        # load_best_model_at_end=True means trainer.model now holds the lowest-eval_loss weights,
        # whether we stopped early or ran all n_epochs. Rewrite best_model from them so the final
        # artifact is consistent with trainer_state.best_model_checkpoint.
        save_best_callback.best_metric = trainer.state.best_metric
        save_best_callback.save(trainer.state, reason="end of training")
        logger.info(
            "Training finished after %.2f/%s epochs (best checkpoint: %s). Best model -> %s",
            trainer.state.epoch or 0.0, args.n_epochs, trainer.state.best_model_checkpoint, best_model_dir,
        )

    if profiling:
        write_profiling_metrics(trainer, total_epoch_tokens, output_dir)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


# python -u -m femr.training.pretrain_motor \
#   --pretraining_data $PRETRAINING_DATA \
#   --meds_reader $OMOP_MEDS_READER

'''

export CUDA_VISIBLE_DEVICES=2

kuvira

export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch \
  --num_processes 4 \
  pretrain.py \
    --pretraining_data  /data2/processed_datasets/zj2398/femr/mimic/mtpp  \
    --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --output_dir /data2/processed_datasets/zj2398/femr/mimic/mtpp/mamba  \
    --hf_run_name mimic_mamba_revised_mtpp_6100 \
    --max_grad_norm 1.0 \
    --model mamba \
    --n_layers 28 \
    --loss_type mtpp_shared

accelerate launch \
    --num_processes 3 \
    pretrain.py \
    --pretraining_data /user/zj2398/cache/mtpp_8k_T_8_V_5_all_nonnume \
    --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --per_device_train_batch_size 1 \
    --output_dir /user/zj2398/cache/mtpp_8k_T_8_V_5_all_nonnume/transformer_test  \
    --model transformer \
    --loss_type mtpp_shared \
    --hf_run_name mimic_mtpp_all_nonnumerical_test


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



python pretrain.py \
    --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/mtpp_T_10_V_5  \
    --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
    --per_device_train_batch_size 1 \
    --output_dir /data2/processed_datasets/zj2398/femr/cumc/mtpp_T_10_V_5/transformer \
    --model transformer \
    --checkpoint_dir /data2/processed_datasets/zj2398/femr/cumc/mtpp_T_10_V_5/transformer/checkpoint-144177 \
    --loss_type mtpp_shared \
    --hf_run_name cumc_mtpp_T_10_V_5_transformer

python pretrain.py \
    --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/icarefm \
    --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader  \
    --output_dir /data2/processed_datasets/zj2398/femr/cumc/icarefm/transformer \
    --model transformer \
    --loss_type icarefm \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --hf_run_name cumc_icarefm_transformer_value_1000



'''
