from __future__ import annotations

import argparse
import pathlib
import pickle

import datasets
import transformers
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import femr.models.config
from femr.models.architecture.embedding_survivehr import FEMRModel
from femr.models.processor import FEMRBatchProcessor
from femr.models.tokenizer import HierarchicalTokenizer

datasets.disable_caching()


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Pretrain FEMR with a SurvivEHR-style causal head.")
    parser.add_argument("--pretraining_data", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--model", choices=["transformer", "mamba"], required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--hf_run_name", type=str, default="survivehr_causal")
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_config(args, tokenizer, task):
    conf_dir = pathlib.Path(__file__).parent / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        hydra_cfg = compose(config_name="config", overrides=[f"model={args.model}"])

    model_cfg_dict = OmegaConf.to_container(hydra_cfg.model, resolve=True)
    if args.n_layers is not None:
        model_cfg_dict["n_layers"] = args.n_layers

    if args.model == "transformer":
        model_config = femr.models.config.FEMRTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            is_hierarchical=True,
            hidden_size=model_cfg_dict.get("hidden_size"),
            intermediate_size=model_cfg_dict.get("intermediate_size"),
            n_heads=model_cfg_dict.get("n_heads"),
            n_layers=model_cfg_dict.get("n_layers"),
            attention_width=model_cfg_dict.get("attention_width"),
            use_normed_ages=model_cfg_dict.get("use_normed_ages"),
            use_bias=model_cfg_dict.get("use_bias"),
            hidden_act=model_cfg_dict.get("hidden_act"),
        )
    else:
        model_config = femr.models.config.FEMRMambaConfig(
            vocab_size=tokenizer.vocab_size,
            is_hierarchical=True,
            hf_name=model_cfg_dict.get("hf_name"),
            hidden_size=model_cfg_dict.get("hidden_size"),
            intermediate_size=model_cfg_dict.get("intermediate_size"),
            n_layers=model_cfg_dict.get("n_layers"),
            d_state=model_cfg_dict.get("d_state"),
            use_normed_ages=model_cfg_dict.get("use_normed_ages"),
            use_bias=model_cfg_dict.get("use_bias"),
            config_kwargs=model_cfg_dict.get("config_kwargs") or {},
        )

    return femr.models.config.FEMRModelConfig.from_task_configs(model_config, task.get_task_config())


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ontology_path = pretraining_data / "ontology.pkl"
    tokenizer_path = pretraining_data / "tokenizer"
    task_path = pretraining_data / "survivehr_causal_task.pkl"
    if args.model == "mamba":
        train_batches_path = pretraining_data / "train_batches_survivehr_causal_mamba"
        val_batches_path = pretraining_data / "val_batches_survivehr_causal_mamba"
    else:
        train_batches_path = pretraining_data / "train_batches_survivehr_causal"
        val_batches_path = pretraining_data / "val_batches_survivehr_causal"

    if not ontology_path.exists():
        raise FileNotFoundError(f"Expected ontology at {ontology_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Expected tokenizer at {tokenizer_path}")
    if not task_path.exists():
        raise FileNotFoundError(f"Expected prepared task at {task_path}")
    if not train_batches_path.exists() or not val_batches_path.exists():
        raise FileNotFoundError(
            f"Prepared batches not found. Expected {train_batches_path} and {val_batches_path}."
        )

    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)
    tokenizer = HierarchicalTokenizer.from_pretrained(tokenizer_path, ontology=ontology)
    with open(task_path, "rb") as f:
        task = pickle.load(f)

    processor = FEMRBatchProcessor(tokenizer, task)
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))
    train_batches.set_format("pt")
    val_batches.set_format("pt")

    config = build_model_config(args, tokenizer, task)
    model = FEMRModel(config, loss_type="survivehr_causal")

    trainer_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        output_dir=str(output_dir),
        remove_unused_columns=False,
        bf16=args.bf16,
        weight_decay=0.05,
        adam_beta2=0.95,
        report_to=args.report_to,
        max_grad_norm=args.max_grad_norm,
        run_name=args.hf_run_name,
        num_train_epochs=args.n_epochs,
        max_steps=args.max_steps,
        ddp_find_unused_parameters=False,
        warmup_steps=1000,
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_num_workers=args.dataloader_num_workers,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_args,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model param count: {count_parameters(model)}")
    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)
    with open(final_model_dir / "survivehr_causal_task.pkl", "wb") as f:
        pickle.dump(task, f)


if __name__ == "__main__":
    main()
