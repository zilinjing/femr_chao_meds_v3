import numpy as np
import transformers
import pathlib
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
from femr.omop_meds_tutorial.generate_labels import create_omop_meds_tutorial_arg_parser
import torch.nn as nn
import os # Import the os module

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
        "--n_layers",
        dest="n_layers",
        type=int,
        default=11
    )
    arg_parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=50
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
    # Add a new argument for the GPU device ID
    arg_parser.add_argument(
        "--gpu_id",
        dest="gpu_id",
        type=int,
        default=0, # Default to GPU 0 if not specified
        help="The ID of the GPU to use for training (e.g., 0, 1, 5)."
    )
    return arg_parser

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = create_arg_parser().parse_args()

    # --- REVISION 1: Set CUDA_VISIBLE_DEVICES ---
    # This is crucial. It tells the CUDA runtime which GPUs are "visible".
    # By setting it to the chosen GPU ID, other GPUs will not be initialized
    # or used by PyTorch, preventing accidental usage of GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Verify that PyTorch sees only one device and it's device 0 (which maps to args.gpu_id)
    if torch.cuda.is_available():
        print(f"PyTorch sees {torch.cuda.device_count()} CUDA device(s).")
        if torch.cuda.device_count() > 0:
            print(f"Current PyTorch device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training on CPU.")


    pretraining_data = pathlib.Path(args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        tokenizer_path, ontology=ontology
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)
    print(f"Motor task: {motor_task}")
    print(f"Motor task length: {len(motor_task.pretraining_task_codes)}")
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    print(f"Train batches length: {len(train_batches)}, batch : {train_batches}")

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        n_layers=args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    print(f"Transformer config: {transformer_config}")
    model = femr.models.transformer.FEMRModel(config,attn_implementation="flash_attention_2")

    # --- REVISION 2: Use "cuda:0" after setting CUDA_VISIBLE_DEVICES ---
    # Since CUDA_VISIBLE_DEVICES is set, the GPU specified by args.gpu_id
    # will now be device 0 within the PyTorch context.
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to device: {device}")


    print(f"Model param count: {count_parameters(model)}")

    learning_rate = args.learning_rate
    output_dir = args.output_dir
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,
        weight_decay=0.1,
        adam_beta2=0.95,
        report_to="none",
        num_train_epochs=args.n_epochs,
        ddp_find_unused_parameters=False,
        warmup_steps=500,
        logging_strategy='epoch',
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='epoch',
        dataloader_num_workers=32,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --- REVISION 3: Explicitly set `dataloader_pin_memory_device` (if using pin_memory) ---
        # While not strictly necessary for single GPU training if `CUDA_VISIBLE_DEVICES` is set,
        # if `pin_memory=True` is used in data loaders, this ensures pinned memory is allocated
        # on the correct device. The `Trainer` sets `pin_memory=True` by default.
        dataloader_pin_memory_device=f"cuda:{0}" if torch.cuda.is_available() else None,

    )
    print(f"CONFIRMATION: The actual per_device_train_batch_size is {trainer_config.per_device_train_batch_size}")


    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        # eval_dataset=val_batches,
        args=trainer_config,
        callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)],
    )

    # The profiler setup needs to be adjusted. If you want to profile
    # only the training loop, move it around the trainer.train() call.
    # The current setup only profiles a few manual steps, not the entire training.
    # For single GPU training, you might not even need the profiler unless
    # you are debugging performance bottlenecks specific to CUDA operations.
    # I'll leave it as is for now, but be aware of its scope.
    # with torch.profiler.profile(
    # activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir/profile'),
    # record_shapes=True,
    # with_stack=True
    # ) as prof:
    #     for step, batch in enumerate(trainer.get_train_dataloader()):
    #         if step >= (1 + 1 + 3): # wait, warmup, active steps
    #             break
    #         # Manually perform a training step
    #         loss = trainer.training_step(model, batch)
    #         prof.step() # Advance the profiler to the next step

    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


'''
python pretrain_motor_old.py --gpu_id 5 \
  --pretraining_data /user/zj2398/cache/motor \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/motor/output
'''