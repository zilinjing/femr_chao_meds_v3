"""
End-to-end fine-tuning for MOTOR model.

APPROACH (from Table 7 - FoMoH paper):
- Grid search: 
  - Finetuned LR / Pretrained LR: {0.05, 0.1, 0.2, 0.3, 0.5}
  - Classification Head LR / Finetuned LR: {1, 2, 5}
  - Total: 15 combinations
- Warmup: 1 epoch
- Scheduler: Linear decay
- Maximum training epochs: 10
- Early stopping patience: 3
- Early stopping metric: Validation Log-loss
- Overall selection: Validation Log-loss
- Seed: 42
"""

import argparse
import os
import pickle
import datetime
import glob
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import meds
import meds_reader
import sklearn.metrics
import wandb

import femr.models.tokenizer
import femr.splits
import femr.models.processor as processor_local
from femr.omop_meds_tutorial.finetune_script.task_local import LabeledSubjectTaskWithLabels
from femr.models.architecture.embedding import FEMRModel, to_device
from femr.models.architecture.rewrite_stat_dict import load_femr_model_with_compat

SEED = 42
OBSERVATION_WINDOW = 730  # 2 years, matching linear probing
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class FineTuneModel(nn.Module):
    def __init__(self, backbone: FEMRModel, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, batch):
        batch = self._remove_first_dimension(batch)
        s = torch.zeros_like(batch['subject_ids'], device=batch['subject_ids'].device)
        s[1:] = batch['subject_ids'][1:] != batch['subject_ids'][:-1]
        s = torch.cumsum(s, dim=0).type(torch.uint8)
        features = self.backbone.backbone(batch["transformer"], s)
        features = features.reshape(-1, features.shape[-1])
        label_indices = batch["transformer"]["label_indices"]
        features = features[label_indices, :]
        logits = self.classifier(features)
        return logits, features
    
    def _remove_first_dimension(self, data):
        if isinstance(data, dict):
            return {k: self._remove_first_dimension(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            if data.shape[0] == 1:
                return data.squeeze(dim=0)
            return data
        elif isinstance(data, np.ndarray):
            if data.shape[0] == 1:
                return np.squeeze(data, axis=0)
            return data
        else:
            return data


def read_recursive_parquet(root_dir: str) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(root_dir, '**', '*.parquet'), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {root_dir}")
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
    return df


def load_cohort(cohort_dir: str) -> pd.DataFrame:
    if os.path.isdir(cohort_dir):
        cohort = read_recursive_parquet(cohort_dir)
    else:
        ext = os.path.splitext(cohort_dir)[1].lower()
        if ext == ".parquet":
            cohort = pd.read_parquet(cohort_dir)
        elif ext == ".csv":
            cohort = pd.read_csv(cohort_dir)
        else:
            raise RuntimeError(f"Unknown file extension: {ext}")
    if len(cohort) > 0 and isinstance(cohort.prediction_time.iloc[0], datetime.date):
        cohort["prediction_time"] = pd.to_datetime(cohort["prediction_time"])
    return cohort


def load_presplit_cohort(cohort_dir: str):
    """Load cohort with pre-split files (train.parquet, tuning.parquet, held_out.parquet)."""
    cols = ['subject_id', 'prediction_time', 'boolean_value']
    train = pd.read_parquet(os.path.join(cohort_dir, 'train.parquet'), columns=cols)
    val = pd.read_parquet(os.path.join(cohort_dir, 'tuning.parquet'), columns=cols)
    test = pd.read_parquet(os.path.join(cohort_dir, 'held_out.parquet'), columns=cols)
    for df in [train, val, test]:
        if len(df) > 0 and isinstance(df.prediction_time.iloc[0], datetime.date):
            df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    return train, val, test


def create_labels_from_cohort(cohort: pd.DataFrame) -> List[meds.Label]:
    return [
        meds.Label(
            subject_id=row["subject_id"],
            prediction_time=row["prediction_time"],
            boolean_value=row["boolean_value"],
        )
        for row in cohort.to_dict(orient="records")
    ]


def get_subject_ids_from_labels(labels: List) -> List:
    if len(labels) == 0:
        return []
    first = labels[0]
    if isinstance(first, dict):
        return list(set(l["subject_id"] for l in labels))
    else:
        return list(set(l.subject_id for l in labels))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                global_step=0, logging_steps=1000, use_wandb=False, debug=False):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if "batch" in batch:
            batch = batch["batch"]
        labels = batch["task"]["labels"]
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)
        if len(labels.shape) > 1 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        labels = labels.long()
        batch_size = labels.shape[0]
        batch = to_device(batch, device)
        labels = labels.to(device)

        if debug and total_samples == 0:
            print(f"\n===== DEBUG: Training Batch =====")
            print(f"batch keys: {batch.keys()}")
            print(f"task keys: {batch['task'].keys()}")
            print(f"labels shape: {labels.shape}")
            print(f"labels unique values: {torch.unique(labels)}")
            print(f"=================================\n")

        optimizer.zero_grad()
        logits, _ = model(batch)
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"Shape mismatch: logits {logits.shape[0]} vs labels {labels.shape[0]}")
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        # Measure (but do not clip) the gradient norm; max_norm=inf leaves grads untouched.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        global_step += 1
        if use_wandb and wandb.run is not None and global_step % logging_steps == 0:
            lrs = scheduler.get_last_lr()  # [backbone_lr, head_lr]
            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": float(grad_norm),
                "train/backbone_lr": lrs[0],
                "train/head_lr": lrs[1] if len(lrs) > 1 else lrs[0],
                "train/epoch": epoch,
            }, step=global_step)

    return total_loss / total_samples, global_step


def evaluate(model, dataloader, device, debug=False, collect_metadata=False):
    """Evaluate model. If collect_metadata=True, also return subject_ids, prediction_timestamps, and event_timestamps from task data."""
    model.eval()
    all_preds = []
    all_labels = []
    all_subject_ids = []
    all_prediction_timestamps = []
    all_event_timestamps = []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if "batch" in batch:
                batch = batch["batch"]
            
            labels = batch["task"]["labels"]
            if isinstance(labels, np.ndarray):
                labels = torch.tensor(labels)
            if len(labels.shape) > 1 and labels.shape[0] == 1:
                labels = labels.squeeze(0)
            labels = labels.long()
            batch_size = labels.shape[0]
            
            if collect_metadata:
                # Read metadata from task data (set by task_local.py)
                for field, collector in [
                    ("subject_ids", all_subject_ids),
                    ("prediction_timestamps", all_prediction_timestamps),
                    ("event_timestamps", all_event_timestamps),
                ]:
                    data = batch["task"][field]
                    if isinstance(data, torch.Tensor):
                        if data.dim() > 1 and data.shape[0] == 1:
                            data = data.squeeze(0)
                        data = data.cpu().numpy()
                    elif isinstance(data, np.ndarray):
                        if data.ndim > 1 and data.shape[0] == 1:
                            data = np.squeeze(data, 0)
                    collector.extend([int(x) for x in data])
            
            batch = to_device(batch, device)
            labels_device = labels.to(device)
            
            if debug and idx == 0:
                print(f"\n===== DEBUG: Eval Batch =====")
                print(f"labels shape: {labels.shape}")
                print(f"task keys: {batch['task'].keys()}")
                print(f"=============================\n")
            
            logits, _ = model(batch)
            loss = F.cross_entropy(logits, labels_device)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            probs = F.softmax(logits, dim=1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    if collect_metadata:
        return (np.array(all_preds), np.array(all_labels), avg_loss,
                all_subject_ids, all_prediction_timestamps, all_event_timestamps)
    return np.array(all_preds), np.array(all_labels), avg_loss


def main():
    parser = argparse.ArgumentParser(description="End-to-end fine-tuning for MOTOR")
    parser.add_argument("--meds_reader", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--ontology_path", required=True)
    parser.add_argument("--main_split_path", required=True)
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--tokens_per_batch", type=int, default=65536)
    parser.add_argument("--min_subjects_per_batch", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--pretraining_lr", type=float, default=0.00001)
    parser.add_argument("--loss_type", default="labeled_subjects")
    parser.add_argument("--lr_ratios", default=None, help="Comma-separated LR ratios, e.g. 0.05,0.5")
    parser.add_argument("--head_lr_ratios", default=None, help="Comma-separated head LR ratios, e.g. 1,5")
    parser.add_argument("--output_suffix", default="", help="Suffix for output files")
    parser.add_argument("--cache_dir", default=None,
                        help="Local disk for the HF dataset batch cache (~17GB/task). Avoid network FS.")
    parser.add_argument("--observation_window", type=int, default=OBSERVATION_WINDOW,
                        help="Lookback days for labeling; <=0 means full history (None). "
                             "Phenotype tasks (stroke/masld/scz) use 730; outcome tasks (e.g. readmission) use <=0.")
    parser.add_argument("--debug", action="store_true", help="Run on small subset for debugging")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging (one run per LR config).")
    parser.add_argument("--wandb_project", default="ehr_benchmark_finetune_mimic",
                        help="W&B project name to log runs under.")
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B entity (team/user). Defaults to your wandb default.")
    parser.add_argument("--logging_steps", type=int, default=1000,
                        help="Log train metrics (loss/grad_norm/lr) every N optimizer steps.")
    args = parser.parse_args()

    obs_window = args.observation_window if args.observation_window > 0 else None

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Task: {args.task_name}")
    print(f"Seed: {SEED}")
    print(f"Observation window: {obs_window} days")
    print(f"Early stopping patience: {args.patience}")
    print(f"Early stopping metric: Validation Log-loss")
    if args.debug:
        print("*** DEBUG MODE ***")
    
    if args.lr_ratios:
        lr_ratios = [float(x) for x in args.lr_ratios.split(",")]
    else:
        lr_ratios = [0.05,0.1, 0.2, 0.3,0.5]
    if args.head_lr_ratios:
        head_lr_ratios = [float(x) for x in args.head_lr_ratios.split(",")]
    else:
        head_lr_ratios = [1, 2, 5]
    
    if args.debug:
        lr_ratios = [0.1]
        head_lr_ratios = [1]
    
    print(f"LR ratios: {lr_ratios}")
    print(f"Head LR ratios: {head_lr_ratios}")
    print(f"Total combinations: {len(lr_ratios) * len(head_lr_ratios)}")
    print(f"Pretraining LR: {args.pretraining_lr}")
    print(f"Weight decay: {args.weight_decay}")
    
    # ==========================================
    # LOAD COHORT
    # ==========================================
    print(f"Loading cohort from {args.cohort_dir}")
    
    # Check for pre-split files (MIMIC fomoh_subsample format)
    train_path = os.path.join(args.cohort_dir, 'train.parquet')
    if os.path.exists(train_path):
        print("Found pre-split files (train/tuning/held_out)")
        train_cohort, val_cohort, test_cohort = load_presplit_cohort(args.cohort_dir)
    else:
        # CUMC format: single directory with split column or main_split.csv
        cohort = load_cohort(args.cohort_dir)
        print(f"Total cohort size: {len(cohort)}")
        
        if 'split' in cohort.columns:
            print("Using splits from cohort: train, tuning (val), held_out (test)")
            train_cohort = cohort[cohort.split == 'train']
            val_cohort = cohort[cohort.split == 'tuning']
            test_cohort = cohort[cohort.split == 'held_out']
        else:
            print("No split column in cohort, using main_split.csv")
            main_split = femr.splits.SubjectSplit.load_from_csv(args.main_split_path)
            train_cohort_full = cohort[cohort.subject_id.isin(main_split.train_subject_ids)]
            test_cohort = cohort[cohort.subject_id.isin(main_split.test_subject_ids)]
            train_subject_ids_all = train_cohort_full.subject_id.unique()
            np.random.seed(SEED)
            shuffled_ids = np.random.permutation(train_subject_ids_all)
            val_size = int(len(shuffled_ids) * 0.1)
            val_subject_ids_set = set(shuffled_ids[:val_size])
            train_subject_ids_set = set(shuffled_ids[val_size:])
            train_cohort = train_cohort_full[train_cohort_full.subject_id.isin(train_subject_ids_set)]
            val_cohort = train_cohort_full[train_cohort_full.subject_id.isin(val_subject_ids_set)]
    
    num_training = len(train_cohort) + len(val_cohort)
    
    if args.debug:
        print("DEBUG MODE: Using stratified samples")
        train_pos = train_cohort[train_cohort.boolean_value == True].head(100)
        train_neg = train_cohort[train_cohort.boolean_value == False].head(100)
        train_cohort = pd.concat([train_pos, train_neg])
        val_pos = val_cohort[val_cohort.boolean_value == True].head(10)
        val_neg = val_cohort[val_cohort.boolean_value == False].head(10)
        val_cohort = pd.concat([val_pos, val_neg])
        test_pos = test_cohort[test_cohort.boolean_value == True].head(25)
        test_neg = test_cohort[test_cohort.boolean_value == False].head(25)
        test_cohort = pd.concat([test_pos, test_neg])
        num_training = len(train_cohort) + len(val_cohort)
    
    print(f"Train: {len(train_cohort)}, Val: {len(val_cohort)}, Test: {len(test_cohort)}, Filename count (train+val): {num_training}")
    
    train_labels = create_labels_from_cohort(train_cohort)
    val_labels = create_labels_from_cohort(val_cohort)
    test_labels = create_labels_from_cohort(test_cohort)
    
    train_subject_ids = get_subject_ids_from_labels(train_labels)
    val_subject_ids = get_subject_ids_from_labels(val_labels)
    test_subject_ids = get_subject_ids_from_labels(test_labels)
    
    with open(args.ontology_path, 'rb') as f:
        ontology = pickle.load(f)
    
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        args.tokenizer_path, ontology=ontology
    )
    
    num_epochs = 1 if args.debug else args.num_epochs
    
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_proc) as database:
        
        # ==========================================
        # CREATE BATCHES (with observation_window)
        # ==========================================
        print(f"\nCreating training batches (this may take a while)...")
        train_task = LabeledSubjectTaskWithLabels(train_labels, observation_window=obs_window)
        train_processor = processor_local.FEMRBatchProcessor(tokenizer, task=train_task, cache_path=args.cache_dir)
        train_filtered = database.filter(train_subject_ids)
        train_batches = train_processor.convert_dataset(
            train_filtered,
            tokens_per_batch=args.tokens_per_batch,
            min_subjects_per_batch=args.min_subjects_per_batch,
            num_proc=args.num_proc,
        )
        train_batches.set_format("pt")
        num_train_batches = len(train_batches)
        
        print("Creating validation batches...")
        val_task = LabeledSubjectTaskWithLabels(val_labels, observation_window=obs_window)
        val_processor = processor_local.FEMRBatchProcessor(tokenizer, task=val_task, cache_path=args.cache_dir)
        val_filtered = database.filter(val_subject_ids)
        val_batches = val_processor.convert_dataset(
            val_filtered,
            tokens_per_batch=args.tokens_per_batch,
            min_subjects_per_batch=args.min_subjects_per_batch,
            num_proc=args.num_proc,
        )
        val_batches.set_format("pt")
        
        print("Creating test batches...")
        test_task = LabeledSubjectTaskWithLabels(test_labels, observation_window=obs_window)
        test_processor = processor_local.FEMRBatchProcessor(tokenizer, task=test_task, cache_path=args.cache_dir)
        test_filtered = database.filter(test_subject_ids)
        test_batches = test_processor.convert_dataset(
            test_filtered,
            tokens_per_batch=args.tokens_per_batch,
            min_subjects_per_batch=args.min_subjects_per_batch,
            num_proc=args.num_proc,
        )
        test_batches.set_format("pt")

        train_loader = DataLoader(
            train_batches, num_workers=args.num_proc, pin_memory=True,
            collate_fn=train_processor.collate, shuffle=False,
        )
        val_loader = DataLoader(
            val_batches, num_workers=args.num_proc, pin_memory=True,
            collate_fn=val_processor.collate, shuffle=False,
        )
        test_loader = DataLoader(
            test_batches, num_workers=args.num_proc, pin_memory=True,
            collate_fn=test_processor.collate, shuffle=False,
        )
        
        steps_per_epoch = num_train_batches
        total_training_steps = steps_per_epoch * num_epochs
        warmup_steps = steps_per_epoch
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_training_steps}")
        print(f"Warmup steps: {warmup_steps}")
        
        # ==========================================
        # GRID SEARCH
        # ==========================================
        best_val_loss = float('inf')
        best_model_state = None
        best_config = None
        all_results = []
        hidden_size = None
        
        combo_idx = 0
        total_combos = len(lr_ratios) * len(head_lr_ratios)
        
        for lr_ratio in lr_ratios:
            for head_lr_ratio in head_lr_ratios:
                combo_idx += 1
                backbone_lr = args.pretraining_lr * lr_ratio
                head_lr = backbone_lr * head_lr_ratio
                
                print(f"\n{'='*60}")
                print(f"[{combo_idx}/{total_combos}] Backbone LR = {backbone_lr:.2e}, Head LR = {head_lr:.2e}")
                print(f"  (LR ratio: {lr_ratio}, Head LR ratio: {head_lr_ratio})")
                print(f"{'='*60}")

                if args.use_wandb:
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=f"motor_{args.task_name}{args.output_suffix}_lr{lr_ratio}_head{head_lr_ratio}",
                        group=f"{args.task_name}{args.output_suffix}",
                        reinit=True,
                        config={
                            "task_name": args.task_name,
                            "lr_ratio": lr_ratio,
                            "head_lr_ratio": head_lr_ratio,
                            "backbone_lr": backbone_lr,
                            "head_lr": head_lr,
                            "pretraining_lr": args.pretraining_lr,
                            "weight_decay": args.weight_decay,
                            "num_epochs": num_epochs,
                            "warmup_steps": warmup_steps,
                            "total_training_steps": total_training_steps,
                            "observation_window": obs_window,
                            "seed": SEED,
                            "logging_steps": args.logging_steps,
                        },
                    )

                model, _ = load_femr_model_with_compat(
                    args.model_path,
                    model_cls=FEMRModel,
                    task_config=train_task.get_task_config(),
                    loss_type=args.loss_type,
                    drop_task_head=True,
                )
                hidden_size = model.hidden_size
                
                finetune_model = FineTuneModel(model, hidden_size, num_classes=2).to(device)
                
                optimizer = torch.optim.AdamW([
                    {'params': finetune_model.backbone.parameters(), 'lr': backbone_lr},
                    {'params': finetune_model.classifier.parameters(), 'lr': head_lr},
                ], weight_decay=args.weight_decay, betas=(0.9, 0.95))
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps
                )
                
                config_best_val_loss = float('inf')
                config_best_model_state = None
                config_best_epoch = 0
                patience_counter = 0
                global_step = 0

                for epoch in range(num_epochs):
                    train_loss, global_step = train_epoch(
                        finetune_model, train_loader, optimizer, scheduler, device,
                        epoch=epoch, global_step=global_step,
                        logging_steps=args.logging_steps, use_wandb=args.use_wandb,
                        debug=(args.debug and epoch == 0)
                    )
                    val_preds, val_labels_arr, val_loss = evaluate(
                        finetune_model, val_loader, device,
                        debug=(args.debug and epoch == 0)
                    )

                    try:
                        val_auroc = sklearn.metrics.roc_auc_score(val_labels_arr, val_preds)
                    except ValueError:
                        val_auroc = 0.0

                    if args.use_wandb and wandb.run is not None:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/auroc": val_auroc,
                            "train/epoch_loss": train_loss,
                            "epoch": epoch + 1,
                        }, step=global_step)

                    if val_loss < config_best_val_loss:
                        config_best_val_loss = val_loss
                        config_best_model_state = {k: v.cpu().clone() for k, v in finetune_model.state_dict().items()}
                        config_best_epoch = epoch + 1
                        patience_counter = 0
                        print(f"  Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} *** best ***")
                    else:
                        patience_counter += 1
                        print(f"  Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} (patience: {patience_counter}/{args.patience})")
                    
                    if patience_counter >= args.patience:
                        print(f"  Early stopping triggered at epoch {epoch+1}")
                        break
                
                print(f"  Best epoch: {config_best_epoch} with Val Loss: {config_best_val_loss:.4f}")

                if args.use_wandb and wandb.run is not None:
                    wandb.run.summary["best_val_loss"] = config_best_val_loss
                    wandb.run.summary["best_epoch"] = config_best_epoch
                    wandb.finish()

                all_results.append({
                    "lr_ratio": lr_ratio,
                    "head_lr_ratio": head_lr_ratio,
                    "backbone_lr": float(backbone_lr),
                    "head_lr": float(head_lr),
                    "val_loss": float(config_best_val_loss),
                    "best_epoch": config_best_epoch
                })
                
                if config_best_val_loss < best_val_loss:
                    best_val_loss = config_best_val_loss
                    best_model_state = config_best_model_state
                    best_config = {
                        "lr_ratio": lr_ratio,
                        "head_lr_ratio": head_lr_ratio,
                        "backbone_lr": backbone_lr,
                        "head_lr": head_lr
                    }
                    print(f"  *** New global best! ***")
        
        # ==========================================
        # TEST EVALUATION
        # ==========================================
        print(f"\n{'='*60}")
        print(f"Best config: LR ratio={best_config['lr_ratio']}, Head LR ratio={best_config['head_lr_ratio']}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Evaluating on TEST set...")
        print(f"{'='*60}")

        model, _ = load_femr_model_with_compat(
            args.model_path,
            model_cls=FEMRModel,
            task_config=test_task.get_task_config(),
            loss_type=args.loss_type,
            drop_task_head=True,
        )
        finetune_model = FineTuneModel(model, hidden_size, num_classes=2).to(device)
        finetune_model.load_state_dict(best_model_state)

        test_results = evaluate(finetune_model, test_loader, device, collect_metadata=True)
        test_preds, test_labels_arr, test_loss, test_sids, test_pred_ts, test_event_ts = test_results

        try:
            test_auroc_raw = sklearn.metrics.roc_auc_score(test_labels_arr, test_preds)
        except ValueError:
            print("Warning: Only one class in test set")
            test_auroc_raw = 0.0

        print(f"TEST Loss (raw): {test_loss:.4f}")
        print(f"TEST AUROC (raw): {test_auroc_raw:.4f}")
        print(f"TEST predictions (raw): {len(test_preds)}")

        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.task_name}{args.output_suffix}_summary",
                group=f"{args.task_name}{args.output_suffix}",
                reinit=True,
                config={"best_config": best_config},
            )
            wandb.log({
                "test/auroc": test_auroc_raw,
                "test/loss": test_loss,
                "best_val_loss": best_val_loss,
            })
            wandb.run.summary.update({
                "test/auroc": test_auroc_raw,
                "test/loss": test_loss,
                "best_val_loss": best_val_loss,
            })
            wandb.finish()

        # ==========================================
        # SAVE RAW RESULTS (apply fix_prediction_times.py afterwards)
        # ==========================================
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        raw_df = pl.DataFrame({
            "subject_id": pl.Series(test_sids, dtype=pl.Int64),
            "prediction_time": pl.Series(test_event_ts, dtype=pl.Int64).cast(pl.Datetime("ns")),
            "predicted_boolean_probability": test_preds.tolist(),
            "predicted_boolean_value": (test_preds >= 0.5).tolist(),
            "boolean_value": test_labels_arr.astype(bool).tolist(),
        })

        output_file = output_path / f"motor_{args.task_name}_{num_training}{args.output_suffix}_raw.parquet"
        raw_df.write_parquet(output_file)
        print(f"Saved raw: {output_file} ({len(raw_df)} rows)")
        print(f"Run fix_prediction_times.py to get final deduped output")

        metrics = {
            "task_name": args.task_name,
            "test_auroc_raw": float(test_auroc_raw),
            "test_loss": float(test_loss),
            "best_val_loss": float(best_val_loss),
            "best_config": best_config,
            "num_training": num_training,
            "num_test_raw": len(test_preds),
            "num_test_expected": len(test_cohort),
            "observation_window": obs_window,
            "all_results": all_results,
            "seed": SEED,
            "patience": args.patience,
            "warmup_steps": warmup_steps,
            "total_training_steps": total_training_steps,
        }

        # Save best model weights
        model_file = output_path / f"motor_{args.task_name}_{num_training}{args.output_suffix}_model.pt"
        torch.save(best_model_state, model_file)
        print(f"Saved model: {model_file}")
        
        metrics_file = output_path / f"motor_{args.task_name}_{num_training}{args.output_suffix}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {metrics_file}")


if __name__ == "__main__":
    main()