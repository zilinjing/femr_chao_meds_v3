# Add DDP (multi-GPU) to MOTOR fine-tuning — implementation runbook

> Status: **plan only, not yet implemented.** Execute the edits below when ready.
> Targets: `finetuning.py` and `run_motor_finetune.sh` in this directory.

## Context
`finetuning.py` runs a 15-cell grid search with a **hand-written training loop** (not HF
`Trainer`) on a single GPU. Goal: use **2+ GPUs (DDP)** to speed up fine-tuning with **minimum
but clear** revisions. The repo has **no native DDP** (pretraining uses `accelerate`+`Trainer`);
the fine-tune loop is custom, so the cleanest fit is native `torch.distributed` DDP launched via
`torchrun` (present in `tte`, NCCL on, 4×A100 on this cluster).

Two correctness traps drive the design:
1. **Shared HF batch cache.** `FEMRBatchProcessor.convert_dataset` builds batches with
   `datasets.Dataset.from_generator(..., cache_dir=self.cache_path)` (src/femr/models/processor.py:582).
   Output is deterministic (seed 342342, processor.py:527) but **unguarded** — under `torchrun`
   every rank would write the *same* cache files concurrently → race. Fix: rank-0 builds first,
   others wait on a barrier then read from cache.
2. **Lock-step control flow.** The grid search early-stops on validation loss, so every rank must
   make the *same* early-stop/best-epoch decisions or DDP gradient all-reduce will hang. Fix:
   rank 0 computes val loss and **broadcasts** it; all ranks branch on that value.

Scope: only training data is sharded (the expensive part). Validation + test inference + all
file saving happen on rank 0 (cheap; keeps the saved `*_raw.parquet` byte-identical to the
single-GPU path so `dedup.py` is unaffected). Non-DDP runs (`python -m ...`, the current `.sh`
default with `--num_gpus 1`) must keep working unchanged.

---

## Changes to `finetuning.py`

**1. Imports (top of file):**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

**2. Distributed init — replace the `device = torch.device(args.device)` block (~line 290):**
```python
world_size = int(os.environ.get("WORLD_SIZE", "1"))
distributed = world_size > 1
if distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    rank, local_rank = 0, 0
    device = torch.device(args.device)
is_main = rank == 0
```
Guard the existing banner/grid `print(...)` calls with `if is_main:` (selective — the per-epoch,
per-config, and final-summary prints). Add `disable=not is_main` to the `tqdm(...)` bars in
`train_epoch`/`evaluate`.

**3. Batch building — bracket the three `convert_dataset` calls (lines ~392-427) with a
rank-0-first barrier** so the cache is built once and reused:
```python
if distributed and not is_main:
    dist.barrier()                       # wait until rank 0 has populated the cache
# ... existing train/val/test convert_dataset(...) + set_format("pt") unchanged ...
if distributed and is_main:
    dist.barrier()                       # release the other ranks now that the cache exists
```

**4. Train loader — shard with `DistributedSampler`; size the schedule from the per-rank loader:**
```python
train_sampler = DistributedSampler(train_batches, shuffle=True, seed=SEED) if distributed else None
train_loader = DataLoader(train_batches, sampler=train_sampler, num_workers=args.num_proc,
                          pin_memory=True, collate_fn=train_processor.collate)
# val_loader / test_loader stay plain (full set), as now.
steps_per_epoch = len(train_loader)      # per-rank; replaces num_train_batches
```
(`warmup_steps`/`total_training_steps` already derive from `steps_per_epoch`. `DistributedSampler`
defaults to `drop_last=False`, which pads so every rank runs equal steps → no DDP hang.)

**5. Grid loop — keep a reference to the base model, wrap a separate DDP handle, build the
optimizer from the base model's submodules:**
```python
base_model = FineTuneModel(model, hidden_size, num_classes=2).to(device)
train_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank) if distributed else base_model
optimizer = torch.optim.AdamW([
    {'params': base_model.backbone.parameters(),   'lr': backbone_lr},
    {'params': base_model.classifier.parameters(), 'lr': head_lr},
], weight_decay=args.weight_decay, betas=(0.9, 0.95))
```
- Train with `train_model` (DDP); `loss.backward()` is unchanged (DDP syncs grads in backward).
- Before each epoch: `if train_sampler is not None: train_sampler.set_epoch(epoch)`.
- **Validation + broadcast** (rank 0 evaluates `base_model`; everyone branches on the same value):
```python
if is_main:
    val_preds, val_labels_arr, val_loss = evaluate(base_model, val_loader, device)
    try:
        val_auroc = sklearn.metrics.roc_auc_score(val_labels_arr, val_preds)
    except ValueError:
        val_auroc = 0.0
else:
    val_loss = 0.0
if distributed:
    t = torch.tensor([val_loss], device=device); dist.broadcast(t, src=0); val_loss = t.item()
```
- Early-stop / best-epoch logic runs on all ranks off this `val_loss`; only rank 0 stores the
  weights: `if is_main: config_best_model_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}`.
  (Keys match the fresh `FineTuneModel` used at test time.) Global-best tracking
  (`best_model_state`, `best_config`) stays rank-0-only.

**6. Test + save — after the grid loop, barrier then rank-0-only (existing block, unchanged
logic, builds a fresh single-GPU `FineTuneModel` on `device` and loads `best_model_state`):**
```python
if distributed: dist.barrier()
if is_main:
    # ... existing test evaluate + raw.parquet / model.pt / metrics.json saving ...
```

**7. Teardown (end of `main`):** `if distributed: dist.destroy_process_group()`.

---

## Changes to `run_motor_finetune.sh`
Add one flag and switch the launcher when >1 GPU (keep current single-GPU path identical):
```bash
# add to the defaults block:
NUM_GPUS=1

# add to the flag parser (case statement):
--num_gpus) NUM_GPUS="$2"; shift 2 ;;

# replace the final `python -u -m ... finetuning` invocation with:
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER=(torchrun --standalone --nproc_per_node="$NUM_GPUS" -m)
else
    LAUNCHER=(python -u -m)
fi
"${LAUNCHER[@]}" femr.omop_meds_tutorial.finetune_script.finetuning \
    --meds_reader "${MEDS_READER}" \
    ... (same args as now) ... \
    "${EXTRA_ARGS[@]}"
```
`--device` is still passed (ignored in distributed mode; device comes from `LOCAL_RANK`). Which
physical GPUs are used is controlled by `CUDA_VISIBLE_DEVICES`; `torchrun` maps local ranks onto
the visible set.

---

## Launch commands (this cluster)
```bash
source /home/t-zilinjing/tte/bin/activate
cd /home/t-zilinjing/femr_chao_meds_v3/src/femr/omop_meds_tutorial/finetune_script

# 2-GPU smoke test (1 cell) on physical GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 bash run_motor_finetune.sh --task stroke --num_gpus 2 --debug

# full 15-cell grid on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 bash run_motor_finetune.sh --task stroke --num_gpus 2

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_motor_finetune.sh --task stroke --num_gpus 4
```
Single-GPU usage is unchanged (`--num_gpus` omitted → defaults to 1 → plain `python`).

## Caveats
- **Effective batch scales with #GPUs.** DDP averages gradients across ranks, so N GPUs ≈ N×
  the global batch and ~`len/N` optimizer steps per epoch at the same LR. Throughput goes up;
  optimization dynamics shift slightly vs single-GPU. The grid still searches the same LRs.
- Cache (`--cache_dir`) is built once by rank 0 and shared — keep it on local disk as today.
- Per-cell `best_model_state` cloning stays on rank 0 only (no extra GPU memory on other ranks).

## Verification
1. `bash -n run_motor_finetune.sh` (syntax) and missing-`--task` still prints usage.
2. **Single-GPU regression:** `bash run_motor_finetune.sh --task stroke --debug` runs exactly as
   before (distributed path not entered; `WORLD_SIZE` unset).
3. **2-GPU smoke test:** `CUDA_VISIBLE_DEVICES=0,1 bash run_motor_finetune.sh --task stroke --num_gpus 2 --debug`
   → both ranks start, cache builds once (rank 0), grid cell trains, only rank 0 prints epoch
   lines, a single `motor_*_raw.parquet` + `_model.pt` + `_metrics.json` are written, process
   group tears down cleanly (no NCCL hang/timeout). `nvidia-smi` shows both GPUs busy during train.
4. Run `dedup.py` on the smoke-test raw output to confirm the saved schema is unchanged.
