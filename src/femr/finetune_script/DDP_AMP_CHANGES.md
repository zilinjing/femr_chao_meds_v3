# DDP + AMP fine-tuning — changes, design & run guide

> Status: **implemented & verified.** Two **independent** knobs were added to `finetuning.py` and
> `run_motor_finetune.sh`:
> - **DDP** (`--num_gpus N`, N>1) — multi-GPU data parallelism via `torchrun`, for **speed**.
> - **AMP bf16** (`--amp`) — mixed precision, for **memory + speed**.
>
> Use either, both, or neither. With both off, behavior is **byte-identical to before**. Batch
> *building* stays single-process; only the **training loop** is parallelized.
>
> Companion doc: **`AMP_ATTENTION_NOTES.md`** explains the bf16/autocast and xformers→FlashAttention
> concepts behind `--amp`.

---

## 1. Quick start (sample commands)

First activate the env and `cd` in:
```bash
source /home/t-zilinjing/tte/bin/activate
cd /home/t-zilinjing/femr_chao_meds_v3/src/femr/omop_meds_tutorial/finetune_script
```

Pick **free** GPUs with `CUDA_VISIBLE_DEVICES` (GPU 0 on this box is often occupied):

| Run | Command |
|-----|---------|
| fp32, 1 GPU (unchanged) | `CUDA_VISIBLE_DEVICES=1 bash run_motor_finetune.sh --task stroke` |
| **AMP** bf16, 1 GPU | `CUDA_VISIBLE_DEVICES=1 bash run_motor_finetune.sh --task stroke --amp` |
| **DDP**, 2 GPUs | `CUDA_VISIBLE_DEVICES=1,2 bash run_motor_finetune.sh --task stroke --num_gpus 2` |
| **DDP + AMP**, 3 GPUs | `CUDA_VISIBLE_DEVICES=1,2,3 bash run_motor_finetune.sh --task stroke --num_gpus 3 --amp` |

`--num_gpus` omitted → 1 → plain `python`. `--amp` omitted → fp32. Both are orthogonal: any of the
four combinations works.

Useful extra flags (forwarded to `finetuning.py`):
```bash
# W&B logging (rank-0 only under DDP)
... --use_wandb --wandb_project ehr_benchmark_finetune_mimic

# fit a smaller batch into memory (per-GPU activation memory lever)
... --tokens_per_batch 32768

# fast smoke test: 1 LR cell, 1 epoch, write to a non-clobbering suffix
... --num_epochs 1 --lr_ratios 0.1 --head_lr_ratios 1 --output_suffix _test

# tiny end-to-end debug run (rebuilds a tiny cohort; ignores the disk cache)
... --debug
```

Pick the physical GPUs via `CUDA_VISIBLE_DEVICES`; `torchrun` maps local ranks onto the visible set.

---

## 2. What DDP adds (`--num_gpus N`, N>1)
- Launches via `torchrun --standalone --nproc_per_node=N`; native `torch.distributed` (NCCL). N
  processes (ranks), one GPU each.
- **Training only** is sharded with a `DistributedSampler` over the pre-built batches
  (`shuffle=False`, matching the build-time pre-shuffle; `drop_last=False` pads → **equal steps/rank
  → no all-reduce hang**). Each rank holds a full model copy; gradients are all-reduced in
  `backward()`.
- The model is wrapped `DDP(..., find_unused_parameters=True)` — **required** (see §4).
- **Batch building stays single-process.** Under DDP, rank 0 builds the HF cache first behind a
  barrier; the others wait then read it (no concurrent-write race). If the cache already exists every
  rank just `load_from_disk`s it and the barrier is a no-op.
- **Validation, test inference, and all file saving run on rank 0 only.** `val_loss` is broadcast so
  every rank makes the same early-stop decision. Saved `*_raw.parquet` / `*_model.pt` /
  `*_metrics.json` are identical to the single-GPU path (so `dedup.py` is unaffected).

## 3. What AMP adds (`--amp`)
- Wraps **forward + loss** in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` in both
  `train_epoch` and `evaluate`. Backward, optimizer, scheduler, and master weights stay **fp32**.
- **No `GradScaler`** — bf16 shares fp32's exponent range (a scaler is an fp16-only thing). Safe with
  the model's RMSNorm (it upcasts internally) and native on A100.
- `enabled=use_amp` → a **no-op when `--amp` is absent** (default path byte-identical to before).
- Independent of DDP: it only touches the forward call sites, never the DDP wrap / sampler /
  barriers. (Concept details in `AMP_ATTENTION_NOTES.md`.)

## 4. Memory & throughput (which knob for what)
- **DDP = speed, not memory.** Each rank still processes a full `tokens_per_batch=65536` batch, so
  DDP alone does **not** fix an OOM — every rank would OOM identically. Throughput ≈ N× batches/step,
  ~`len/N` optimizer steps per epoch (effective batch scales with N; optimization dynamics shift
  slightly vs single-GPU).
- **AMP = the memory lever** (~halves activation memory) **and** a large speedup: bf16 unlocks
  tensor-core matmuls **and** the xformers FlashAttention kernel (fp32 can use neither). Measured on
  A100/masld: **fp32 ≈ 3.0 s/it vs bf16 ≈ 0.62 s/it (~5×)**.
- If still OOM: lower `--tokens_per_batch` (e.g. 32768). And run on free GPUs
  (`CUDA_VISIBLE_DEVICES=1,2,3`) to dodge the contended GPU 0.
- Pretraining used `bf16=True`, so `--amp` also makes fine-tuning numerically closer to the
  pretraining regime.

---

## 5. Design details & correctness traps
Three traps drove the design — worth knowing if you modify this code:

1. **Shared HF batch-cache race.** `convert_dataset` builds via
   `datasets.Dataset.from_generator(..., cache_dir=self.cache_path)` (`processor.py:582`). Output is
   deterministic (seed 342342) but unguarded — under `torchrun` every rank would write the same cache
   concurrently. **Fix:** rank-0 builds first; other ranks wait on a `dist.barrier()`, then read from
   cache. (No-op when the cache already exists.)
2. **Lock-step control flow.** The grid early-stops on validation loss; every rank must take the
   *same* early-stop/best-epoch branch or DDP's gradient all-reduce hangs on the next epoch. **Fix:**
   rank 0 computes `val_loss` and `dist.broadcast`s it; all ranks branch on that one value.
3. **The dropped task head leaves unused parameters.**
   `load_femr_model_with_compat(..., drop_task_head=True)` only drops checkpoint *keys* — it does
   **not** delete the `FEMRModel.task_model` submodule. Those params stay registered but get **no
   gradient** (`FineTuneModel.forward` uses only `self.backbone.backbone` + `self.classifier`).
   Default DDP (`find_unused_parameters=False`) therefore **hangs/errors** on the first backward.
   **Fix:** create DDP with **`find_unused_parameters=True`**. We *keep* `task_model` (rather than
   deleting it) so `best_model_state = base_model.state_dict()` keeps the same keys as the fresh
   `FineTuneModel` rebuilt at test time (strict `load_state_dict`).

Also: the model uses **RMSNorm, not BatchNorm** → no `SyncBatchNorm` needed; and no Dropout → train/
eval modes are simple.

---

## 6. Exactly what changed

### `finetuning.py`
1. **Imports:** `torch.distributed as dist`, `DistributedDataParallel as DDP`, `DistributedSampler`.
2. **Distributed init** (replaces `device = torch.device(args.device)`): reads
   `WORLD_SIZE/RANK/LOCAL_RANK` (set by `torchrun`). `WORLD_SIZE` unset → `distributed=False`,
   `is_main=True`, `device = args.device` → **byte-identical to the old single-process path.**
   Distributed → `init_process_group("nccl")`, `set_device(local_rank)`, `device = cuda:local_rank`.
3. **`--amp` flag** (`action="store_true"`): enables bf16 autocast; independent of DDP.
4. **Rank-0-first cache barrier** bracketing the **three** `get_or_create_batches` calls (train/val/
   test all hit the same `cache_path`). `get_or_create_batches` itself is unchanged, so an existing
   `<output_dir>/cache/{train,val,test}_batches` is reused as-is.
5. **Train loader sharding:** `train_sampler = DistributedSampler(train_batches, shuffle=False)` when
   distributed; `steps_per_epoch = len(train_loader)` (per-rank). val/test loaders stay plain (full
   set, rank-0 inference). `shuffle=False` mirrors single-GPU semantics (batches are already globally
   pre-shuffled at build time) and needs no `set_epoch`. `drop_last=False` pads to equal steps/rank.
   `warmup_steps`/`total_training_steps` derive from `steps_per_epoch` → per-rank (intended).
6. **Grid loop:** keep an unwrapped `base_model`; wrap a separate
   `train_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank,
   find_unused_parameters=True)`. Build the optimizer from `base_model.backbone` /
   `base_model.classifier` (same Parameter objects DDP syncs). Train with `train_model`.
7. **bf16 autocast** wraps only forward + loss in `train_epoch` and `evaluate`
   (`torch.autocast("cuda", torch.bfloat16, enabled=use_amp)`); backward/optimizer stay fp32; no
   `GradScaler`; `evaluate` takes softmax in fp32. No-op when `--amp` is off.
8. **Validation + broadcast:** rank 0 runs `evaluate(base_model, …)`; `val_loss` is broadcast. Counter
   / early-stop / `break` run on all ranks (identical); the best-weights CPU clone
   (`base_model.state_dict()`), per-cell prints, wandb, `all_results`, and global-best tracking are
   **rank-0 only**.
9. **wandb** is rank-0 only everywhere; per-step logging inside `train_epoch` gets
   `use_wandb=(args.use_wandb and is_main)`. `tqdm` bars disabled on non-main ranks
   (`disable_tqdm=not is_main`).
10. **Test + save:** after the grid, `if distributed: dist.barrier()`; non-main ranks then
    `destroy_process_group()` and `return` (the test/save block has **no collectives**). Rank 0 runs
    the existing test eval + writes the three output files. At the end of `main`, rank 0
    `destroy_process_group()`.

### `run_motor_finetune.sh`
- `NUM_GPUS=1` default + `--num_gpus) NUM_GPUS="$2"; shift 2 ;;` parser arm. The explicit arm keeps
  `--num_gpus` **out of** `EXTRA_ARGS`, so it is not forwarded to `finetuning.py` (which has no such
  flag).
- Launcher: `LAUNCHER=(torchrun --standalone --nproc_per_node="$NUM_GPUS" -m)` when `NUM_GPUS>1`, else
  `(python -u -m)`. All existing `--flags` + `${EXTRA_ARGS[@]}` preserved.
- `--device` is still passed (ignored under DDP; device comes from `LOCAL_RANK`).
- **AMP needs no shell change:** `--amp` is unknown to the parser, so the `*)` catch-all forwards it.

---

## 7. Cache reuse
- The disk batch cache lives at `<output_dir>/cache/{train,val,test}_batches` and is **reused as-is**
  by the new code — none of the DDP/AMP changes touch batch building.
- The cache is **path-keyed and content-blind** (it only checks the directory exists). Keep the same
  `--output_dir`, `--tokens_per_batch`, `--min_subjects_per_batch`, and `--observation_window` to
  reuse it validly; changing a batch-affecting flag while reusing the dir loads **stale** batches.
- `--debug` ignores this cache (rebuilds tiny subsampled batches each run).

---

## 8. Caveats
- **Effective batch scales with #GPUs** — N GPUs ≈ N× global batch and ~`len/N` steps/epoch at the
  same LR; optimization dynamics shift slightly vs single-GPU.
- Under DDP, each rank spawns its own `--num_proc` DataLoader workers (N × num_proc total).
- Don't "simplify" `clip_grad_norm_(model.parameters())` inside `train_epoch` to `base_model.*` — in
  that scope `model` is the DDP handle, whose `.parameters()` correctly forwards to the wrapped module.

---

## 9. Verification (and observed results)
Checks run from the `finetune_script/` dir with `tte` active:
1. `python -m py_compile finetuning.py`; `bash -n run_motor_finetune.sh`; missing-`--task` prints
   usage and exits 1. — **pass.**
2. **Single-GPU regression** (`--num_gpus` omitted → no `torchrun`, `distributed=False`): on the
   masld cache it loaded the cache and ran the full **2569** unsharded steps on the original code
   path. — **pass** (non-distributed path unchanged).
3. **DDP + AMP, 2 GPUs, end-to-end** on the masld cache: cache reused (no rebuild), sharded **1285
   batches/rank** (=ceil 2569/2), both GPUs ~100%, `val_loss` broadcast + early-stop, strict
   `load_state_dict` at test time OK, rank-0 wrote `*_raw.parquet`/`*_model.pt`/`*_metrics.json`,
   clean NCCL teardown (exit 0), `TEST AUROC 0.715`. — **pass.**
4. Speedup confirmed: fp32 ≈ 3.0 s/it vs bf16 (`--amp`) ≈ 0.62 s/it on A100.
5. Run `dedup.py` on a raw output to confirm the saved schema is unchanged.

To smoke-test all four combos quickly, add `--debug` (1 LR cell, 1 epoch, tiny cohort):
```bash
CUDA_VISIBLE_DEVICES=1   bash run_motor_finetune.sh --task stroke --debug          # fp32, 1 GPU
CUDA_VISIBLE_DEVICES=1   bash run_motor_finetune.sh --task stroke --debug --amp    # AMP, 1 GPU
CUDA_VISIBLE_DEVICES=1,2 bash run_motor_finetune.sh --task stroke --num_gpus 2 --debug         # DDP
CUDA_VISIBLE_DEVICES=1,2 bash run_motor_finetune.sh --task stroke --num_gpus 2 --debug --amp   # DDP+AMP
```

---

## 10. Files
- `finetuning.py` — distributed init, `--amp` flag, cache barrier, `DistributedSampler`, DDP wrap +
  optimizer from base submodules, bf16 autocast, rank-0 val/broadcast/save, teardown.
- `run_motor_finetune.sh` — `--num_gpus` flag + torchrun launcher (`--amp` forwarded automatically).
- `AMP_ATTENTION_NOTES.md` — companion: bf16 AMP / autocast scope / xformers→FlashAttention concepts.
