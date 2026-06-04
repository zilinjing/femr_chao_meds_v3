# Mixed precision (bf16 AMP) & attention — MOTOR fine-tuning notes

## TL;DR
- The `--amp` path in `finetuning.py` **is** standard Automatic Mixed Precision (`torch.autocast`),
  identical in mechanism to PyTorch-Lightning `Trainer(precision="bf16")` / HF Trainer `bf16=True`.
- Attention is **always** xformers `memory_efficient_attention`, which **dispatches to a
  FlashAttention kernel in bf16/fp16** and to a slower fp32 kernel otherwise.
- No autocast ⇒ everything fp32 ⇒ **no FlashAttention + no tensor-core matmuls** ⇒ ~5× slower.
  Measured on A100/masld: fp32 ≈ 3.0 s/it vs bf16 ≈ 0.62 s/it.

## 1. bf16 AMP = autocast, and it equals the Trainer
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` is the AMP API. While active it runs
matmul/`nn.Linear`/attention in bf16 (speed + memory) while **automatically** keeping numerically
sensitive ops (softmax, RMSNorm/LayerNorm reductions, `cross_entropy`) in fp32. Weights and the
optimizer stay fp32 ("master weights"); grads accumulate in fp32. That per-op dtype mix is the
"mixed" in mixed precision.

Lightning `precision="bf16"` (= bf16-*mixed*) does the same thing under the hood: it wraps the
`*_step` (forward + loss) in `torch.autocast(dtype=bfloat16)`, keeps fp32 master weights, and only
for **fp16** adds a `GradScaler`. So the manual loop and the Trainer are the same recipe.

## 2. Scope of autocast — what runs in which dtype (same at every row as the Trainer)
| Stage | In autocast? | dtype |
|---|---|---|
| Weights / init | — (persistent) | **fp32** master weights |
| forward: linear / matmul / attention | inside | **bf16** |
| forward: softmax / RMSNorm / `cross_entropy` | inside | **fp32** (autocast promotes) |
| `loss.backward()` | outside | bf16 ops' grads in bf16, **accumulated into fp32 `.grad`** |
| `optimizer.step()` | outside | **fp32** (fp32 grads update fp32 weights) |
| `scheduler.step()` | outside | n/a — adjusts LR (Python floats) |

## 3. Why backward / optimizer / scheduler are OUTSIDE autocast (and that's correct)
Autocast only needs to be active during the **forward** pass: autograd records the dtype each op
used, so `loss.backward()` (textually outside the `with`) replays each op's backward in the dtype
its forward used — bf16 ops' backward still runs in bf16. The PyTorch AMP recipe explicitly says to
wrap *only* the forward + loss and keep backward outside. Grads landing on fp32 params accumulate
into fp32 `.grad`; `optimizer.step()` updates fp32 master weights; `scheduler.step()` is not a tensor
op. The Trainer leaves these outside autocast too — it does nothing extra. fp32 weight updates are
*desired* (stability), not a downgrade; you still keep the bf16 speedup in the forward/backward math.

## 4. Attention: xformers that dispatches to FlashAttention
One attention path, shared by pretraining and fine-tuning:
`FineTuneModel.forward` → `FEMRModel.backbone` (`FEMRTransformer`) → `FEMREncoderLayer.forward`
→ `memory_efficient_attention_wrapper(...)` (`transformer.py:125`)
→ on GPU: `xformers.ops.memory_efficient_attention(q, k, v, attn_bias)` (`xformers.py:57`).
There is **no** `flash_attn` package and **no** `F.scaled_dot_product_attention` anywhere in `femr`.

`xformers.ops.memory_efficient_attention` is a **dispatcher** that picks the best CUDA kernel for the
inputs; **FlashAttention (`flshattF`) is one of its backends** (alongside Cutlass `cutlassF`). The
choice hinges on dtype:
- **bf16/fp16 (with `--amp`)** → eligible for the **FlashAttention** kernel on A100 (fused, fast).
- **fp32 (no `--amp`)** → Flash unsupported → fp32 Cutlass/mem-efficient kernel — correct but slow.
q/k/v come from `input_proj` (`nn.Linear`), so under `--amp` they're bf16 → flash kernel.

## 5. Why fp32 (no `--amp`) is slow — the user's hypothesis, confirmed
Two compounding reasons:
1. **No FlashAttention** — the fp32 attention kernel is much slower than the fused bf16 flash kernel.
2. **No tensor-core matmuls** — PyTorch 2.x defaults TF32 **off**, so fp32 `nn.Linear`/matmuls run at
   full fp32 (no tensor cores).
Both vanish under `--amp` (bf16). Also: **pretraining used `bf16=True`** (`pretrain.py:352`), so the
attention was pretrained with the bf16/flash kernel — running fine-tuning **with `--amp`** matches
that regime (faster *and* numerically closer to pretraining). (There's even a commented-out
`with torch.autocast("cuda", dtype=torch.bfloat16):` around the attention at `transformer.py:119`.)

## 6. If you ever want fp16 instead of bf16
fp16 has a narrow exponent range and needs a `torch.cuda.amp.GradScaler` to avoid gradient
underflow. bf16 shares fp32's exponent range, so it needs **no** scaler — which is why the current
bf16 path is just autocast + plain `backward()`.

## Code references
- `finetuning.py` — `train_epoch` autocast (forward+loss), `evaluate` autocast + fp32 softmax.
- `transformer.py:103-144` — `FEMREncoderLayer.forward`; `:125` `memory_efficient_attention_wrapper`.
- `xformers.py:53-57` — wrapper → `xformers.ops.memory_efficient_attention` (GPU) / `ref_attention` (CPU).
- `pretrain.py:352` — `bf16=True` (pretraining precision).
