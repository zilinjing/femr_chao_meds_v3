import os

import torch
import xformers.ops

# On Blackwell (sm_100, e.g. B200) xformers' automatic backend dispatch selects an FMHA kernel
# whose *backward* pass fails with "CUDA error: unspecified launch failure". The forward pass
# succeeds, so the crash only surfaces at the first loss.backward(). Pinning a verified backend
# avoids it; both flash and cutlass were checked on B200 with this model's block-diagonal +
# local-attention bias, and flash is ~1.7x faster. Set FEMR_ATTENTION_BACKEND=cutlass to use the
# more general (slower) kernel, or =default to restore xformers' own dispatch.
_ATTENTION_BACKEND = os.environ.get("FEMR_ATTENTION_BACKEND", "flash")
_ATTENTION_OP = {
    "flash": (xformers.ops.fmha.flash.FwOp, xformers.ops.fmha.flash.BwOp),
    "cutlass": (xformers.ops.fmha.cutlass.FwOp, xformers.ops.fmha.cutlass.BwOp),
    "default": None,
}[_ATTENTION_BACKEND]

# From https://github.com/facebookresearch/xformers/blob/042abc8aa47d1f5bcc2e82df041811de218924ba/tests/test_mem_eff_attention.py#L511 # noqa


def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    q = q.float()
    k = k.float()
    v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape([-1, *attn_bias_tensor.shape[2:]])
        attn = attn + attn_bias_tensor.float()
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def ref_attention_bmhk(q, k, v, attn_bias, scale=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


def memory_efficient_attention_wrapper(q, k, v, attn_bias):
    if q.device.type == "cpu":
        return ref_attention_bmhk(q, k, v, attn_bias)
    else:
        return xformers.ops.memory_efficient_attention(q, k, v, attn_bias, op=_ATTENTION_OP)
