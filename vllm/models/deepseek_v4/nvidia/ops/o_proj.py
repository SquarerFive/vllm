# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, is_deep_gemm_supported
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _fused_inverse_rope_gptj,
    _get_cached_wo_a_bf16,
)


def _linear_forward(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(linear, "weight") and x.dtype != linear.weight.dtype:
        x = x.to(linear.weight.dtype)
    output = linear(x)
    if isinstance(output, tuple):
        output = output[0]
    return output


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    if cos_sin_cache.dtype != torch.float32:
        cos_sin_cache = cos_sin_cache.float()

    if not is_deep_gemm_supported():
        o_ref = _fused_inverse_rope_gptj(o, positions, cos_sin_cache, rope_dim)
        o_ref = o_ref.view(o.shape[0], n_groups, -1)
        if hasattr(wo_a, "weight"):
            wo_a_weight = _get_cached_wo_a_bf16(
                wo_a, n_groups, o_lora_rank, o_ref.shape[-1]
            )
            z = torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)
        else:
            z = _linear_forward(wo_a, o_ref.flatten(1))
            z = z.view(o.shape[0], n_groups, o_lora_rank)
        return _linear_forward(wo_b, z.flatten(1))

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, wo_a.weight_scale_inv),
        z,
        recipe=einsum_recipe,
    )
    return _linear_forward(wo_b, z.flatten(1))
