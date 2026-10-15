# FlashAttention-4 Forward

Forward-only FlashAttention-4 CuTeDSL kernels from `thinking-machines-lab/colfax_collab`.

This repo keeps the FA4 forward attention paths, including relative bias shearing, split-KV combine, scheduler metadata, mask/score modifiers, block-sparse support, and the SM100 hd256 forward kernel.

## Install

```sh
pip install -e ".[dev,cu13]"
```

## Use

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out, lse = flash_attn_func(q, k, v, causal=True)
```

The public attention functions return forward outputs only. Inputs may require gradients, but returned tensors are not wired to FA4 backward kernels.

## SM120 / SM121 (consumer Blackwell and DGX Spark)

SM120 and SM121 share the SM12x dispatch path. Dense FP16/BF16 attention uses
single-CTA TMA loads with warp specialization when its shared memory budget and
layout permit. Other shapes, variable-length sequences, paged KV, and block-sparse
attention use the SM80-style MMA mainloop with SM12x's 99 KiB shared memory limit.
GQA/MQA use the unpacked layout. Head dimensions for Q/K and V may differ, and
must each be in `[8, 128]` and divisible by 8 (16 for FP8).

Unscaled `torch.float8_e4m3fn` Q/K/V inputs are also supported on SM12x, with BF16
output and FP32 accumulation/LSE. With `nvidia-cutlass-dsl>=4.8`, both QK and PV
use native `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Q/K/V remain
FP8 in shared memory. Softmax runs in FP32; P is scaled by 256 and converted to
FP8 before PV, and output normalization removes that scale. V is transposed
within its shared memory tile, and P is redistributed through shared memory to
match the FP8 operand layout across lanes. No full BF16 Q/K/V buffers are created.
The native path additionally quantizes intermediate P to E4M3; numerical
comparisons to BF16 computation should use FP8-appropriate tolerances.

Older DSL versions retain the tile-local FP8-to-BF16 computation path; native FP8
and BF16 fallback entries have separate compilation cache keys. Both paths support
dense, varlen, paged KV, block sparsity, and `rel_bias`. SM100/SM103 retain their
existing native FP8 MMA and sheared-bias path. To enable native SM12x FP8:

```sh
pip install -e ".[dev,cu13,sm120-fp8]"
```

```python
q8, k8, v8 = [x.to(torch.float8_e4m3fn) for x in (q, k, v)]
out, lse = flash_attn_func(q8, k8, v8, causal=True, return_lse=True)
# out: bfloat16; lse: float32
```

For paged KV, use `flash_attn_varlen_func(..., page_table=..., seqused_k=...)`;
`seqused_k` contains the actual KV lengths for each sequence. `rel_bias` may be
FP16, BF16, FP32, or the input dtype; SM12x reads it directly with sequence-local
indices, including ragged sequences and sliding windows.

As in the reference bundle, SM12x runs with a single KV split (`num_splits` is
clamped to 1). Block-scaled FP8 (`sfq`/`sfk`/`sfv`), `return_logits_max`, and
SM100 scheduler metadata are unsupported on this path.

The SM12x TMA, paged-KV, and block-sparse work is adapted from
[SecondNatureComputing/flash-attn-4-sm120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120/tree/60117041e10fcc6f19882afd274318c755a5ef6e)
(BSD-3-Clause), with forward-only APIs and this tree's CuTe DSL updates retained.
Run `pytest tests/test_sm120.py tests/test_sm120_native_fp8.py` on an SM120/SM121
GPU for numerical coverage, native PTX checks, and fallback/cache regression tests.

## Ragged Relative Bias

Thread one extra tensor through the attention backend SPI and pass it as `rel_bias`:

```python
# r: [total_q, n_q_heads, d_rel], flattened in the same order as q.
rel_bias = torch.einsum("thd,de->the", r, rel_proj).contiguous()

from flash_attn.cute import flash_attn_varlen_func

out, lse = flash_attn_varlen_func(
    q,  # [total_q, n_q_heads, head_dim]
    k,  # [total_k, n_kv_heads, head_dim]
    v,  # [total_k, n_kv_heads, head_dim_v]
    rel_bias=rel_bias,
    cu_seqlens_q=cu_seqlens_q,  # int32 CUDA tensor, [batch + 1], last == total_q
    cu_seqlens_k=cu_seqlens_k,  # int32 CUDA tensor, [batch + 1], last == total_k
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,
    return_lse=True,
    out=preallocated_out,  # optional, same shape/dtype/device as the output
)
```

Argument contract:

- `q`: `[total_q, n_q_heads, head_dim]`
- `k`: `[total_k, n_kv_heads, head_dim]`
- `v`: `[total_k, n_kv_heads, head_dim_v]`
- `cu_seqlens_q`, `cu_seqlens_k`: contiguous int32 CUDA tensors of shape `[batch + 1]`
- `max_seqlen_q`, `max_seqlen_k`: max per-sequence lengths from the corresponding `cu_seqlens`
- `rel_bias`: `[total_q, n_q_heads, rel_extent]`; no batch dimension; same flattened order, CUDA device, and dtype as `q`; `rel_extent` is inferred from this shape and must be a multiple of 128

```python
q_len_b = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
k_len_b = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
d = i + (k_len_b - q_len_b) - j
score[i, j, h] += rel_bias[cu_seqlens_q[b] + i, h, d]  # when 0 <= d < rel_extent
```

In causal self-attention, `d == 0` is the query token's own key position, `d == 1` is the previous key, and keys farther back than `rel_extent - 1` receive no relative-bias contribution.
