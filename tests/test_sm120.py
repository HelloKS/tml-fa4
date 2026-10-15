"""SM120/SM121 forward regression tests (run numerical tests on either GPU)."""

import math
import sys

import pytest

torch = pytest.importorskip("torch")
cutlass = pytest.importorskip("cutlass")
if sys.platform == "win32" and "fcntl" not in sys.modules:
    pytest.skip("quack's package initializer requires a POSIX runtime", allow_module_level=True)

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # noqa: E402
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch  # noqa: E402
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120  # noqa: E402
from flash_attn.cute.interface import _flash_attn_fwd, _parse_arch_str  # noqa: E402

sm12x = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires an SM120/SM121 GPU",
)


@pytest.mark.parametrize("arch", ["sm_120", "sm_120a", "sm_121", "sm_121a", "121"])
def test_parse_sm12x_arch(arch):
    assert _parse_arch_str(arch) in (120, 121)


def test_smem_budget_uses_padded_compute_tiles():
    check = FlashAttentionForwardSm120.can_implement
    assert check(cutlass.BFloat16, 128, 128, 128, 64, 1, 128, False)
    assert check(cutlass.Float8E4M3FN, 128, 128, 128, 64, 1, 128, False)
    assert not check(cutlass.Float8E4M3FN, 128, 128, 128, 128, 2, 128, False)
    assert not check(cutlass.BFloat16, 256, 256, 128, 64, 1, 128, False)
    assert not check(cutlass.BFloat16, 64, 64, 0, 64, 1, 128, False)


def reference(q, k, v, causal=False, window=(None, None), rel=None, softcap=None):
    q, k, v = [x.float().transpose(1, 2) for x in (q, k, v)]
    ratio = q.shape[1] // k.shape[1]
    k, v = [x.repeat_interleave(ratio, dim=1) for x in (k, v)]
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    sq, sk = q.shape[2], k.shape[2]
    distance = (
        torch.arange(sq, device=q.device)[:, None] + sk - sq - torch.arange(sk, device=q.device)
    )
    if rel is not None:
        extent = rel.shape[-1]
        indices = distance.clamp(0, extent - 1)[None, None].expand(scores.shape)
        contribution = rel.float().transpose(1, 2).gather(-1, indices)
        scores += torch.where((distance >= 0) & (distance < extent), contribution, 0)
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    mask = torch.ones((sq, sk), dtype=torch.bool, device=q.device)
    if causal:
        mask &= distance >= 0
    if window[0] is not None:
        mask &= distance <= window[0]
    if window[1] is not None:
        mask &= distance >= -window[1]
    scores.masked_fill_(~mask, -torch.inf)
    lse = scores.logsumexp(-1)
    probabilities = torch.nan_to_num(scores.softmax(-1))
    return (probabilities @ v).transpose(1, 2), lse


def check_result(actual, expected, input_dtype=torch.bfloat16):
    out, lse = actual
    ref_out, ref_lse = expected
    if input_dtype == torch.float8_e4m3fn:
        # Native FP8 quantizes P's three-bit mantissa as well as the inputs.
        # Keep BF16/FP16 tolerances unchanged and bound aggregate FP8 error too.
        torch.testing.assert_close(out.float(), ref_out, atol=0.05, rtol=0.05)
        relative_l2 = (out.float() - ref_out).norm() / ref_out.norm().clamp_min(1e-8)
        assert relative_l2 < 0.035
    else:
        torch.testing.assert_close(out.float(), ref_out, atol=0.025, rtol=0.025)
    if lse is not None:
        torch.testing.assert_close(lse, ref_lse, atol=0.025, rtol=0.025)


@sm12x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (32, 64)])
@pytest.mark.parametrize("causal,ratio", [(False, 1), (True, 2), (True, 3)])
def test_dense(dtype, dims, causal, ratio):
    torch.manual_seed(0)
    d, dv = dims
    q = torch.randn(2, 31, 2 * ratio, d, device="cuda", dtype=torch.bfloat16).to(dtype)
    k = torch.randn(2, 193, 2, d, device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn(2, 193, 2, dv, device="cuda", dtype=torch.bfloat16).to(dtype)
    actual = flash_attn_func(q, k, v, causal=causal, return_lse=True, pack_gqa=True, num_splits=4)
    assert actual[0].dtype == (torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype)
    check_result(actual, reference(q, k, v, causal), input_dtype=q.dtype)
    # Reuse the compiled entry, with different tensor contents.
    neg_v = (-v.float()).to(dtype)
    check_result(
        flash_attn_func(q, k, neg_v, causal=causal, return_lse=True),
        reference(q, k, neg_v, causal),
        input_dtype=q.dtype,
    )


@sm12x
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("window,softcap", [((None, None), None), ((63, 0), 3.0)])
def test_relative_bias(dtype, varlen, window, softcap):
    torch.manual_seed(1)
    qlens, klens = [17, 39], [129, 67]
    qs, ks, vs, rels, expected = [], [], [], [], []
    for sq, sk in zip(qlens, klens):
        q = torch.randn(1, sq, 6, 128, device="cuda", dtype=torch.bfloat16).to(dtype)
        k = torch.randn(1, sk, 2, 128, device="cuda", dtype=torch.bfloat16).to(dtype)
        v = torch.randn_like(k.float()).to(dtype)
        rel = torch.randn(1, sq, 6, 256, device="cuda", dtype=torch.bfloat16) * 3
        qs.append(q)
        ks.append(k)
        vs.append(v)
        rels.append(rel)
        expected.append(reference(q, k, v, causal=True, window=window, rel=rel, softcap=softcap))
    if varlen:
        q, k, v, rel = [torch.cat(xs, dim=1).squeeze(0) for xs in (qs, ks, vs, rels)]
        cuq = torch.tensor([0, 17, 56], dtype=torch.int32, device="cuda")
        cuk = torch.tensor([0, 129, 196], dtype=torch.int32, device="cuda")
        out = torch.empty_like(q, dtype=torch.bfloat16)
        actual = flash_attn_varlen_func(
            q,
            k,
            v,
            rel_bias=rel,
            cu_seqlens_q=cuq,
            cu_seqlens_k=cuk,
            max_seqlen_q=39,
            max_seqlen_k=129,
            causal=True,
            window_size=window,
            softcap=softcap or 0,
            return_lse=True,
            out=out,
        )
        assert actual[0].data_ptr() == out.data_ptr()
        for b, (start, end) in enumerate([(0, 17), (17, 56)]):
            check_result(
                (actual[0][None, start:end], actual[1][None, :, start:end]),
                expected[b],
                input_dtype=q.dtype,
            )
    else:
        for q, k, v, rel, ref in zip(qs, ks, vs, rels, expected):
            check_result(
                flash_attn_func(
                    q,
                    k,
                    v,
                    rel_bias=rel,
                    causal=True,
                    window_size=window,
                    softcap=softcap or 0,
                    return_lse=True,
                ),
                ref,
                input_dtype=q.dtype,
            )


@sm12x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("page_size", [16, 64])
def test_paged_kv(dtype, page_size):
    torch.manual_seed(2)
    q = torch.randn(2, 7, 6, 64, device="cuda", dtype=torch.bfloat16).to(dtype)
    k = torch.randn(20, page_size, 2, 64, device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn(20, page_size, 2, 32, device="cuda", dtype=torch.bfloat16).to(dtype)
    table = torch.tensor(
        [[7, 2, 9, 5, 1, -1], [3, 11, 6, -1, -1, -1]], device="cuda", dtype=torch.int32
    )
    lengths = [4 * page_size + 3, 2 * page_size + 1]
    used = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(
        q,
        k,
        v,
        page_table=table,
        seqused_k=used,
        max_seqlen_k=max(lengths),
        causal=True,
        return_lse=True,
        num_splits=0,
    )
    for b, length in enumerate(lengths):
        pages = (length + page_size - 1) // page_size
        # Index FP8 via its byte representation, for PyTorch releases without
        # index_select/indexing kernels for float8.
        ids = table[b, :pages].long()
        kd, vd = [x.float()[ids].flatten(0, 1)[:length][None].to(dtype) for x in (k, v)]
        check_result(
            (actual[0][b : b + 1], actual[1][b : b + 1]),
            reference(q[b : b + 1], kd, vd, True),
            input_dtype=q.dtype,
        )


@sm12x
def test_empty_causal_rows_and_zero_kv():
    q = torch.randn(1, 193, 4, 64, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(1, 17, 2, 64, device="cuda", dtype=torch.bfloat16).to(q.dtype)
    v = torch.randn_like(k.float()).to(q.dtype)
    check_result(
        flash_attn_func(q, k, v, causal=True, return_lse=True),
        reference(q, k, v, True),
        input_dtype=q.dtype,
    )
    out, lse = flash_attn_func(q, k[:, :0], v[:, :0], return_lse=True)
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


@sm12x
def test_reject_unsupported_sm12x_features():
    q = torch.zeros(1, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="between 8 and 128"):
        flash_attn_func(q, q, q)
    q = q[..., :64].contiguous()
    with pytest.raises(AssertionError, match="logits_max"):
        flash_attn_func(q, q, q, return_logits_max=True)
    with pytest.raises(AssertionError, match="shared memory"):
        _flash_attn_fwd(q, q, q, tile_mn=(512, 256))


@sm12x
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("varlen", [False, True])
def test_block_sparse_including_empty_tile(dtype, varlen):
    torch.manual_seed(3)
    device = "cuda"

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    if varlen:
        q = torch.randn(56, 4, 64, device=device, dtype=torch.bfloat16).to(dtype)
        k = torch.randn(196, 2, 64, device=device, dtype=torch.bfloat16).to(dtype)
        v = torch.randn_like(k.float()).to(dtype)
        sparse = BlockSparseTensorsTorch(
            tensor([[0, 0]]),
            tensor([[0, 0, 0]]),
            tensor([[2, 0]]),
            tensor([[0, 1, 0]]),
            cu_total_m_blocks=tensor([0, 1, 2]),
            cu_block_idx_offsets=tensor([0, 2, 3]),
            block_size=(128, 128),
        )
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=tensor([0, 17, 56]),
            cu_seqlens_k=tensor([0, 129, 196]),
            max_seqlen_q=39,
            max_seqlen_k=129,
            block_sparse_tensors=sparse,
            return_lse=True,
        )
        check_result(
            (out[None, :17], lse[None, :, :17]),
            reference(q[None, :17], k[None, :129], v[None, :129]),
            input_dtype=q.dtype,
        )
        assert torch.count_nonzero(out[17:]) == 0
        assert torch.isneginf(lse[:, 17:]).all()
    else:
        q = torch.randn(1, 192, 4, 64, device=device, dtype=torch.bfloat16).to(dtype)
        k = torch.randn(1, 257, 2, 64, device=device, dtype=torch.bfloat16).to(dtype)
        v = torch.randn_like(k.float()).to(dtype)
        sparse = BlockSparseTensorsTorch(
            tensor([[[0, 0]]]),
            tensor([[[[0, 0, 0], [0, 0, 0]]]]),
            tensor([[[2, 0]]]),
            tensor([[[[0, 2, 0], [0, 0, 0]]]]),
            block_size=(128, 128),
        )
        out, lse = flash_attn_func(q, k, v, block_sparse_tensors=sparse, return_lse=True)
        indices = torch.cat([torch.arange(128), torch.tensor([256])]).to(device)
        kd, vd = [x.float()[:, indices].to(dtype) for x in (k, v)]
        check_result(
            (out[:, :128], lse[:, :, :128]), reference(q[:, :128], kd, vd), input_dtype=q.dtype
        )
        assert torch.count_nonzero(out[:, 128:]) == 0
        assert torch.isneginf(lse[:, :, 128:]).all()


@sm12x
def test_sm121_dispatch_on_sm12x():
    q = torch.randn(1, 7, 2, 64, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    out, lse, *_ = _flash_attn_fwd(q, q, q, _arch=121, return_lse=True)
    check_result((out, lse), reference(q, q, q), input_dtype=q.dtype)


@sm12x
@pytest.mark.parametrize(
    "dtype,d,dv",
    [(torch.bfloat16, 8, 128), (torch.bfloat16, 120, 72), (torch.float8_e4m3fn, 16, 128)],
)
def test_padded_head_dimensions_and_wide_output(dtype, d, dv):
    q = torch.randn(1, 7, 2, d, device="cuda", dtype=torch.bfloat16).to(dtype)
    k = torch.randn(1, 39, 1, d, device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn(1, 39, 1, dv, device="cuda", dtype=torch.bfloat16).to(dtype)
    check_result(flash_attn_func(q, k, v, return_lse=True), reference(q, k, v), input_dtype=q.dtype)
