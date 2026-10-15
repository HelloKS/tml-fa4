"""Native SM12x FP8 lowering, dispatch, and legacy fallback checks."""

import math
import sys

import pytest

torch = pytest.importorskip("torch")
cutlass = pytest.importorskip("cutlass")
if sys.platform == "win32" and "fcntl" not in sys.modules:
    pytest.skip("quack's package initializer requires a POSIX runtime", allow_module_level=True)

import cutlass.cute as cute  # noqa: E402

from flash_attn.cute import flash_attn_func  # noqa: E402
from flash_attn.cute import interface  # noqa: E402
from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: E402
from flash_attn.cute.flash_fwd_sm120_fp8 import (  # noqa: E402
    FlashAttentionForwardSm120Fp8,
    native_fp8_available,
)

sm12x = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires an SM120/SM121 GPU",
)
native_sm12x = pytest.mark.skipif(
    not native_fp8_available()
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 12,
    reason="requires an SM120/SM121 GPU and CuTeDSL 4.8+",
)


@pytest.mark.parametrize(
    "version,expected",
    [("4.5.2", False), ("4.6.0", False), ("4.8.0", True), ("4.8.1.dev0", True), ("5.0.0", True)],
)
def test_native_fp8_version_guard(monkeypatch, version, expected):
    monkeypatch.setattr(cutlass, "__version__", version)
    monkeypatch.setattr(cute.nvgpu.warp, "MmaFP8Op", object(), raising=False)
    assert native_fp8_available() is expected


def test_native_fp8_requires_warp_op(monkeypatch):
    monkeypatch.setattr(cutlass, "__version__", "4.8.0")
    monkeypatch.delattr(cute.nvgpu.warp, "MmaFP8Op", raising=False)
    assert not native_fp8_available()


def test_native_fp8_smem_budget():
    check = FlashAttentionForwardSm120Fp8.can_implement
    args = (cutlass.Float8E4M3FN, 128, 128, 128, 128, 1, 128, False)
    assert check(*args)
    assert not check(cutlass.Float8E4M3FN, 128, 128, 512, 256, 1, 128, False)
    assert not check(cutlass.Float8E4M3FN, 128, 128, 128, 48, 1, 128, False)


def tensors():
    torch.manual_seed(11)
    q = torch.randn(1, 17, 4, 48, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(1, 67, 2, 48, device="cuda", dtype=torch.bfloat16).to(q.dtype)
    v = torch.randn(1, 67, 2, 80, device="cuda", dtype=torch.bfloat16).to(q.dtype)
    return q, k, v


def reference(q, k, v):
    k, v = [x.float().repeat_interleave(2, dim=2).transpose(1, 2) for x in (k, v)]
    scores = q.float().transpose(1, 2) @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return (scores.softmax(-1) @ v).transpose(1, 2), scores.logsumexp(-1)


@native_sm12x
def test_qk_and_pv_lower_to_native_fp8_mma():
    q, k, v = tensors()
    out = torch.empty(1, 17, 4, 80, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(1, 4, 17, device="cuda", dtype=torch.float32)
    kernel = FlashAttentionForwardSm120Fp8(
        cutlass.Float8E4M3FN,
        48,
        80,
        2,
        tile_m=128,
        tile_n=128,
        num_threads=128,
        num_stages=1,
    )
    scale = 1.0 / math.sqrt(48)
    compiled = cute.compile(
        kernel,
        *[to_cute_tensor(x, assumed_align=4 if x is lse else 16) for x in (q, k, v, out, lse)],
        cutlass.Float32(scale),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --keep-ptx",
    )
    compiled(q, k, v, out, lse, scale)
    torch.cuda.synchronize()
    ref, ref_lse = reference(q, k, v)
    torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.03)
    torch.testing.assert_close(lse, ref_lse, atol=0.01, rtol=0.01)
    ptx = compiled.artifacts.PTX
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    mma = [line for line in str(ptx).splitlines() if "mma.sync" in line]
    # Both GEMMs have been compiled in the same kernel, with no BF16 MMA.
    assert len(mma) >= 2
    assert all(".f32.e4m3.e4m3.f32" in line for line in mma)


@sm12x
def test_legacy_bf16_compute_fallback(monkeypatch):
    monkeypatch.setattr(interface, "native_fp8_available", lambda: False)
    q, k, v = tensors()
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    ref, ref_lse = reference(q, k, v)
    torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse, ref_lse, atol=0.01, rtol=0.01)


@native_sm12x
def test_native_and_fallback_have_distinct_cache_entries(monkeypatch):
    q, k, v = tensors()
    native_out, native_lse = flash_attn_func(q, k, v, return_lse=True)
    monkeypatch.setattr(interface, "native_fp8_available", lambda: False)
    fallback_out, fallback_lse = flash_attn_func(q, k, v, return_lse=True)
    ref, ref_lse = reference(q, k, v)
    torch.testing.assert_close(native_out.float(), ref, atol=0.03, rtol=0.03)
    torch.testing.assert_close(fallback_out.float(), ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(native_lse, ref_lse, atol=0.01, rtol=0.01)
    torch.testing.assert_close(fallback_lse, ref_lse, atol=0.01, rtol=0.01)


@native_sm12x
@pytest.mark.parametrize("d,dv", [(16, 16), (80, 96), (96, 112), (112, 80), (128, 16)])
def test_native_padding_with_non_power_of_two_head_dimensions(d, dv):
    torch.manual_seed(19)
    q = torch.randn(1, 19, 4, d, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(1, 71, 2, d, device="cuda", dtype=torch.bfloat16).to(q.dtype)
    v = torch.randn(1, 71, 2, dv, device="cuda", dtype=torch.bfloat16).to(q.dtype)
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    ref, ref_lse = reference(q, k, v)
    torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.03)
    torch.testing.assert_close(lse, ref_lse, atol=0.01, rtol=0.01)
