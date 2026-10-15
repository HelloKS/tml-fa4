# SPDX-License-Identifier: BSD-3-Clause
"""Native E4M3 attention for SM120/SM121 using warp-level FP8 MMA.

QK and PV both use mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32.
Softmax and accumulators stay FP32; only the final output converts to BF16.
V is transposed within each shared-memory tile so PV needs no 8-bit ldmatrix
transpose. P passes through shared memory to remap QK's accumulator layout
to PV's FP8 operand layout across lanes.
"""

import math
import re
from collections.abc import Callable
from types import SimpleNamespace

import cutlass
from cutlass import Float32, Int32, const_expr, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack import layout_utils

from flash_attn.cute import ampere_helpers, utils
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax


def native_fp8_available() -> bool:
    # Earlier DSLs exposed an FP8 op backed by the SM80 IR type and could
    # segfault while making its fragments. Use the SM89 lowering in 4.8+.
    version = re.match(r"(\d+)\.(\d+)", cutlass.__version__)
    return (
        version is not None
        and tuple(map(int, version.groups())) >= (4, 8)
        and hasattr(warp, "MmaFP8Op")
    )


class SoftmaxFp8(Softmax):
    @staticmethod
    def create(scale_log2, num_rows, softmax_scale=None):
        return SoftmaxFp8(
            scale_log2,
            num_rows,
            cute.make_rmem_tensor(num_rows, Float32),
            cute.make_rmem_tensor(num_rows, Float32),
            80,
            softmax_scale,
        )

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0, sink_val=None):
        # Each PV multiply uses P * 256 to retain small E4M3 probabilities.
        # Row sums/LSE use the original FP32 P, so undo that scale only on O.
        return super().finalize(final_scale * (1.0 / 256.0), sink_val)


def _layout(shape, stride):
    # A no-op swizzle retains the composed-layout ABI of the shared mainloop.
    return cute.make_composed_layout(
        cute.make_swizzle(0, 4, 4), 0, cute.make_layout(shape, stride=stride)
    )


class FlashAttentionForwardSm120Fp8(FlashAttentionForwardSm120):
    def __init__(self, dtype, head_dim, head_dim_v=None, *args, **kwargs):
        assert native_fp8_available(), "Native SM12x FP8 requires nvidia-cutlass-dsl>=4.8"
        assert dtype == cutlass.Float8E4M3FN
        super().__init__(dtype, head_dim, head_dim_v, *args, **kwargs)
        assert self.num_stages == 1 and not self.Q_in_regs
        self.dtype = dtype
        self.tile_hdim = math.ceil(self.head_dim / 32) * 32
        self.tile_hdimv = math.ceil(self.head_dim_v / 32) * 32
        self.check_hdim_oob = self.head_dim != self.tile_hdim
        self.check_hdim_v_oob = self.head_dim_v != self.tile_hdimv

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ):
        if dtype != cutlass.Float8E4M3FN or num_stages != 1 or Q_in_regs:
            return False
        if not (16 <= head_dim <= 128 and 16 <= head_dim_v <= 128):
            return False
        if head_dim % 16 or head_dim_v % 16:
            return False
        if tile_m <= 0 or tile_m % 64 or tile_n <= 0 or tile_n % 32 or num_threads != 128:
            return False
        d, dv = math.ceil(head_dim / 32) * 32, math.ceil(head_dim_v / 32) * 32
        sizes = (tile_m * d, tile_n * d, tile_n * dv, tile_m * tile_n)
        return sum((x + 1023) // 1024 * 1024 for x in sizes) <= 99 * 1024

    def _check_type(self, q_type, k_type, v_type, o_type, lse_type, *index_types):
        assert q_type == k_type == v_type == cutlass.Float8E4M3FN
        assert o_type == cutlass.BFloat16
        assert lse_type in (None, Float32)
        assert all(t in (None, Int32) for t in index_types)

    def _get_tiled_mma(self):
        def make_mma():
            return cute.make_tiled_mma(
                warp.MmaFP8Op(self.dtype, Float32, (16, 8, 32)),
                (4, 1, 1),
                permutation_mnk=(64, 16, 32),
            )

        return make_mma(), make_mma()

    def _setup_attributes(self):
        d, dv, m, n = self.tile_hdim, self.tile_hdimv, self.tile_m, self.tile_n
        self.sQ_layout = _layout((m, d), (d, 1))
        self.sK_layout = _layout((n, d, 1), (d, 1, n * d))
        # Logical (sequence, dim), physically (dim, sequence) for FP8 PV MMA.
        self.sV_layout = _layout((n, dv, 1), (1, n, n * dv))
        self.sP_layout = _layout((m, n), (n, 1))
        self.sO_layout = _layout((m, dv), (dv, 1))  # epilogue stores directly

        def copy_for(dtype, dim, is_async):
            elems = 128 // dtype.width
            # Padded dimensions such as 96 have six FP8 vectors per row.
            # A power-of-two thread group must divide both that row and 128.
            per_row = math.gcd(8, dim // elems)
            atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
                if is_async
                else cute.nvgpu.CopyUniversalOp(),
                dtype,
                num_bits_per_copy=128,
            )
            return cute.make_tiled_copy_tv(
                atom,
                cute.make_ordered_layout((128 // per_row, per_row), order=(1, 0)),
                cute.make_layout((1, elems)),
            )

        self.gmem_tiled_copy_Q = copy_for(self.dtype, d, True)
        self.gmem_tiled_copy_K = copy_for(self.dtype, d, True)
        self.gmem_tiled_copy_V = copy_for(self.dtype, dv, True)
        self.gmem_tiled_copy_O = copy_for(cutlass.BFloat16, dv, False)

    def _get_shared_storage_cls(self):
        def buffer(layout):
            return cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]

        q, k, v, p = [
            buffer(x) for x in (self.sQ_layout, self.sK_layout, self.sV_layout, self.sP_layout)
        ]

        @cute.struct
        class SharedStorage:
            sQ: q
            sK: k
            sV: v
            sP: p

        return SharedStorage

    def _make_softmax(self, scale_log2, num_rows, softmax_scale):
        return SoftmaxFp8.create(scale_log2, num_rows, softmax_scale)

    def _make_smem_copy_atom_v(self):
        return cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )

    @cute.jit
    def load_Q(self, gmem_thr_copy, gQ, sQ, block, seqlen, headdim):
        FlashAttentionForwardSm80.load_Q(self, gmem_thr_copy, gQ, sQ, block, seqlen, headdim)

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy,
        tKgK,
        tKsK,
        tKcK,
        t0KcK,
        tKpK,
        block,
        smem_pipe_write,
        seqlen,
        need_predicates,
    ):
        FlashAttentionForwardSm80.load_K(
            self,
            gmem_tiled_copy,
            tKgK,
            tKsK,
            tKcK,
            t0KcK,
            tKpK,
            block,
            smem_pipe_write,
            seqlen,
            need_predicates,
        )

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn=None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        self.mma_one_n_block_bs(
            n_block,
            mma_params,
            smem_copy_params,
            softmax,
            load_K,
            load_V,
            score_mod,
            batch_idx,
            head_idx,
            m_block,
            seqlen,
            aux_tensors,
            fastdiv_mods,
            mask_fn,
            is_first_n_block,
        )

    @cute.jit
    def mma_one_n_block_bs(
        self,
        n_block: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn=None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        # Drain the common prologue before reusing the single-stage buffers.
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        load_K(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()
        load_V(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        acc_S = cute.make_rmem_tensor(
            mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_S.fill(0.0)
        ampere_helpers.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=True)
        softmax.rescale_O(mma_params.acc_O, row_scale)

        # The FP8 A fragment packs columns differently from a QK accumulator.
        # Store by logical (m,n), then let ldmatrix redistribute P across lanes.
        thr_qk = mma_params.thr_mma_qk
        p_dst = thr_qk.partition_C(mma_params.sP)
        for i in cutlass.range_constexpr(cute.size(acc_S)):
            p_dst[i] = (acc_S[i] * 256.0).to(self.dtype)
        cute.arch.barrier()
        mma_pv = mma_params.thr_mma_pv
        copy_p = utils.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
            ),
            mma_pv,
        ).get_slice(mma_params.tidx)
        p_frag = mma_pv.make_fragment_A(mma_pv.partition_A(mma_params.sP))
        ampere_helpers.gemm(
            mma_pv,
            mma_params.acc_O,
            p_frag,
            mma_params.tOrVt,
            copy_p.partition_S(mma_params.sP),
            smem_copy_params.tOsVt[None, None, None, 0],
            copy_p,
            smem_copy_params.smem_thr_copy_V,
        )
        cute.arch.barrier()

    @cute.jit
    def epilogue(
        self,
        acc_O,
        lse,
        mO,
        mLSE,
        sO,
        seqlen,
        gmem_tiled_copy_O,
        tma_atom_O,
        tiled_mma,
        tidx,
        m_block,
        head_idx,
        batch_idx,
    ):
        # Direct BF16 stores avoid interpreting an FP8 Q buffer as BF16 output.
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        thr = tiled_mma.get_slice(tidx)
        dst = layout_utils.reshape_acc_to_mn(thr.partition_C(gO))
        coord = layout_utils.reshape_acc_to_mn(
            thr.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_hdimv)))
        )
        src = layout_utils.reshape_acc_to_mn(acc_O)
        if const_expr(mLSE is not None):
            lse_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
        for r in cutlass.range_constexpr(cute.size(src.shape[0])):
            row = coord[r, 0][0]
            if row < seqlen.seqlen_q - m_block * self.tile_m:
                for n in cutlass.range_constexpr(cute.size(src.shape[1])):
                    if coord[r, n][1] < mO.shape[1]:
                        dst[r, n] = src[r, n].to(cutlass.BFloat16)
                # Keep the compile-time None guard separate from the lane predicate.
                if const_expr(mLSE is not None):  # noqa: SIM102
                    if coord[r, 0][1] == 0:
                        lse_cur[m_block * self.tile_m + row] = lse[r]
