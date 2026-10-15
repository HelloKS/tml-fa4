# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass with TMA loads and warp specialization.
#
# Key differences from FlashAttentionForwardSm120 (CpAsync):
#   - TMA (cp.async.bulk) for Q/K/V global → shared memory transfers
#   - Warp specialization: 1 DMA warp (TMA loads) + N MMA warps (compute)
#   - PipelineTmaAsync with mbarrier synchronization for KV double-buffering
#   - SM80-compatible tensor cores (mma.sync.aligned.m16n8k16) for MMA
#   - Swizzle(B, 4, 3) for SMEM layouts (TMA requirement, not M=3 like CpAsync)
#
# Validated on SM121a (DGX Spark).
# Contributed by Second Nature Computing (https://joinsecondnature.com)
# Adapted from SecondNatureComputing/flash-attn-4-sm120, revision
# 60117041e10fcc6f19882afd274318c755a5ef6e (BSD-3-Clause).


import math
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr, cute, pipeline
from cutlass.cute.nvgpu import cpasync, warp
from quack import layout_utils

from . import ampere_helpers as sm80_utils
from . import utils
from .block_info import BlockInfo
from .cute_dsl_utils import assume_tensor_aligned
from .flash_fwd import FlashAttentionForwardBase
from .mask import AttentionMask
from .seqlen_info import SeqlenInfoQK
from .softmax import Softmax, apply_score_mod_inner
from .tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments


def get_smem_layout_atom_tma(dtype: type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    """TMA-compatible SMEM layout atom using Swizzle(B, 4, 3).

    TMA hardware requires swizzle_base=4 (i.e., Swizzle(B, 4, 3)), unlike CpAsync
    which uses swizzle_base=3 (Swizzle(B, 3, 3)). The swizzle_bits B is chosen
    based on the row width in bytes:
      - 128-byte rows (64 bf16 elems): SW128 → B=3
      - 64-byte rows  (32 bf16 elems): SW64  → B=2
    """
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        cutlass.const_expr(
            128
            if bytes_per_row % 128 == 0
            else 64
            if bytes_per_row % 64 == 0
            else 32
            if bytes_per_row % 32 == 0
            else 16
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else 3
        if smem_k_block_size == 64
        else 2
        if smem_k_block_size == 32
        else 1
    )
    swizzle_base = 4
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, 3),
        0,
        cute.make_ordered_layout(
            (8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size), order=(1, 0)
        ),
    )


class FlashAttentionForwardSm120Tma(FlashAttentionForwardBase):
    """Flash Attention v2 forward for SM120 using TMA loads and warp specialization.

    Uses TMA (cp.async.bulk) for global→shared memory transfers with a dedicated
    DMA warp, while MMA warps perform computation. This enables overlapping loads
    with compute via double-buffered KV pipelining.

    Architecture constraints:
      - SM80-era mma.sync.aligned.m16n8k16 tensor core instructions
      - TMA (cp.async.bulk) for bulk memory transfers (no multicast)
      - PipelineTmaAsync with mbarrier synchronization
      - 99 KB shared memory capacity
      - No WGMMA, no tcgen05, no TMEM
    """

    arch = 80

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 64,
        num_mma_warps: int = 4,
        kv_stages: int = 2,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: bool = False,
    ):
        num_threads = (num_mma_warps + 1) * 32
        super().__init__(
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=kv_stages,
            num_threads=num_threads,
            Q_in_regs=False,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
        )
        from cutlass.base_dsl.arch import Arch

        self.arch = Arch.sm_80
        self.is_split_kv = False
        self.num_mma_warps = num_mma_warps
        self.kv_stages = kv_stages
        self.use_tma_O = False
        self.pack_gqa = False

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, tile_m, tile_n, num_mma_warps, kv_stages, is_causal
    ) -> bool:
        """Check if the TMA kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if num_mma_warps != 4 or kv_stages < 1 or tile_m <= 0 or tile_n <= 0 or tile_n % 16:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v is None:
            head_dim_v = head_dim
        if not (8 <= head_dim_v <= head_dim <= 128):
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_m % (num_mma_warps * 16) != 0:
            return False
        hdim_multiple_of = 16
        tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        elem_bytes = dtype.width // 8
        smem_q = tile_m * tile_hdim * elem_bytes
        smem_k = tile_n * tile_hdim * elem_bytes * kv_stages
        smem_v = tile_n * tile_hdimv * elem_bytes * kv_stages
        smem_mbar = (1 * 2 + kv_stages * 2 * 2) * 8
        smem_mbar_region = (smem_mbar + 1023) // 1024 * 1024
        smem_total = smem_mbar_region + smem_q + smem_k + smem_v
        smem_total += 2 * 1024
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        return smem_total <= smem_capacity

    def _get_smem_layout_atom(self):
        """TMA-compatible SMEM layout atoms with Swizzle(B, 4, 3)."""
        sQ_layout_atom = get_smem_layout_atom_tma(self.dtype, self.tile_hdim)
        sK_layout_atom = get_smem_layout_atom_tma(self.dtype, self.tile_hdim)
        sV_layout_atom = get_smem_layout_atom_tma(self.dtype, self.tile_hdimv)
        sO_layout_atom = get_smem_layout_atom_tma(self.dtype, self.tile_hdimv)
        sP_layout_atom = None
        return (sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom)

    def _get_tiled_mma(self):
        """SM80-compatible MMA for QK and PV GEMMs, using only MMA warps."""
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_mma_warps, 1, 1),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_mma_warps, 1, 1),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        return (tiled_mma_qk, tiled_mma_pv)

    def _get_shared_storage_cls(self):
        """Shared storage with mbarrier arrays for TMA pipelines."""
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)], 1024
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: mbar_ptr_Q_struct
            k_mbar_ptr: mbar_ptr_K_struct
            v_mbar_ptr: mbar_ptr_V_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct

        return SharedStorage

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        seqlen,
        softmax_scale,
        aux_tensors=None,
        fastdiv_mods=None,
    ):
        """Apply score_mod to attention scores."""
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)
        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def _setup_attributes_tma(self):
        """Setup only the attributes needed for TMA kernel (skip CpAsync Q/K/V copies).

        TMA uses hardware-managed bulk copies instead of CpAsync, so we only need:
          - SMEM layouts (sQ_layout, sK_layout, sV_layout, sO_layout)
          - gmem_tiled_copy_O (for epilogue O store via SMEM)
        """
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, _sP_layout_atom = (
            self._get_smem_layout_atom()
        )
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.tile_m, self.tile_hdim, 1), (0, 1, 2)
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.tile_n, self.tile_hdim, self.num_stages), (0, 1, 2)
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.tile_n, self.tile_hdimv, self.num_stages), (0, 1, 2)
        )
        self.sO_layout = cute.tile_to_shape(sO_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1))
        self.sP_layout = None
        universal_copy_bits = 128
        o_copy_elems = universal_copy_bits // self.dtype.width
        atom_universal_copy_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits
        )
        tO_shape_dim_1 = sO_layout_atom.outer.shape[1] // o_copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1), order=(1, 0)
        )
        vO_layout = cute.make_layout((1, o_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy_O, tO_layout, vO_layout
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        softmax_scale: Float32,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        mPageTable: cute.Tensor | None = None,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
        learnable_sink: cute.Tensor | None = None,
        blocksparse_tensors=None,
        aux_tensors=None,
        num_splits_dynamic_ptr: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the TMA SM120 flash attention kernel.

        mQ/mK/mV/mO layout: (batch_size, seqlen, num_head, head_dim)
        """
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
        assert mPageTable is None, "Paged KV not supported with TMA kernel (use CpAsync fallback)"
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            )
        )
        self.o_dtype = mO.element_type
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_mma_threads
        self.num_epilogue_threads = self.num_mma_threads
        self._setup_attributes_tma()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ_t = layout_utils.select(mQ, Q_layout_transpose)
        mK_t = layout_utils.select(mK, KV_layout_transpose)
        mV_t = layout_utils.select(mV, KV_layout_transpose)
        O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        num_splits = Int32(1)
        mO_t = layout_utils.select(mO, O_layout_transpose)
        mLSE_t = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )
        sQ_layout_one_stage = cute.slice_(self.sQ_layout, (None, None, 0))
        sK_layout_one_stage = cute.slice_(self.sK_layout, (None, None, 0))
        sV_layout_one_stage = cute.slice_(self.sV_layout, (None, None, 0))
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_op, mQ_t, sQ_layout_one_stage, (self.tile_m, self.tile_hdim), num_multicast=1
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_op, mK_t, sK_layout_one_stage, (self.tile_n, self.tile_hdim), num_multicast=1
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_op, mV_t, sV_layout_one_stage, (self.tile_n, self.tile_hdimv), num_multicast=1
        )
        q_copy_bytes = cute.size_in_bytes(self.dtype, sQ_layout_one_stage)
        kv_copy_bytes = cute.size_in_bytes(self.dtype, sK_layout_one_stage)
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = (
            mCuSeqlensQ.shape[0] - 1 if const_expr(mCuSeqlensQ is not None) else mQ_t.shape[3]
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mQ_t.shape[0], self.tile_m),
            num_head=cute.size(mQ_t.shape[2]),
            num_batch=num_batch,
            num_splits=num_splits,
            seqlen_k=0,
            headdim=mQ_t.shape[1],
            headdim_v=mV_t.shape[1],
            total_q=cute.size(mQ_t.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ_t.shape[0]) * num_batch,
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale_adj = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ_t, mK_t, self.qhead_per_kvhead, self.pack_gqa, aux_tensors
        )
        self.kernel(
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            mQ_t,
            mK_t,
            mV_t,
            mO_t,
            mLSE_t,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale_log2,
            softmax_scale_adj,
            window_size_left,
            window_size_right,
            q_copy_bytes,
            kv_copy_bytes,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            num_splits,
            aux_tensors,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            cluster=[1, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        mQ_tma: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_tma: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_tma: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        q_copy_bytes: cutlass.Constexpr,
        kv_copy_bytes: cutlass.Constexpr,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        num_splits: Int32,
        aux_tensors=None,
        fastdiv_mods=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _split_idx = work_tile.tile_idx
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        n_block = cutlass.max(n_block_max - 1, 0)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        if const_expr(mCuSeqlensQ is None):
            gQ = cute.local_tile(mQ_tma, (self.tile_m, self.tile_hdim), (None, 0, None, None))
        else:
            gQ = cute.local_tile(mQ_tma, (self.tile_m, self.tile_hdim), (None, 0, None))
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        if const_expr(mCuSeqlensK is None):
            gK = cute.local_tile(mK_tma, (self.tile_n, self.tile_hdim), (None, 0, None, None))
            gV = cute.local_tile(mV_tma, (self.tile_n, self.tile_hdimv), (None, 0, None, None))
        else:
            gK = cute.local_tile(mK_tma, (self.tile_n, self.tile_hdim), (None, 0, None))
            gV = cute.local_tile(mV_tma, (self.tile_n, self.tile_hdimv), (None, 0, None))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        num_head_kv = head_idx // self.qhead_per_kvhead
        if const_expr(mCuSeqlensQ is None):
            tQgQ_block = tQgQ[None, m_block, head_idx, batch_idx]
        else:
            tQgQ_block = tQgQ[None, m_block + seqlen.offset_q // self.tile_m, head_idx]
        if const_expr(mCuSeqlensK is None):
            tKgK_block = tKgK[None, None, num_head_kv, batch_idx]
            tVgV_block = tVgV[None, None, num_head_kv, batch_idx]
        else:
            tKgK_block = tKgK[None, None, num_head_kv]
            tVgV_block = tVgV[None, None, num_head_kv]
        q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=q_copy_bytes,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=kv_copy_bytes,
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=kv_copy_bytes,
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
        )
        pipeline.sync(barrier_id=0)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        sQ_one = sQ[None, None, 0]
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ_one))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ_one)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()
        q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stages
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stages
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stages
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stages
        )
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(232)
            if n_block_max > n_block_min:
                q_pipeline.consumer_wait(
                    pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                )
                mask = AttentionMask(
                    self.tile_m,
                    self.tile_n,
                    seqlen,
                    window_size_left,
                    window_size_right,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                )
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                )
                for n_tile in range(0, n_block_max - n_block_min, 1, unroll=1):
                    cur_n_block = n_block_max - n_tile - 1
                    k_pipeline.consumer_wait(k_consumer_state)
                    k_stage = k_consumer_state.index
                    acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
                    acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
                    acc_S.fill(0.0)
                    sm80_utils.gemm(
                        thr_mma_qk,
                        acc_S,
                        tSrQ,
                        tSrK,
                        tSsQ,
                        tSsK[None, None, None, k_stage],
                        smem_thr_copy_Q,
                        smem_thr_copy_K,
                        A_in_regs=False,
                    )
                    k_pipeline.consumer_release(k_consumer_state)
                    k_consumer_state.advance()
                    if const_expr(self.score_mod is not None):
                        self.apply_score_mod(
                            thr_mma_qk,
                            batch_idx,
                            head_idx,
                            m_block,
                            acc_S,
                            cur_n_block,
                            seqlen,
                            softmax_scale=softmax.softmax_scale,
                            aux_tensors=aux_tensors,
                            fastdiv_mods=fastdiv_mods,
                        )
                    mask_fn(acc_S, n_block=cur_n_block, mask_mod=self.mask_mod, mask_seqlen=True)
                    row_scale = softmax.online_softmax(acc_S, is_first=False, check_inf=True)
                    softmax.rescale_O(acc_O, row_scale)
                    rP = cute.make_fragment_like(acc_S, self.dtype)
                    rP.store(acc_S.load().to(self.dtype))
                    tOrP = layout_utils.reshape_acc_to_frgA(rP)
                    v_pipeline.consumer_wait(v_consumer_state)
                    v_stage = v_consumer_state.index
                    sm80_utils.gemm_rs(
                        thr_mma_pv,
                        acc_O,
                        tOrP,
                        tOrVt,
                        tOsVt[None, None, None, v_stage],
                        smem_thr_copy_V,
                    )
                    v_pipeline.consumer_release(v_consumer_state)
                    v_consumer_state.advance()
            row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale)
            sO = cute.make_tensor(sQ.iterator, sO_layout.outer)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(40)
            if n_block_max > n_block_min:
                q_pipeline.producer_acquire(q_producer_state)
                cute.copy(
                    tma_atom_q,
                    tQgQ_block,
                    tQsQ[None, q_producer_state.index],
                    tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state),
                )
                q_pipeline.producer_commit(q_producer_state)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    cur_n_block = n_block_max - n_tile - 1
                    if const_expr(mCuSeqlensK is not None):
                        kv_tma_idx = cur_n_block + seqlen.offset_k // self.tile_n
                    else:
                        kv_tma_idx = cur_n_block
                    k_pipeline.producer_acquire(k_producer_state)
                    cute.copy(
                        tma_atom_k,
                        tKgK_block[None, kv_tma_idx],
                        tKsK[None, k_producer_state.index],
                        tma_bar_ptr=k_pipeline.producer_get_barrier(k_producer_state),
                    )
                    k_pipeline.producer_commit(k_producer_state)
                    k_producer_state.advance()
                    v_pipeline.producer_acquire(v_producer_state)
                    cute.copy(
                        tma_atom_v,
                        tVgV_block[None, kv_tma_idx],
                        tVsV[None, v_producer_state.index],
                        tma_bar_ptr=v_pipeline.producer_get_barrier(v_producer_state),
                    )
                    v_pipeline.producer_commit(v_producer_state)
                    v_producer_state.advance()
                k_pipeline.producer_tail(k_producer_state)
                v_pipeline.producer_tail(v_producer_state)

    @cute.jit
    def mma_one_n_block(
        self,
        n_block: Int32,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr = False,
        aux_tensors=None,
        fastdiv_mods=None,
    ):
        """Consumer: compute one n_block of S and O with TMA pipeline synchronization."""
        k_pipeline.consumer_wait(k_consumer_state)
        k_stage = k_consumer_state.index
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, k_stage],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            A_in_regs=False,
        )
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        if const_expr(self.score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                seqlen,
                softmax_scale=softmax.softmax_scale,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=True)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        v_pipeline.consumer_wait(v_consumer_state)
        v_stage = v_consumer_state.index
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, v_stage],
            smem_copy_params.smem_thr_copy_V,
        )
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()
