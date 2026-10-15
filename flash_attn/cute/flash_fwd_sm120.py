# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
# Pradeep Ramani, Tri Dao.
# SM120/SM121 paged KV and block-sparse mainloops adapted from
# https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120 (BSD-3-Clause).
# Keep these overrides separate so the SM80/SM90 kernels retain their own epilogue.

import math
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr, cute
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import warp
from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparse_utils import get_total_block_count, run_block_sparse_mainloop_sm80
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


def create_sm120_relative_bias_scoremod(score_mod, aux_index: int, is_varlen: bool):
    @cute.jit
    def add_relative_bias(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        values = cute.make_rmem_tensor(scores.shape, Float32)
        values.store(scores)
        batches = cute.make_rmem_tensor(batch_idx.shape, Int32)
        batches.store(batch_idx)
        heads = cute.make_rmem_tensor(head_idx.shape, Int32)
        heads.store(head_idx)
        queries = cute.make_rmem_tensor(q_idx.shape, Int32)
        queries.store(q_idx)
        keys = cute.make_rmem_tensor(kv_idx.shape, Int32)
        keys.store(kv_idx)
        rel = aux_tensors[aux_index]
        for i in cutlass.range_constexpr(cute.size(values)):
            distance = queries[i] + seqlen_info.seqlen_k - seqlen_info.seqlen_q - keys[i]
            if queries[i] < seqlen_info.seqlen_q and distance >= 0 and distance < rel.shape[-1]:
                if const_expr(is_varlen):
                    values[i] += Float32(rel[seqlen_info.offset_q + queries[i], heads[i], distance])
                else:
                    values[i] += Float32(rel[batches[i], queries[i], heads[i], distance])
        result = values.load()
        if const_expr(score_mod is not None):
            result = score_mod(
                result,
                batch_idx,
                head_idx,
                q_idx=q_idx,
                kv_idx=kv_idx,
                seqlen_info=seqlen_info,
                aux_tensors=aux_tensors[:aux_index],
            )
        return result

    add_relative_bias.__vec_size__ = 1
    return add_relative_bias


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    def __init__(self, dtype, head_dim, head_dim_v=None, *args, **kwargs):
        self.head_dim = head_dim
        self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.input_dtype = dtype
        compute_dtype = cutlass.BFloat16 if dtype == cutlass.Float8E4M3FN else dtype
        super().__init__(compute_dtype, head_dim, head_dim_v, *args, **kwargs)
        # Base __init__ reads the runtime arch: a class attribute does not override it.
        self.arch = Arch.sm_80
        self.pack_gqa = False
        self.is_split_kv = False

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
        if dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN):
            return False
        if not (8 <= head_dim <= 128 and 8 <= head_dim_v <= 128):
            return False
        alignment = 16 if dtype == cutlass.Float8E4M3FN else 8
        if head_dim % alignment or head_dim_v % alignment:
            return False
        if tile_m <= 0 or tile_n <= 0 or tile_n % 16 or num_stages < 1:
            return False
        if num_threads != 128 or tile_m % 64:
            return False
        d = math.ceil(head_dim / 16) * 16
        dv = math.ceil(head_dim_v / 16) * 16
        # All shared tiles, including FP8 input tiles, hold 16-bit compute values.
        q = tile_m * d * 2
        k = tile_n * d * num_stages * 2
        v = tile_n * dv * num_stages * 2
        o = tile_m * dv * 2

        def aligned(size):
            return (size + 1023) // 1024 * 1024

        qv = aligned(max(q, v, o)) if Q_in_regs else aligned(max(q, o)) + aligned(v)
        return qv + aligned(k) <= utils_basic.get_smem_capacity_in_bytes("sm_120")

    def _get_shared_storage_cls(self):
        # The epilogue reuses Q storage for O, including when Dv > D.
        q_elems = max(cute.cosize(self.sQ_layout), cute.cosize(self.sO_layout))
        q_storage = cute.struct.Align[cute.struct.MemRange[self.dtype, q_elems], 1024]
        k_storage = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
        ]
        v_storage = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)], 1024
        ]

        @cute.struct
        class SharedStorage:
            sV: v_storage
            sQ: q_storage
            sK: k_storage

        return SharedStorage

    def _check_type(self, q_type, k_type, v_type, o_type, *metadata_types):
        if not (q_type == k_type == v_type == self.input_dtype):
            raise TypeError("Q/K/V must have the same input dtype")
        super()._check_type(self.dtype, self.dtype, self.dtype, o_type, *metadata_types)

    def _make_softmax(self, scale_log2, num_rows, softmax_scale):
        return Softmax.create(scale_log2, num_rows, softmax_scale=softmax_scale)

    def _make_smem_copy_atom_v(self):
        return cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
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
        blocksparse_tensors: BlockSparseTensors | None = None,
        aux_tensors=None,
        num_splits_dynamic_ptr: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        Q/K/V are FP16, BF16, or E4M3; O uses the 16-bit compute dtype.
        Inputs use (batch, sequence, heads, dim), or (tokens, heads, dim)
        for variable lengths. Each shared memory tile holds compute values.
        """
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            )
        )
        self.o_dtype = mO.element_type
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = self.arch >= Arch.sm_90 and self.arch < Arch.sm_120
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ = layout_utils.select(mQ, Q_layout_transpose)
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        if const_expr(self.is_split_kv):
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = layout_utils.select(mO, O_layout_transpose)
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = mCuSeqlensQ.shape[0] - 1 if const_expr(mCuSeqlensQ is not None) else mQ.shape[3]
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mQ.shape[0], self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=num_splits,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        mPageTable: cute.Tensor | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        num_splits: Int32,
        aux_tensors=None,
        fastdiv_mods=None,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, _split_idx = work_tile.tile_idx
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=blocksparse_tensors.cu_total_m_blocks
            if const_expr(blocksparse_tensors is not None)
            else None,
            mCuBlockIdxOffsets=blocksparse_tensors.cu_block_idx_offsets
            if const_expr(blocksparse_tensors is not None)
            else None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        n_block = cutlass.max(n_block_max - 1, 0)
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        num_head_kv = num_head // self.qhead_per_kvhead
        if const_expr(not seqlen.has_cu_seqlens_q):
            mQ_cur = mQ[None, None, num_head, batch_size]
        else:
            mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, None, num_head])
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        sVt = layout_utils.transpose_view(sV)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = self._make_smem_copy_atom_v()
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        if const_expr(mPageTable is not None):
            load_K = partial(
                self.load_paged,
                mPageTable,
                mK,
                sK,
                tidx,
                batch_size,
                num_head_kv,
                self.tile_hdim,
                seqlen=seqlen.seqlen_k,
            )
            load_V = partial(
                self.load_paged,
                mPageTable,
                mV,
                sV,
                tidx,
                batch_size,
                num_head_kv,
                self.tile_hdimv,
                seqlen=seqlen.seqlen_k,
            )
        else:
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur = mK[None, None, num_head_kv, batch_size]
                mV_cur = mV[None, None, num_head_kv, batch_size]
            else:
                mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, num_head_kv])
                mV_cur = cute.domain_offset((seqlen.offset_k, 0), mV[None, None, num_head_kv])
            gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
            gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))
            gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
            gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
            tKsK, tKgK = (gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK))
            tVsV, tVgV = (gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV))
            cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tKcK = gmem_thr_copy_K.partition_S(cK)
            t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
            if const_expr(self.tile_hdim == self.tile_hdimv):
                tVcV = tKcK
                t0VcV = t0KcK
            else:
                cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                tVcV = gmem_thr_copy_V.partition_S(cV)
                t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
            tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
            if const_expr(self.same_hdim_kv):
                tVpV = tKpK
            else:
                tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])
            load_K = partial(
                self.load_K,
                gmem_tiled_copy_K,
                tKgK,
                tKsK,
                tKcK,
                t0KcK,
                tKpK,
                seqlen=seqlen.seqlen_k,
            )
            load_V = partial(
                self.load_V,
                gmem_tiled_copy_V,
                tVgV,
                tVsV,
                tVcV,
                t0VcV,
                tVpV,
                seqlen=seqlen.seqlen_k,
            )
        softmax = self._make_softmax(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()
        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
            sP=storage.sP.get_tensor(sP_layout) if const_expr(sP_layout is not None) else None,
            tidx=tidx,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
        )
        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            load_K=load_K,
            load_V=load_V,
            score_mod=self.score_mod,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )
        if const_expr(blocksparse_tensors is not None):
            qkv_factor = self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            subtile = self.q_subtile_factor if self.q_subtile_factor is not None else 1
            total_block_cnt = get_total_block_count(
                blocksparse_tensors, batch_size, num_head, m_block, qkv_factor, subtile, seqlen
            )
            bs_mask = AttentionMask(
                self.tile_m, self.tile_n, seqlen, window_size_left, window_size_right, qkv_factor
            )
            bs_mask_fn = partial(
                bs_mask.apply_mask,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=False,
                mask_local=False,
                aux_tensors=aux_tensors,
            )
            if total_block_cnt > 0:
                gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
                self.load_Q(
                    gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1]
                )
                cute.arch.cp_async_commit_group()
                if const_expr(self.Q_in_regs):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                    cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
                    cute.arch.barrier()
                else:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                mma_one_n_block = partial(
                    self.mma_one_n_block_bs,
                    mma_params=mma_params,
                    smem_copy_params=smem_copy_params,
                    softmax=softmax,
                    load_K=load_K,
                    load_V=load_V,
                    score_mod=self.score_mod,
                    batch_idx=batch_size,
                    head_idx=num_head,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )
                run_block_sparse_mainloop_sm80(
                    blocksparse_tensors,
                    batch_size,
                    num_head,
                    m_block,
                    mma_one_n_block,
                    bs_mask_fn,
                    self.mask_mod,
                    fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                    qkv_factor,
                    subtile,
                    seqlen,
                )
            bs_row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, bs_row_scale)
            bs_sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                bs_sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                num_head,
                batch_size,
            )
        if const_expr(blocksparse_tensors is None):
            if n_block_max > n_block_min:
                gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
                self.load_Q(
                    gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1]
                )
                cute.arch.cp_async_commit_group()

                def preprocess_Q():
                    cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
                    if const_expr(self.Q_in_regs):
                        cute.arch.barrier()
                        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                        cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

                if const_expr(self.Q_in_regs):
                    load_K(n_block, smem_pipe_write=0, need_predicates=True)
                    cute.arch.cp_async_commit_group()
                    preprocess_Q()
                    cute.arch.barrier()
                for stage in cutlass.range_constexpr(self.num_stages):
                    if const_expr(not self.Q_in_regs or stage > 0):
                        if stage == 0 or n_block - stage >= 0:
                            load_K(
                                n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0
                            )
                        cute.arch.cp_async_commit_group()
                    if const_expr(stage < self.num_stages - 1):
                        if stage == 0 or n_block - stage >= 0:
                            load_V(
                                n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0
                            )
                        cute.arch.cp_async_commit_group()
                if const_expr(not self.Q_in_regs):
                    preprocess_Q()
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
                    batch_idx=batch_size,
                    head_idx=num_head,
                    m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                )
                smem_pipe_read = Int32(0)
                smem_pipe_write = Int32(self.num_stages - 1)
                compute_one_n_block(
                    n_block,
                    smem_pipe_read,
                    smem_pipe_write,
                    is_first_n_block=True,
                    seqlen=seqlen,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(
                        n_block_max - 1 - n_block_min_causal_local_mask, unroll=1
                    ):
                        n_block = n_block_max - 2 - n_tile
                        compute_one_n_block(
                            n_block,
                            smem_pipe_read,
                            smem_pipe_write,
                            seqlen=seqlen,
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                        )
                        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
                for n_tile in cutlass.range(n_block - n_block_min, unroll=1):
                    compute_one_n_block(
                        n_block - n_tile - 1,
                        smem_pipe_read,
                        smem_pipe_write,
                        seqlen=seqlen,
                        is_first_n_block=False,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                    )
                    smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                    smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale)
            sO = cute.make_tensor(sQ.iterator, sO_layout)
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
                num_head,
                batch_size,
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
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """

        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        sync()

        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )
            cute.arch.cp_async_commit_group()

        load_V_next()
        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[
                None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
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
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)
            cute.arch.cp_async_commit_group()

        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[
                None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_V,
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
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        """Process one KV block for block-sparse attention (load, GEMM QK, mask, softmax, PV GEMM).

        Unlike compute_one_n_block, this does not overlap loads with the next block since the
        next block address is not known ahead of time in the block-sparse case.
        """
        acc_S = cute.make_rmem_tensor(
            mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_S.fill(0.0)
        load_K(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()
        load_V(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
        )
        if const_expr(score_mod is not None):
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
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=True)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, 0],
            smem_copy_params.smem_thr_copy_V,
        )

    @cute.jit
    def load_paged(
        self,
        page_table: cute.Tensor,
        mX: cute.Tensor,
        sX: cute.Tensor,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        tile_dim: cutlass.Constexpr,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # After the layout transpose X is (page_size, dim, head, page).
        # Predicate the row before looking up its page; masked pages may be invalid.
        stage = smem_pipe_write if const_expr(self.num_stages > 1) else 0
        for i in cutlass.range_constexpr(cute.ceil_div(self.tile_n * tile_dim, self.num_threads)):
            idx = tidx + i * self.num_threads
            row = idx // tile_dim
            dim = idx % tile_dim
            if row < self.tile_n:
                logical_row = block * self.tile_n + row
                if logical_row >= 0 and logical_row < seqlen and dim < mX.shape[1]:
                    page = page_table[batch_idx, logical_row // mX.shape[0]]
                    sX[row, dim, stage] = mX[logical_row % mX.shape[0], dim, head_idx, page].to(
                        self.dtype
                    )
                else:
                    sX[row, dim, stage] = self.dtype(0)

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ):
        if const_expr(self.input_dtype == cutlass.Float8E4M3FN):
            src = gmem_thr_copy.partition_S(gQ)
            dst = gmem_thr_copy.partition_D(sQ)
            coords = gmem_thr_copy.partition_S(
                cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            )
            for i in cutlass.range_constexpr(cute.size(src)):
                row, dim = coords[i]
                if row < seqlen - block * self.tile_m and dim < headdim:
                    dst[i] = src[i].to(self.dtype)
                else:
                    dst[i] = self.dtype(0)
        else:
            super().load_Q(gmem_thr_copy, gQ, sQ, block, seqlen, headdim)

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        if const_expr(self.input_dtype == cutlass.Float8E4M3FN):
            self.load_fp8_tile(tKgK, tKsK, tKcK, block, smem_pipe_write, seqlen, self.head_dim)
        else:
            super().load_K(
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
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        if const_expr(self.input_dtype == cutlass.Float8E4M3FN):
            self.load_fp8_tile(tVgV, tVsV, tVcV, block, smem_pipe_write, seqlen, self.head_dim_v)
        else:
            super().load_V(
                gmem_tiled_copy,
                tVgV,
                tVsV,
                tVcV,
                t0VcV,
                tVpV,
                block,
                smem_pipe_write,
                seqlen,
                need_predicates,
            )

    @cute.jit
    def load_fp8_tile(
        self,
        gX: cute.Tensor,
        sX: cute.Tensor,
        coords: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        headdim: cutlass.Constexpr,
    ):
        # FP8 exists only in GMEM/registers. Convert each tile into BF16 shared
        # memory, then use the ordinary FP32-accumulating SM80 tensor core MMA.
        stage = smem_pipe_write if const_expr(self.num_stages > 1) else 0
        src = gX[None, None, None, block]
        dst = sX[None, None, None, stage]
        for i in cutlass.range_constexpr(cute.size(dst)):
            row, dim = coords[i]
            if row < self.tile_n:
                if block >= 0 and row < seqlen - block * self.tile_n and dim < headdim:
                    dst[i] = src[i].to(self.dtype)
                else:
                    dst[i] = self.dtype(0)
