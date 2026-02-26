import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Tuned configs to balance SM occupancy and register pressure
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32},  num_warps=4, num_stages=4),
    ],
    key=['seq_len', 'HEAD_DIM'],
)
@triton.jit
def _flash_fwd_kernel(
    Q, K, V, sm_scale_log2,
    Out, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_os, stride_om, stride_ok,
    stride_lz, stride_lh, stride_ls, stride_lm,
    Z, H_Q, H_KV, seq_len, num_kv_splits,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPLIT_KV: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_s  = tl.program_id(1)
    pid_zh = tl.program_id(2)
    
    # Grouped-Query Attention (GQA) Logic
    off_z   = pid_zh // H_Q
    off_hq  = pid_zh  % H_Q
    GROUP_SIZE = H_Q // H_KV
    off_hkv = off_hq // GROUP_SIZE # Map Q head to the corresponding K/V head

    # Split-KV partition indices
    kv_chunk = tl.cdiv(seq_len, num_kv_splits)
    kv_start = pid_s * kv_chunk
    kv_end   = tl.minimum(kv_start + kv_chunk, seq_len)
    
    # Causal loop splitting limits
    if IS_CAUSAL:
        kv_end = tl.minimum(kv_end, (pid_m + 1) * BLOCK_M)
        # Identify the boundary where no causal mask is needed
        kv_end_unmasked = tl.minimum(kv_end, pid_m * BLOCK_M)
        # Prevent backward loop indexing when blocks are fully masked
        kv_end_unmasked = tl.maximum(kv_start, kv_end_unmasked)
    else:
        kv_end_unmasked = kv_end

    # Hardware-accelerated Block Pointers (TMA)
    q_ptr = tl.make_block_ptr(
        base=Q + off_z * stride_qz + off_hq * stride_qh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    k_ptr = tl.make_block_ptr(
        base=K + off_z * stride_kz + off_hkv * stride_kh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    v_ptr = tl.make_block_ptr(
        base=V + off_z * stride_vz + off_hkv * stride_vh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )

    # Load Q block once
    q = tl.load(q_ptr, boundary_check=(0, 1))

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],               dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],     dtype=tl.float32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # =======================================================
    # LOOP 1: Fast-path (No causal mask evaluation needed)
    # =======================================================
    for start_n in range(kv_start, kv_end_unmasked, BLOCK_N):
        k = tl.load(k_ptr, boundary_check=(0, 1))
        v = tl.load(v_ptr, boundary_check=(0, 1))

        # Base-2 scaled dot product
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale_log2
        
        m_block = tl.max(scores, axis=1)
        m_new   = tl.maximum(m_i, m_block)
        
        # BF16/FP16 NaN Shield: Prevents (-inf - (-inf) = NaN) when entire block is masked
        m_new_math = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha   = tl.math.exp2(m_i - m_new_math)
        exp_s   = tl.math.exp2(scores - m_new_math[:, None])

        l_i = l_i * alpha + tl.sum(exp_s, axis=1)
        acc = acc * alpha[:, None] + tl.dot(exp_s.to(INPUT_DTYPE), v, out_dtype=tl.float32)
        m_i = m_new

        # Advance block pointers (TMA glides)
        k_ptr = tl.advance(k_ptr, (BLOCK_N, 0))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    # =======================================================
    # LOOP 2: Diagonal blocks (Causal mask required)
    # =======================================================
    if IS_CAUSAL:
        for start_n in range(kv_end_unmasked, kv_end, BLOCK_N):
            k = tl.load(k_ptr, boundary_check=(0, 1))
            v = tl.load(v_ptr, boundary_check=(0, 1))

            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale_log2
            
            # Apply causal mask
            offs_n = start_n + tl.arange(0, BLOCK_N)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))

            m_block = tl.max(scores, axis=1)
            m_new   = tl.maximum(m_i, m_block)
            
            # BF16 NaN Shield
            m_new_math = tl.where(m_new == float('-inf'), 0.0, m_new)
            alpha   = tl.math.exp2(m_i - m_new_math)
            exp_s   = tl.math.exp2(scores - m_new_math[:, None])

            l_i = l_i * alpha + tl.sum(exp_s, axis=1)
            acc = acc * alpha[:, None] + tl.dot(exp_s.to(INPUT_DTYPE), v, out_dtype=tl.float32)
            m_i = m_new

            k_ptr = tl.advance(k_ptr, (BLOCK_N, 0))
            v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    # Epilogue: Compute final output and LogSumExp
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    out_p  = acc / safe_l[:, None]
    
    # Convert LSE back to base 'e' and force fully masked splits to -inf
    lse = tl.where(l_i > 0.0, tl.log(safe_l) + m_i * 0.6931471805599453, float("-inf"))

    # Memory writing phase
    if SPLIT_KV:
        # Save partial results in FP32 to DRAM for global reduction
        out_partial_ptr = tl.make_block_ptr(
            base=Out + off_z * stride_oz + off_hq * stride_oh + pid_s * stride_os,
            shape=(seq_len, HEAD_DIM),
            strides=(stride_om, stride_ok),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0)
        )
        tl.store(out_partial_ptr, out_p, boundary_check=(0, 1))
        row_mask = offs_m < seq_len
        tl.store(LSE + off_z * stride_lz + off_hq * stride_lh + pid_s * stride_ls + offs_m * stride_lm,
                 lse, mask=row_mask)
    else:
        # Fast-path: Write directly to target output in native precision (FP16/BF16)
        out_ptr = tl.make_block_ptr(
            base=Out + off_z * stride_oz + off_hq * stride_oh,
            shape=(seq_len, HEAD_DIM),
            strides=(stride_om, stride_ok),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0)
        )
        tl.store(out_ptr, out_p.to(INPUT_DTYPE), boundary_check=(0, 1))
        row_mask = offs_m < seq_len
        tl.store(LSE + off_z * stride_lz + off_hq * stride_lh + offs_m * stride_lm,
                 lse, mask=row_mask)


@triton.jit
def _flash_reduce_kernel(
    Out_partial, LSE_partial, Out,
    stride_opz, stride_oph, stride_ops, stride_opm, stride_opk,
    stride_lpz, stride_lph, stride_lps, stride_lpm,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H_Q, seq_len, num_kv_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_zh = tl.program_id(1)
    off_z  = pid_zh // H_Q
    off_hq = pid_zh  % H_Q

    offs_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # 1. Find global max LSE across all KV splits
    m_final = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    for s in range(0, num_kv_splits):
        lse_s = tl.load(LSE_partial + off_z * stride_lpz + off_hq * stride_lph
                        + s * stride_lps + offs_q * stride_lpm,
                        mask=offs_q < seq_len, other=float('-inf'))
        m_final = tl.maximum(m_final, lse_s)

    # 2. Weighted reduction
    acc    = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_norm = tl.zeros([BLOCK_M],           dtype=tl.float32)
    for s in range(0, num_kv_splits):
        lse_s = tl.load(LSE_partial + off_z * stride_lpz + off_hq * stride_lph
                        + s * stride_lps + offs_q * stride_lpm,
                        mask=offs_q < seq_len, other=float('-inf'))
        
        # Reduction NaN Shield
        m_final_math = tl.where(m_final == float('-inf'), 0.0, m_final)
        w = tl.exp(lse_s - m_final_math)
        w = tl.where(lse_s == float('-inf'), 0.0, w)
        
        out_s = tl.load(Out_partial + off_z * stride_opz + off_hq * stride_oph
                        + s * stride_ops + offs_q[:, None] * stride_opm
                        + offs_d[None, :] * stride_opk,
                        mask=offs_q[:, None] < seq_len, other=0.0)
        
        acc    += w[:, None] * out_s
        l_norm += w

    # Avoid zero-division for fully masked sequence rows
    out = acc / tl.where(l_norm[:, None] > 0.0, l_norm[:, None], 1.0)
    
    tl.store(Out + off_z * stride_oz + off_hq * stride_oh
             + offs_q[:, None] * stride_om + offs_d[None, :] * stride_on,
             out, mask=offs_q[:, None] < seq_len)

def flash_attention(q, k, v, sm_scale, num_kv_splits=None, is_causal=False):
    """
    Triton-based FlashAttention-2 implementation with GQA and Split-KV support.
    """
    Z, H_Q, seq_len, HEAD_DIM = q.shape
    _, H_KV, _, _ = k.shape

    assert H_Q % H_KV == 0, "Number of Q heads must be a multiple of KV heads for GQA"

    # Enforce memory contiguity for TMA Block Pointers
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if q.dtype == torch.float16:
        input_dtype = tl.float16
    elif q.dtype == torch.bfloat16:
        input_dtype = tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype {q.dtype}. Use FP16 or BF16.")

    # Convert hardware standard base-e scale to base-2 scale
    sm_scale_log2 = sm_scale * 1.4426950408889634

    # Dynamic SM-occupancy heuristic for Split-KV
    num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
    if num_kv_splits is None:
        num_q_blocks = math.ceil(seq_len / 128)
        tb_fa1 = num_q_blocks * Z * H_Q
        
        if tb_fa1 >= num_SMs * 0.8:
            num_kv_splits = 1 # GPU is saturated, no split needed
        else:
            max_splits = math.ceil(seq_len / 64)
            num_kv_splits = max(1, min(math.ceil(num_SMs / tb_fa1), max_splits))

    split_kv = num_kv_splits > 1
    out = torch.empty_like(q)

    grid_fwd = lambda META: (math.ceil(seq_len / META['BLOCK_M']), num_kv_splits, Z * H_Q)

    if split_kv:
        out_partial = torch.empty((Z, H_Q, num_kv_splits, seq_len, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse_partial = torch.empty((Z, H_Q, num_kv_splits, seq_len), dtype=torch.float32, device=q.device)
        
        _flash_fwd_kernel[grid_fwd](
            q, k, v, sm_scale_log2, out_partial, lse_partial,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3), out_partial.stride(4),
            lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2), lse_partial.stride(3),
            Z, H_Q, H_KV, seq_len, num_kv_splits,
            HEAD_DIM=HEAD_DIM, INPUT_DTYPE=input_dtype, IS_CAUSAL=is_causal, SPLIT_KV=True,
        )

        # Reduction step is memory-bound, BLOCK_M_REDUCE=128 is optimal for DRAM reads
        BLOCK_M_REDUCE = 128
        grid_reduce = (math.ceil(seq_len / BLOCK_M_REDUCE), Z * H_Q)
        _flash_reduce_kernel[grid_reduce](
            out_partial, lse_partial, out,
            out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3), out_partial.stride(4),
            lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2), lse_partial.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H_Q, seq_len, num_kv_splits,
            HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M_REDUCE,
        )
    else:
        lse = torch.empty((Z, H_Q, seq_len), dtype=torch.float32, device=q.device)
        _flash_fwd_kernel[grid_fwd](
            q, k, v, sm_scale_log2, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), 0, out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), 0, lse.stride(2),
            Z, H_Q, H_KV, seq_len, 1, 
            HEAD_DIM=HEAD_DIM, INPUT_DTYPE=input_dtype, IS_CAUSAL=is_causal, SPLIT_KV=False,
        )

    return out.to(q.dtype)

if __name__ == '__main__':
    # Simple example usage for Llama 2 70B style GQA
    torch.manual_seed(42)
    Z, H_Q, H_KV, seq_len, HEAD_DIM = 2, 32, 8, 2048, 64 
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    is_causal = True 

    q = torch.randn((Z, H_Q, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    k = torch.randn((Z, H_KV, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    v = torch.randn((Z, H_KV, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')

    output = flash_attention(q, k, v, sm_scale, is_causal=is_causal)
    print(f"Output shape: {output.shape} | Expected: ({Z}, {H_Q}, {seq_len}, {HEAD_DIM})")
    print("Execution successful!")