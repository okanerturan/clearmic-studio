"""Ultra-fast Apple Silicon (MLX + custom Metal fused LSTM) engine for VoxLock.

Loads weights directly from VoxLock's ONNX checkpoint (`chunk.onnx`) and executes
100% on Apple Silicon GPU with:
  - Coalesced threadgroup-SRAM custom Metal LSTM kernels (`R` laid out as `[K, 4, H]`
    for 100% contiguous 512-byte coalesced reads across 256 threads + broadcast from SRAM):
      * Causal time-LSTM kernel (runs all K frames in a single Metal dispatch).
      * Fused bidirectional band-LSTM kernel (runs all 32 bands × both directions in 1 dispatch).
  - Precomputed `X @ W.T + Wb + Rb` on M5 Tensor Cores prior to recurrence.
  - Quantized `mk_w1` / `mk_w3` (10.5M decoder params) via `mx.quantized_matmul` (Q8) or BF16/FP16.
  - Full `mx.compile` fusion of STFT, enrollment projection, band-split, cross-attention,
    norms, projections, GLU mask decoder, and iSTFT.
"""

from __future__ import annotations

import ctypes
import pathlib
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import voxlock

SR, HOP, NFFT, PAD = 16000, 128, 512, 256
WIN2 = np.ascontiguousarray((0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32) ** 2)
GAIN = -0.5

_NATIVE_DSP = None
_NATIVE_LIB_PATH = pathlib.Path(__file__).resolve().parent / "libnative_dsp.dylib"
if _NATIVE_LIB_PATH.is_file():
    _NATIVE_DSP = ctypes.CDLL(str(_NATIVE_LIB_PATH))
    _NATIVE_DSP.voxlock_ola_neon.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

# ---------------------------------------------------------------------------
# Custom 1024-Thread Vectorized INT8 (char4) Threadgroup-SRAM Metal LSTM Kernels
# ---------------------------------------------------------------------------
# Layout of R_q: [4*256, 256] as int8 (packed char4 -> [1024, 64] per dir), with
# per-row float scale R_s [1024]. Wait: for 100% coalesced reads across 1024 threads,
# pack 4 consecutive k's into one int32 (char4) at layout [64, 1024] -> R_i32[k4 * 1024 + tid]!
# Every thread tid in 0..1023 computes ONE gate (i, o, f, or c) in 64 vectorized
# 4-way dot products from threadgroup SRAM h4_sh[64], then threads 0..255 update (c, h)!

_LSTM_CAUSAL_INT8_SRC = """
    // Grid: (1024, B, 1), Threadgroup: (1024, 1, 1)
    // xw:   [B, T, 1024]
    // R_q:  [64, 1024] int32 (each int32 packs 4 int8 weights for k = 4*k4 .. 4*k4+3)
    // R_s:  [1024] float32 (per-row dequant scale)
    // h0, c0: [B, 256] float32
    uint tid = thread_position_in_threadgroup.x;  // 0..1023
    uint b = thread_position_in_grid.y;
    uint T_len = xw_shape[1];

    threadgroup float4 h4_sh[64];
    threadgroup float g_sh[1024];

    float h_val = 0.0f;
    float c_val = 0.0f;
    if (tid < 256) {
        h_val = h0[b * 256 + tid];
        c_val = c0[b * 256 + tid];
        ((threadgroup float*)h4_sh)[tid] = h_val;
    }
    float scale = R_s[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < T_len; ++t) {
        float acc = 0.0f;
        #pragma unroll 8
        for (uint k4 = 0; k4 < 64; ++k4) {
            float4 hv = h4_sh[k4];
            char4 w = as_type<char4>(R_q[k4 * 1024 + tid]);
            acc = fma(hv.x, (float)w.x, acc);
            acc = fma(hv.y, (float)w.y, acc);
            acc = fma(hv.z, (float)w.z, acc);
            acc = fma(hv.w, (float)w.w, acc);
        }
        g_sh[tid] = fma(acc, scale, (float)xw[(b * T_len + t) * 1024 + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            float gi = clamp(g_sh[tid], -20.0f, 20.0f);
            float go = clamp(g_sh[256 + tid], -20.0f, 20.0f);
            float gf = clamp(g_sh[512 + tid], -20.0f, 20.0f);
            float gc = clamp(g_sh[768 + tid], -20.0f, 20.0f);
            float i_gate = 1.0f / (1.0f + fast::exp(-gi));
            float o_gate = 1.0f / (1.0f + fast::exp(-go));
            float f_gate = 1.0f / (1.0f + fast::exp(-gf));
            float c_in = tanh(gc);
            c_val = fma(f_gate, c_val, i_gate * c_in);
            h_val = o_gate * tanh(clamp(c_val, -20.0f, 20.0f));
            y[(b * T_len + t) * 256 + tid] = (T)h_val;
            ((threadgroup float*)h4_sh)[tid] = h_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 256) {
        h_out[b * 256 + tid] = h_val;
        c_out[b * 256 + tid] = c_val;
    }
"""

_LSTM_BIDIR_INT8_SRC = """
    // Grid: (1024, B, 2), Threadgroup: (1024, 1, 1)
    // xw:  [B, S, 2048]
    // R_q: [2, 64, 1024] int32
    // R_s: [2, 1024] float32
    uint tid = thread_position_in_threadgroup.x;
    uint b = thread_position_in_grid.y;
    uint dir = thread_position_in_grid.z;
    uint S_len = xw_shape[1];

    threadgroup float4 h4_sh[64];
    threadgroup float g_sh[1024];

    float h_val = 0.0f;
    float c_val = 0.0f;
    uint r_dir_off = dir * 64 * 1024;
    uint xw_dir_off = dir * 1024;
    uint y_dir_off = dir * 256;
    float scale = R_s[dir * 1024 + tid];

    for (uint step = 0; step < S_len; ++step) {
        uint s = (dir == 0) ? step : (S_len - 1 - step);
        float g_val = (float)xw[(b * S_len + s) * 2048 + xw_dir_off + tid];
        if (step > 0) {
            float acc = 0.0f;
            #pragma unroll 8
            for (uint k4 = 0; k4 < 64; ++k4) {
                float4 hv = h4_sh[k4];
                char4 w = as_type<char4>(R_q[r_dir_off + k4 * 1024 + tid]);
                acc = fma(hv.x, (float)w.x, acc);
                acc = fma(hv.y, (float)w.y, acc);
                acc = fma(hv.z, (float)w.z, acc);
                acc = fma(hv.w, (float)w.w, acc);
            }
            g_val = fma(acc, scale, g_val);
        }
        g_sh[tid] = g_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            float gi = clamp(g_sh[tid], -20.0f, 20.0f);
            float go = clamp(g_sh[256 + tid], -20.0f, 20.0f);
            float gf = clamp(g_sh[512 + tid], -20.0f, 20.0f);
            float gc = clamp(g_sh[768 + tid], -20.0f, 20.0f);
            float i_gate = 1.0f / (1.0f + fast::exp(-gi));
            float o_gate = 1.0f / (1.0f + fast::exp(-go));
            float f_gate = 1.0f / (1.0f + fast::exp(-gf));
            float c_in = tanh(gc);
            c_val = fma(f_gate, c_val, i_gate * c_in);
            h_val = o_gate * tanh(clamp(c_val, -20.0f, 20.0f));
            y[(b * S_len + s) * 512 + y_dir_off + tid] = (T)h_val;
            ((threadgroup float*)h4_sh)[tid] = h_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_LSTM_CAUSAL_INT8_KERNEL = mx.fast.metal_kernel(
    name="voxlock_lstm_causal_i8",
    input_names=["xw", "R_q", "R_s", "h0", "c0"],
    output_names=["y", "h_out", "c_out"],
    source=_LSTM_CAUSAL_INT8_SRC,
)

_LSTM_BIDIR_INT8_KERNEL = mx.fast.metal_kernel(
    name="voxlock_lstm_bidir_i8",
    input_names=["xw", "R_q", "R_s"],
    output_names=["y"],
    source=_LSTM_BIDIR_INT8_SRC,
)


def lstm_causal_fused(xw: mx.array, R_q: mx.array, R_s: mx.array, h0: mx.array, c0: mx.array):
    B, T, _ = xw.shape
    return _LSTM_CAUSAL_INT8_KERNEL(
        inputs=[xw, R_q, R_s, h0, c0],
        template=[("T", xw.dtype)],
        grid=(1024, B, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, T, 256), (B, 256), (B, 256)],
        output_dtypes=[xw.dtype, mx.float32, mx.float32],
    )


def lstm_bidir_fused(xw: mx.array, R_q: mx.array, R_s: mx.array):
    B, S, _ = xw.shape
    return _LSTM_BIDIR_INT8_KERNEL(
        inputs=[xw, R_q, R_s],
        template=[("T", xw.dtype)],
        grid=(1024, B, 2),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, S, 512)],
        output_dtypes=[xw.dtype],
    )[0]


def _pack_R_int8(R_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row INT8 quantize ONNX R [1024, 256] -> R_i32 [64, 1024], scales [1024]."""
    max_abs = np.max(np.abs(R_np), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)  # [1024, 1]
    q = np.clip(np.round(R_np / scales), -127, 127).astype(np.int8)  # [1024, 256]
    # Reshape to [1024, 64, 4] -> transpose to [64, 1024, 4] -> view as int32 [64, 1024]
    q_packed = np.ascontiguousarray(q.reshape(1024, 64, 4).transpose(1, 0, 2)).view(np.int32).reshape(64, 1024)
    return q_packed, scales[:, 0]


# ---------------------------------------------------------------------------
# VoxLock MLX Network
# ---------------------------------------------------------------------------

class VoxLockMLXNetwork:
    def __init__(
        self,
        onnx_path: str,
        precision: Literal["fp32", "fp16", "bf16", "hybrid"] = "hybrid",
    ):
        m = onnx.load(onnx_path)
        inits = {t.name: onnx.numpy_helper.to_array(t).astype(np.float32) for t in m.graph.initializer}

        self.precision = precision
        self.dtype = {
            "fp32": mx.float32,
            "fp16": mx.float16,
            "bf16": mx.bfloat16,
            "hybrid": mx.float16,
        }[precision]
        self.use_q8_mask = precision == "hybrid"

        # 1. STFT Conv1d weight: ONNX [514, 1, 512] -> MLX conv1d [out=514, k=512, in=1]
        self.stft_w = mx.array(np.ascontiguousarray(inits["stft.w"].transpose(0, 2, 1))).astype(mx.float32)

        # 2. Band-split encoders (5 groups: 15x3, 10x6, 5x16, 1x64, 1x8 -> 32 bands)
        self.in_groups = []
        for g in range(5):
            # in_g{g}: [G, D, 1], in_b{g}: [G, D, 1], in_w{g}: [G, 128, D], in_c{g}: [G, 128, 1]
            self.in_groups.append((
                mx.array(inits[f"in_g{g}"][:, :, 0]).astype(mx.float32),       # [G, D]
                mx.array(inits[f"in_b{g}"][:, :, 0]).astype(mx.float32),       # [G, D]
                mx.array(np.ascontiguousarray(inits[f"in_w{g}"].transpose(0, 2, 1))).astype(self.dtype),  # [G, D, 128]
                mx.array(inits[f"in_c{g}"][:, :, 0]).astype(self.dtype),       # [G, 128]
            ))

        # 3. Cross-attention with enrollment (find Gemm_2 and Gemm_4 weights from nodes)
        q_w = o_w = None
        for n in m.graph.node:
            if n.op_type == "Gemm" and len(n.input) >= 3:
                if n.input[2] == "bq":
                    q_w = inits[n.input[1]]  # [128, 128]
                elif n.input[2] == "bo":
                    o_w = inits[n.input[1]]  # [128, 128]
        self.wq = mx.array(q_w).astype(self.dtype)
        self.bq = mx.array(inits["bq"]).astype(self.dtype)
        self.wo = mx.array(o_w).astype(self.dtype)
        self.bo = mx.array(inits["bo"]).astype(self.dtype)

        # 4. Extract 6 Dual-Path LSTM blocks directly from ONNX graph order
        time_lstms = []
        time_projs = []
        band_norms = []
        band_lstms = []
        band_projs = []
        ln_nodes = []

        for idx, n in enumerate(m.graph.node):
            if n.op_type == "LayerNormalization":
                ln_nodes.append((inits[n.input[1]], inits[n.input[2]]))
            elif n.op_type == "LSTM":
                W = inits[n.input[1]]  # [D, 1024, 128]
                R = inits[n.input[2]]  # [D, 1024, 256]
                B = inits[n.input[3]]  # [D, 2048]
                if W.shape[0] == 1:
                    time_lstms.append((W[0], R[0], B[0, :1024] + B[0, 1024:]))
                else:
                    band_lstms.append((W, R, B[:, :1024] + B[:, 1024:]))
            elif n.op_type == "Gemm":
                b_name = n.input[2]
                if "band_rnn.proj.bias" in b_name:
                    time_projs.append((inits[n.input[1]], inits[b_name]))  # [256, 128], [128]
                elif "band_comm.proj.bias" in b_name:
                    # transB=1 -> [128, 512], transpose to [512, 128]
                    band_projs.append((inits[n.input[1]].T.copy(), inits[b_name]))
            elif n.op_type == "Mul" and n.name.startswith("/band_comm/norm"):
                # Next node is Add
                add_n = m.graph.node[idx + 1]
                band_norms.append((inits[n.input[1]][:, 0], inits[add_n.input[1]][:, 0]))

        assert len(time_lstms) == 6 and len(band_lstms) == 6

        self.blocks = []
        for i in range(6):
            ln_w, ln_b = ln_nodes[i]
            tW, tR, tB = time_lstms[i]
            tpW, tpB = time_projs[i]
            bn_w, bn_b = band_norms[i]
            bW, bR, bB = band_lstms[i]
            bpW, bpB = band_projs[i]

            # Fuse forward & backward input projections for band Bi-LSTM into ONE [128, 2048] matmul!
            bW_cat = np.concatenate([bW[0].T, bW[1].T], axis=1)  # [128, 2048]
            bB_cat = np.concatenate([bB[0], bB[1]], axis=0)      # [2048]
            tRq, tRs = _pack_R_int8(tR)
            bRq0, bRs0 = _pack_R_int8(bR[0])
            bRq1, bRs1 = _pack_R_int8(bR[1])
            bRq = np.stack([bRq0, bRq1], axis=0)  # [2, 64, 1024]
            bRs = np.stack([bRs0, bRs1], axis=0)  # [2, 1024]

            self.blocks.append({
                "ln_w": mx.array(ln_w).astype(self.dtype),
                "ln_b": mx.array(ln_b).astype(self.dtype),
                "tW": mx.array(np.ascontiguousarray(tW.T)).astype(self.dtype),  # [128, 1024]
                "tB": mx.array(tB).astype(self.dtype),                          # [1024]
                "tRq": mx.array(tRq),                                           # [64, 1024] int32
                "tRs": mx.array(tRs),                                           # [1024] float32
                "tpW": mx.array(tpW).astype(self.dtype),                        # [256, 128]
                "tpB": mx.array(tpB).astype(self.dtype),                        # [128]
                "bn_w": mx.array(bn_w).astype(mx.float32),                      # [128]
                "bn_b": mx.array(bn_b).astype(mx.float32),                      # [128]
                "bW": mx.array(np.ascontiguousarray(bW_cat)).astype(self.dtype),# [128, 2048]
                "bB": mx.array(bB_cat).astype(self.dtype),                      # [2048]
                "bRq": mx.array(bRq),                                           # [2, 64, 1024] int32
                "bRs": mx.array(bRs),                                           # [2, 1024] float32
                "bpW": mx.array(np.ascontiguousarray(bpW)).astype(self.dtype),  # [512, 128]
                "bpB": mx.array(bpB).astype(self.dtype),                        # [128]
            })

        # 5. Mask decoder
        self.mk_g = mx.array(inits["mk_g"][:, :, 0]).astype(mx.float32)  # [32, 128]
        self.mk_b = mx.array(inits["mk_b"][:, :, 0]).astype(mx.float32)  # [32, 128]
        self.mk_c1 = mx.array(inits["mk_c1"][:, :, 0][:, None, :]).astype(self.dtype)  # [32, 1, 512]
        self.mk_c3 = mx.array(inits["mk_c3"][:, :, 0][:, None, :]).astype(self.dtype)  # [32, 1, 512]

        if self.use_q8_mask:
            # Quantize the 10.5M-param decoder heads (mk_w1 [32,512,128], mk_w3 [32,512,512]) to Q8!
            w1 = mx.array(inits["mk_w1"]).astype(self.dtype)
            w3 = mx.array(inits["mk_w3"]).astype(self.dtype)
            self.mk_w1_q = mx.quantize(w1, group_size=64, bits=8)
            self.mk_w3_q = mx.quantize(w3, group_size=64, bits=8)
            self.mk_w1 = self.mk_w3 = None
        else:
            self.mk_w1 = mx.array(np.ascontiguousarray(inits["mk_w1"].transpose(0, 2, 1))).astype(self.dtype)
            self.mk_w3 = mx.array(np.ascontiguousarray(inits["mk_w3"].transpose(0, 2, 1))).astype(self.dtype)
            self.mk_w1_q = self.mk_w3_q = None

        self.mk5 = []
        for g in range(5):
            self.mk5.append((
                mx.array(np.ascontiguousarray(inits[f"mk_w5_{g}"].transpose(0, 2, 1))).astype(self.dtype),  # [G, 512, out]
                mx.array(inits[f"mk_c5_{g}"][:, :, 0][:, None, :]).astype(self.dtype),                     # [G, 1, out]
            ))

        # 6. iSTFT basis
        self.istft_br = mx.array(inits["istft.br"]).astype(mx.float32)  # [257, 512]
        self.istft_bi = mx.array(inits["istft.bi"]).astype(mx.float32)  # [257, 512]

        # Compile front-end, per-block pre/post, and back-end!
        self._compile_stages()
        self._eval_weights()

    def _eval_weights(self):
        arrs = [self.stft_w, self.wq, self.bq, self.wo, self.bo, self.mk_g, self.mk_b, self.mk_c1, self.mk_c3, self.istft_br, self.istft_bi]
        for grp in self.in_groups:
            arrs.extend(grp)
        for blk in self.blocks:
            arrs.extend(blk.values())
        if self.use_q8_mask:
            arrs.extend(self.mk_w1_q)
            arrs.extend(self.mk_w3_q)
        else:
            arrs.extend([self.mk_w1, self.mk_w3])
        for grp in self.mk5:
            arrs.extend(grp)
        mx.eval(*arrs)

    def _compile_stages(self):
        eps_in = 1.1920928955078125e-07
        band_specs = [(0, 45, 15, 3), (45, 105, 10, 6), (105, 185, 5, 16), (185, 249, 1, 64), (249, 257, 1, 8)]

        def _frontend(wav, enroll_n, k_enr, v_enr):
            # wav: [1, S, 1] float32 -> conv1d -> [1, T, 514]
            spec = mx.conv1d(wav, self.stft_w, stride=HOP)[0]  # [T, 514]
            real = spec[:, :257]   # [T, 257]
            imag = spec[:, 257:]   # [T, 257]
            mag = mx.sqrt(real * real + imag * imag)  # [T, 257]

            # Enrollment spectral projection
            mag_norm = mag / mx.maximum(mx.sqrt(mx.sum(mag * mag, axis=1, keepdims=True)), 1e-12)
            attn_e = mx.softmax(mx.matmul(mag_norm, enroll_n), axis=-1)  # [T, 376]
            ref = mx.matmul(attn_e, mx.transpose(enroll_n, (1, 0)))      # [T, 257]
            ref = ref / mx.maximum(mx.sqrt(mx.sum(ref * ref, axis=1, keepdims=True)), 1e-12)
            proj = mx.sum(mag * ref, axis=1, keepdims=True) * ref        # [T, 257]

            # Stack [3, T, 257] -> transpose to [T, 3, 257]
            feat = mx.stack([real, imag, proj], axis=1)  # [T, 3, 257]
            T = real.shape[0]

            # Band split + norm + linear -> list of [G, T, 128]
            parts = []
            for g, (s0, s1, G, w) in enumerate(band_specs):
                # Slice bins [T, 3, G*w] -> [T, 3, G, w] -> [G, T, 3*w]
                sl = feat[:, :, s0:s1].reshape(T, 3, G, w)
                x = mx.transpose(sl, (2, 0, 1, 3)).reshape(G, T, 3 * w)
                mean = mx.mean(x, axis=-1, keepdims=True)
                var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
                ig, ib, iw, ic = self.in_groups[g]
                xn = ((x - mean) * mx.rsqrt(var + eps_in) * ig[:, None, :] + ib[:, None, :]).astype(self.dtype)
                parts.append(mx.matmul(xn, iw) + ic[:, None, :])
            x_bands = mx.concatenate(parts, axis=0)  # [32, T, 128]

            # Enrollment cross-attention
            q = (mx.matmul(x_bands, self.wq) + self.bq).reshape(32, T, 2, 64)
            q = mx.transpose(q, (0, 2, 1, 3))  # [32, 2, T, 64]
            a = mx.softmax(mx.matmul(q, k_enr), axis=-1)  # [32, 2, T, 298]
            ao = mx.transpose(mx.matmul(a, v_enr), (0, 2, 1, 3)).reshape(32, T, 128)
            ao = mx.matmul(ao, self.wo) + self.bo
            x_out = x_bands * ao  # [32, T, 128]
            return x_out, real, imag

        def _make_block_fns(blk):
            def _t_pre(x):
                xn = mx.fast.layer_norm(x, blk["ln_w"], blk["ln_b"], 1e-5)
                return mx.matmul(xn, blk["tW"]) + blk["tB"]  # [32, T, 1024]

            def _t_post_and_b_pre(x, yt):
                x = x + mx.matmul(yt, blk["tpW"]) + blk["tpB"]  # [32, T, 128]
                # Band norm: InstanceNorm over (128, 32) per frame T!
                xf = mx.transpose(x, (1, 0, 2)).astype(mx.float32)  # [T, 32, 128]
                mean = mx.mean(xf, axis=(1, 2), keepdims=True)
                var = mx.mean((xf - mean) ** 2, axis=(1, 2), keepdims=True)
                xf_n = ((xf - mean) * mx.rsqrt(var + 1e-5) * blk["bn_w"][None, None, :] + blk["bn_b"][None, None, :]).astype(self.dtype)
                xw_b = mx.matmul(xf_n, blk["bW"]) + blk["bB"]  # [T, 32, 2048]
                return x, xw_b

            def _b_post(x, yb):
                # yb: [T, 32, 512] -> [T, 32, 128] -> [32, T, 128]
                bo = mx.transpose(mx.matmul(yb, blk["bpW"]) + blk["bpB"], (1, 0, 2))
                return x + bo

            return mx.compile(_t_pre), mx.compile(_t_post_and_b_pre), mx.compile(_b_post)

        def _backend(x, real, imag):
            # x: [32, T, 128], real/imag: [T, 257]
            T = x.shape[1]
            xf = x.astype(mx.float32)
            mean = mx.mean(xf, axis=-1, keepdims=True)
            var = mx.mean((xf - mean) ** 2, axis=-1, keepdims=True)
            xn = ((xf - mean) * mx.rsqrt(var + eps_in) * self.mk_g[:, None, :] + self.mk_b[:, None, :]).astype(self.dtype)

            if self.use_q8_mask:
                w1q, w1s, w1b = self.mk_w1_q
                w3q, w3s, w3b = self.mk_w3_q
                h1 = mx.tanh(mx.quantized_matmul(xn, w1q, w1s, w1b, transpose=True, group_size=64, bits=8) + self.mk_c1)
                h2 = mx.tanh(mx.quantized_matmul(h1, w3q, w3s, w3b, transpose=True, group_size=64, bits=8) + self.mk_c3)
            else:
                h1 = mx.tanh(mx.matmul(xn, self.mk_w1) + self.mk_c1)
                h2 = mx.tanh(mx.matmul(h1, self.mk_w3) + self.mk_c3)

            # 5 band-group mask heads -> complex multiply -> concat [T, 257]
            # Wait: real/imag are [T, 257]; let's transpose to [257, T] or keep [T, G*w]
            real_t = mx.transpose(real, (1, 0))  # [257, T]
            imag_t = mx.transpose(imag, (1, 0))  # [257, T]
            b_offsets = [(0, 15, 0, 45, 3), (15, 25, 45, 105, 6), (25, 30, 105, 185, 16), (30, 31, 185, 249, 64), (31, 32, 249, 257, 8)]
            out_r_parts = []
            out_i_parts = []
            for g, (b0, b1, f0, f1, w) in enumerate(b_offsets):
                G = b1 - b0
                w5, c5 = self.mk5[g]
                # h2[b0:b1]: [G, T, 512] -> [G, T, 4*w] -> transpose [G, 4*w, T] -> reshape [G, 2, 2, w, T]
                y5 = (mx.matmul(h2[b0:b1], w5) + c5).astype(mx.float32)
                y5 = mx.transpose(y5, (0, 2, 1)).reshape(G, 2, 2, w, T)
                mask = y5[:, 0] * mx.sigmoid(y5[:, 1])  # [G, 2, w, T]
                mr = mask[:, 0].reshape(G * w, T)
                mi = mask[:, 1].reshape(G * w, T)
                sr = real_t[f0:f1]
                si = imag_t[f0:f1]
                out_r_parts.append(sr * mr - si * mi)
                out_i_parts.append(sr * mi + si * mr)

            out_r = mx.transpose(mx.concatenate(out_r_parts, axis=0), (1, 0))  # [T, 257]
            out_i = mx.transpose(mx.concatenate(out_i_parts, axis=0), (1, 0))  # [T, 257]
            frames = mx.matmul(out_r, self.istft_br) + mx.matmul(out_i, self.istft_bi)  # [T, 512]
            return frames

        def _spec_pass(real, imag, enroll_n, k_enr, v_enr, h_in, c_in):
            mag = mx.sqrt(real * real + imag * imag + 1e-18)
            mag_norm = mag / mx.maximum(mx.sqrt(mx.sum(mag * mag, axis=1, keepdims=True)), 1e-12)
            attn_e = mx.softmax(mx.matmul(mag_norm, enroll_n), axis=-1)
            ref = mx.matmul(attn_e, mx.transpose(enroll_n, (1, 0)))
            ref = ref / mx.maximum(mx.sqrt(mx.sum(ref * ref, axis=1, keepdims=True)), 1e-12)
            proj = mx.sum(mag * ref, axis=1, keepdims=True) * ref

            feat = mx.stack([real, imag, proj], axis=1)
            T = real.shape[0]
            parts = []
            for g, (s0, s1, G, w) in enumerate(band_specs):
                sl = feat[:, :, s0:s1].reshape(T, 3, G, w)
                x = mx.transpose(sl, (2, 0, 1, 3)).reshape(G, T, 3 * w)
                mean = mx.mean(x, axis=-1, keepdims=True)
                var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
                ig, ib, iw, ic = self.in_groups[g]
                xn = ((x - mean) * mx.rsqrt(var + eps_in) * ig[:, None, :] + ib[:, None, :]).astype(self.dtype)
                parts.append(mx.matmul(xn, iw) + ic[:, None, :])
            x_bands = mx.concatenate(parts, axis=0)

            q = mx.transpose((mx.matmul(x_bands, self.wq) + self.bq).reshape(32, T, 2, 64), (0, 2, 1, 3))
            a = mx.softmax(mx.matmul(q, k_enr), axis=-1)
            ao = mx.matmul(mx.transpose(mx.matmul(a, v_enr), (0, 2, 1, 3)).reshape(32, T, 128), self.wo) + self.bo
            x = x_bands * ao

            new_h, new_c = [], []
            for i, (blk, (f_tpre, f_tb, f_bpost)) in enumerate(zip(self.blocks, self._c_blocks)):
                xw_t = f_tpre(x)
                yt, ho, co = lstm_causal_fused(xw_t, blk["tRq"], blk["tRs"], h_in[i], c_in[i])
                new_h.append(ho)
                new_c.append(co)
                x, xw_b = f_tb(x, yt)
                yb = lstm_bidir_fused(xw_b, blk["bRq"], blk["bRs"])
                x = f_bpost(x, yb)

            xf = x.astype(mx.float32)
            mean = mx.mean(xf, axis=-1, keepdims=True)
            var = mx.mean((xf - mean) ** 2, axis=-1, keepdims=True)
            xn = ((xf - mean) * mx.rsqrt(var + eps_in) * self.mk_g[:, None, :] + self.mk_b[:, None, :]).astype(self.dtype)
            if self.use_q8_mask:
                w1q, w1s, w1b = self.mk_w1_q
                w3q, w3s, w3b = self.mk_w3_q
                h1 = mx.tanh(mx.quantized_matmul(xn, w1q, w1s, w1b, transpose=True, group_size=64, bits=8) + self.mk_c1)
                h2 = mx.tanh(mx.quantized_matmul(h1, w3q, w3s, w3b, transpose=True, group_size=64, bits=8) + self.mk_c3)
            else:
                h1 = mx.tanh(mx.matmul(xn, self.mk_w1) + self.mk_c1)
                h2 = mx.tanh(mx.matmul(h1, self.mk_w3) + self.mk_c3)

            real_t, imag_t = mx.transpose(real, (1, 0)), mx.transpose(imag, (1, 0))
            b_offsets = [(0, 15, 0, 45, 3), (15, 25, 45, 105, 6), (25, 30, 105, 185, 16), (30, 31, 185, 249, 64), (31, 32, 249, 257, 8)]
            out_r_parts, out_i_parts = [], []
            for g, (b0, b1, f0, f1, w) in enumerate(b_offsets):
                G = b1 - b0
                w5, c5 = self.mk5[g]
                y5 = (mx.matmul(h2[b0:b1], w5) + c5).astype(mx.float32)
                y5 = mx.transpose(y5, (0, 2, 1)).reshape(G, 2, 2, w, T)
                mask = y5[:, 0] * mx.sigmoid(y5[:, 1])
                mr, mi = mask[:, 0].reshape(G * w, T), mask[:, 1].reshape(G * w, T)
                sr_b, si_b = real_t[f0:f1], imag_t[f0:f1]
                out_r_parts.append(sr_b * mr - si_b * mi)
                out_i_parts.append(sr_b * mi + si_b * mr)

            out_r = mx.transpose(mx.concatenate(out_r_parts, axis=0), (1, 0))
            out_i = mx.transpose(mx.concatenate(out_i_parts, axis=0), (1, 0))
            return out_r, out_i, new_h, new_c

        self._c_frontend = mx.compile(_frontend)
        self._c_blocks = [_make_block_fns(blk) for blk in self.blocks]
        self._c_backend = mx.compile(_backend)
        self._spec_pass = _spec_pass

        # Single end-to-end compiled graph across all 6 Dual-Path LSTM blocks (and fused STFT multi-pass!)
        def _make_multipass(n_passes: int):
            gain_sc = mx.array(GAIN, dtype=mx.float32)
            def _fn(wav, enr, ke, ve, *states):
                spec = mx.conv1d(wav, self.stft_w, stride=HOP)[0]
                r, im = spec[:, :257], spec[:, 257:]
                out_states = []
                for p in range(n_passes):
                    h_in = states[p * 12 : p * 12 + 6]
                    c_in = states[p * 12 + 6 : p * 12 + 12]
                    r, im, nh, nc = _spec_pass(r, im, enr, ke, ve, h_in, c_in)
                    out_states.extend(nh)
                    out_states.extend(nc)
                    if p < n_passes - 1:
                        r, im = r * gain_sc, im * gain_sc
                frames = mx.matmul(r, self.istft_br) + mx.matmul(im, self.istft_bi)
                return (frames, *out_states)
            return mx.compile(_fn)

        self._c_passes = {1: _make_multipass(1), 2: _make_multipass(2), 3: _make_multipass(3)}

    def run_chunk(
        self,
        wav_mx: mx.array,
        enroll_n: mx.array,
        k_enr: mx.array,
        v_enr: mx.array,
        h_list: list[mx.array],
        c_list: list[mx.array],
        passes: int = 1,
    ) -> tuple[mx.array, list[mx.array], list[mx.array]]:
        # h_list / c_list each have length `6 * passes`
        flat_states = []
        for p in range(passes):
            flat_states.extend(h_list[p * 6 : (p + 1) * 6])
            flat_states.extend(c_list[p * 6 : (p + 1) * 6])
        fn = self._c_passes[passes]
        res = fn(wav_mx, enroll_n, k_enr, v_enr, *flat_states)
        new_h, new_c = [], []
        idx = 1
        for _ in range(passes):
            new_h.extend(res[idx : idx + 6])
            new_c.extend(res[idx + 6 : idx + 12])
            idx += 12
        return res[0], new_h, new_c


class VoxLockMLX:
    """Drop-in Apple Silicon MLX replacement for `voxlock.VoxLock`."""

    def __init__(self, precision: Literal["fp32", "fp16", "bf16", "hybrid"] = "hybrid"):
        mdir = voxlock.core.download("fp32")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self._enroll_sess = ort.InferenceSession(str(mdir / "enroll.onnx"), so, providers=["CPUExecutionProvider"])
        self.net = VoxLockMLXNetwork(str(mdir / "chunk.onnx"), precision=precision)
        self.precision = precision

    def stream(self, sample: np.ndarray, passes: int = 1) -> "VoxLockMLXStream":
        """Create a causal streaming session. `passes` (1, 2, or 3) runs STFT-fused multi-pass
        at a constant 24 ms algorithmic latency regardless of pass count!"""
        s = voxlock.core.fit_sample(sample)
        enr_n, k, v = self._enroll_sess.run(None, {"enroll": s[None]})
        enr_n_mx = mx.array(enr_n.astype(np.float32))
        k_mx = mx.array(k.astype(np.float32)).astype(self.net.dtype)
        v_mx = mx.array(v.astype(np.float32)).astype(self.net.dtype)
        mx.eval(enr_n_mx, k_mx, v_mx)
        return VoxLockMLXStream(self, enr_n_mx, k_mx, v_mx, passes=passes)

    def isolate(self, audio: np.ndarray, sample: np.ndarray, passes: int = 1) -> np.ndarray:
        st = self.stream(sample, passes=passes)
        return np.concatenate([st.push(audio, frames_per_run=100), st.flush()])[: len(audio)]


class VoxLockMLXStream:
    """Stateful causal streaming VoxLock session on MLX/Metal."""

    def __init__(self, vl: VoxLockMLX, enroll_n: mx.array, k_enr: mx.array, v_enr: mx.array, passes: int = 1):
        self.vl = vl
        self.net = vl.net
        self.passes = passes
        self.enroll_n = enroll_n
        self.k_enr = k_enr
        self.v_enr = v_enr
        self.h_list = [mx.zeros((32, 256), dtype=mx.float32) for _ in range(6 * passes)]
        self.c_list = [mx.zeros((32, 256), dtype=mx.float32) for _ in range(6 * passes)]
        mx.eval(*self.h_list, *self.c_list)
        self._start()

    def _start(self):
        self.head = np.zeros(0, dtype=np.float32)
        self.buf = None
        self.ola = np.zeros(NFFT, dtype=np.float32)
        self.env = np.zeros(NFFT, dtype=np.float32)
        self.dropped = 0
        self.n_in = 0
        self.n_out = 0

    def _run(self, frames_per_run: int) -> np.ndarray:
        out = []
        while self.buf is not None and len(self.buf) >= NFFT:
            k = min(frames_per_run, 1 + (len(self.buf) - NFFT) // HOP)
            seg = self.buf[: (k - 1) * HOP + NFFT]
            wav_mx = mx.array(np.ascontiguousarray(seg, dtype=np.float32))[None, :, None]
            fr_mx, self.h_list, self.c_list = self.net.run_chunk(
                wav_mx, self.enroll_n, self.k_enr, self.v_enr, self.h_list, self.c_list, passes=self.passes
            )
            mx.eval(fr_mx, *self.h_list, *self.c_list)
            fr = np.array(fr_mx, copy=False)  # [k, 512]

            ola = np.concatenate([self.ola, np.zeros(k * HOP, dtype=np.float32)])
            env = np.concatenate([self.env, np.zeros(k * HOP, dtype=np.float32)])
            if _NATIVE_DSP is not None:
                fr_c = np.ascontiguousarray(fr, dtype=np.float32)
                _NATIVE_DSP.voxlock_ola_neon(fr_c.ctypes.data, k, WIN2.ctypes.data, ola.ctypes.data, env.ctypes.data)
            else:
                for j in range(k):
                    at = j * HOP
                    ola[at : at + NFFT] += fr[j]
                    env[at : at + NFFT] += WIN2
            done = k * HOP
            y = ola[:done] / np.where(env[:done] > 1e-11, env[:done], 1.0)
            self.ola, self.env = ola[done:], env[done:]
            self.buf = self.buf[done:]
            cut = min(self.dropped, len(y))
            self.dropped -= cut
            out.append(y[cut:] * GAIN)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray, frames_per_run: int = 100) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        self.n_in += len(chunk)
        if self.buf is None:
            self.head = np.concatenate([self.head, chunk])
            if len(self.head) <= PAD:
                return np.zeros(0, dtype=np.float32)
            self.buf = np.concatenate([self.head[1 : PAD + 1][::-1], self.head])
            self.dropped = PAD
        else:
            self.buf = np.concatenate([self.buf, chunk])
        y = self._run(frames_per_run)
        self.n_out += len(y)
        return y

    def flush(self) -> np.ndarray:
        if self.buf is None:
            self._start()
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate([self.buf, self.buf[-PAD - 1 : -1][::-1]])
        nfr = 1 + (len(x) - NFFT) // HOP
        x = np.pad(x, (0, max(0, (nfr - 1) * HOP + NFFT - len(x))))
        self.buf = x
        y = self._run(100)
        y = np.concatenate([y, self.ola / np.where(self.env > 1e-11, self.env, 1.0) * GAIN])
        y = y[: max(0, self.n_in - self.n_out)]
        self._start()
        return y
