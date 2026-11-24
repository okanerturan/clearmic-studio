"""All-Silicon Apple M5 Multi-Accelerator Real-Time Voice Isolation + 48 kHz Restoration Engine.

Simultaneously utilizes all three hardware accelerators on Apple M5 with ZERO PyTorch at runtime:
  1. Apple M5 Metal GPU (MLX + custom Metal shaders):
       - Stage 1A: Inworld Voice Isolator Realtime (BF16 backbone + Q8 decoder + Q8/Q4 two-tier KV
                   + algebraically folded Mamba + float4 threadgroup-SRAM Metal scan)
       - Stage 1B (gpu mode): VoxLock Causal TSE (1024-thread char4 INT8 Metal LSTM + Q8 decoder)
       - Stage 2A: MossFormer2_SR_48K 24-layer Transformer+FSMN (MLX BF16, zero-PyTorch cache)
  2. Apple M5 Neural Engine (CoreML ANE via `CPU_AND_NE`):
       - Stage 2B: HiFi-GAN 48 kHz 55.1M-param neural vocoder running asynchronously on ANE
                   in parallel with GPU stages (0% GPU usage!).
  3. Apple M5 Super CPU (native C++ `libnative_dsp.dylib`: Accelerate vDSP + ARM64 NEON):
       - vDSP_fft_zrip 2048-pt STFT/iSTFT, NEON overlap-add, streaming soxr 3:1/1:3 resampling,
         causal 7.2 kHz crossover + 80 Hz mastering HPF, and optional CPU VoxLock mode.
"""

from __future__ import annotations

import ctypes
import pathlib
import queue
import threading
import time
import warnings
from typing import Literal

warnings.filterwarnings("ignore")

import mlx.core as mx
import numpy as np
from scipy import signal
import soxr

import inworld_mlx
import restorer_mlx
import voxlock_mlx


def _load_coreml_ane_lib() -> ctypes.CDLL | None:
    lib_path = pathlib.Path(__file__).resolve().parent / "libcoreml_ane.dylib"
    if not lib_path.is_file():
        return None
    lib = ctypes.CDLL(str(lib_path))
    lib.coreml_runner_create.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    lib.coreml_runner_create.restype = ctypes.c_void_p
    lib.coreml_runner_predict.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
    ]
    lib.coreml_runner_predict.restype = ctypes.c_int32
    lib.coreml_runner_destroy.argtypes = [ctypes.c_void_p]
    lib.coreml_runner_destroy.restype = None
    return lib


_COREML_ANE = _load_coreml_ane_lib()


def _load_native_dsp_biquad() -> ctypes.CDLL | None:
    lib_path = pathlib.Path(__file__).resolve().parent / "libnative_dsp.dylib"
    if not lib_path.is_file():
        return None
    lib = ctypes.CDLL(str(lib_path))
    lib.biquad_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.biquad_create.restype = ctypes.c_void_p
    lib.biquad_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.biquad_destroy.argtypes = [ctypes.c_void_p]
    lib.neon_add_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    return lib


_NATIVE_BIQUAD = _load_native_dsp_biquad()


class NativeCoreMLRunner:
    """GIL-free native Objective-C++ CoreML ANE runner via libcoreml_ane.dylib."""

    def __init__(self, pkg_path: pathlib.Path, t_frames: int, compute_units: int = 3):
        if _COREML_ANE is None:
            raise RuntimeError("libcoreml_ane.dylib not built")
        self._handle = _COREML_ANE.coreml_runner_create(
            str(pkg_path).encode("utf-8"),
            b"mel",
            b"wav",
            1, 80, t_frames,
            compute_units,  # 3 = CPU_AND_NE, 1 = CPU_AND_GPU, 2 = ALL
        )
        if not self._handle:
            raise RuntimeError(f"coreml_runner_create failed for {pkg_path}")
        self.in_count = 80 * t_frames
        self.out_buf = np.empty(t_frames * 256 + 1024, dtype=np.float32)

    def predict(self, mel_ncl_f32: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(mel_ncl_f32, dtype=np.float32)
        n = _COREML_ANE.coreml_runner_predict(
            self._handle,
            arr.ctypes.data,
            self.in_count,
            self.out_buf.ctypes.data,
            len(self.out_buf),
        )
        if n < 0:
            raise RuntimeError(f"coreml_runner_predict error {n}")
        return self.out_buf[:n].copy()

    def __del__(self):
        if getattr(self, "_handle", None) and _COREML_ANE is not None:
            _COREML_ANE.coreml_runner_destroy(self._handle)
            self._handle = None


class M5HyperStudioPipeline:
    """True concurrent multi-accelerator (Metal GPU || ANE || vDSP/NEON CPU) real-time pipeline."""

    def __init__(
        self,
        repo_root: pathlib.Path | str = pathlib.Path(__file__).resolve().parent.parent,
        inworld_block_frames: int = 4,
        inworld_cache_frames: int = 576,
        inworld_kv_bits: int = 8,
        voxlock_backend: Literal["mlx_gpu", "onnx_cpu"] = "mlx_gpu",
        voxlock_cpu_threads: int = 6,
        sr_window_frames: Literal[48, 96] = 96,
        hifi_backend: Literal["ane", "coreml_gpu", "mlx"] = "ane",
    ):
        self.repo_root = pathlib.Path(repo_root)
        self.inworld_bf = inworld_block_frames
        self.inworld_cf = inworld_cache_frames
        self.voxlock_backend = voxlock_backend
        self.sr_window_frames = sr_window_frames  # 96 = 24576 samples (0.512s), 48 = 12288 samples (0.256s)
        self.sr_win_samples = sr_window_frames * 256
        self.hifi_backend = hifi_backend

        # Stage 1A: Inworld MLX v4 Hybrid + vDSP/NEON
        self.inworld = inworld_mlx.MLXVoiceIsolator(
            precision="hybrid",
            block_frames=inworld_block_frames,
            cache_frames=inworld_cache_frames,
            kv_bits=inworld_kv_bits,
        )

        # Stage 1B: VoxLock (either MLX 1024t char4 INT8 Metal LSTM on GPU, or ONNX INT8 on CPU)
        if voxlock_backend == "mlx_gpu":
            self.voxlock = voxlock_mlx.VoxLockMLX(precision="hybrid")
        else:
            import realtime_pipeline as rp
            self.voxlock = rp.build_fast_voxlock(precision="int8", threads=voxlock_cpu_threads)

        # Stage 2A: MossFormer MLX BF16 (load HiFi-GAN in MLX only if hifi_backend == "mlx")
        self.restorer = restorer_mlx.MLXSpeechRestorer(
            repo_root=self.repo_root,
            precision="bf16",
            load_hifi=(hifi_backend == "mlx"),
        )

        # Stage 2B: Native GIL-free Objective-C++ CoreML ANE (or CoreML GPU) HiFi-GAN
        self.hifi_coreml = None
        if hifi_backend in ("ane", "coreml_gpu"):
            pkg = self.repo_root / "studio/coreml_cache" / f"hifigan_sr_t{sr_window_frames}.mlpackage"
            if not pkg.exists():
                raise FileNotFoundError(f"Missing CoreML package {pkg}; run CoreML export first.")
            cu_code = 3 if hifi_backend == "ane" else 1
            self.hifi_coreml = NativeCoreMLRunner(pkg, sr_window_frames, compute_units=cu_code)

    def warmup(self, mix_48k: np.ndarray, enroll_16k: np.ndarray) -> None:
        """Warms up all MLX compiled graphs, Metal shaders, and CoreML ANE kernels."""
        cs_48 = 480 * self.inworld_bf
        s1 = self.inworld.stream(block_frames=self.inworld_bf, cache_frames=self.inworld_cf)
        for i in range(0, len(mix_48k), cs_48):
            _ = s1.push(mix_48k[i : i + cs_48])
        _ = s1.flush()

        vs = self.voxlock.stream(enroll_16k)
        cs_16 = 160 * self.inworld_bf
        dummy_16 = np.zeros(cs_16 * 8, dtype=np.float32)
        for i in range(0, len(dummy_16), cs_16):
            _ = vs.push(dummy_16[i : i + cs_16])
        _ = vs.flush()

        # Warmup MossFormer + HiFi-GAN for exact window size
        seg = np.zeros(self.sr_win_samples, dtype=np.float32)
        mel = self.restorer._compute_mel_mlx(seg)
        ym = self.restorer._c_moss(mel)
        if self.hifi_coreml is not None:
            ym_ncl = np.ascontiguousarray(np.array(ym[0].astype(mx.float32), copy=False).T[None])
            _ = self.hifi_coreml.predict(ym_ncl)
        else:
            yg = self.restorer._c_hifi(ym)
            mx.eval(yg)

    def process_concurrent(
        self,
        mix_48k: np.ndarray,
        enroll_16k: np.ndarray,
        sr_hop_samples: int | None = None,
        crossover_hz: float = 7200.0,
        voxlock_passes: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Runs the FULL 3-stage causal pipeline concurrently across M5 GPU, ANE, and CPU!

        Architecture:
          - Producer (GPU thread): streams 40ms/80ms chunks through Inworld MLX -> 48k->16k resampler.
          - If voxlock_backend == "onnx_cpu": VoxLock runs in its OWN dedicated CPU thread in parallel!
            If voxlock_backend == "mlx_gpu": VoxLock runs right on the MLX GPU stream!
          - As soon as `sr_hop_samples` of Stage-1B audio accumulate, MossFormer MLX runs on GPU and
            immediately enqueues the enhanced Mel spectrogram `[1, 80, T]` to the dedicated
            **CoreML ANE Worker Thread** (`queue.Queue`), which runs HiFi-GAN on the Apple Neural
            Engine + causal 7.2 kHz Butterworth crossover + overlap-add crossfade IN PARALLEL!
        """
        sr = 48000
        win_s = self.sr_win_samples
        hop_s = sr_hop_samples or (win_s // 2)  # 50% overlap (e.g. 12288 samples = 256ms hop for 512ms win)
        ctx_s = win_s - hop_s
        xf = min(int(0.025 * sr), hop_s // 2)  # 25ms raised-cosine crossfade
        fade_in = (0.5 - 0.5 * np.cos(np.pi * np.arange(xf, dtype=np.float32) / max(xf, 1))).astype(np.float32)
        fade_out = 1.0 - fade_in

        chunk_48 = 480 * self.inworld_bf
        chunk_16 = 160 * self.inworld_bf

        # Queues for true concurrent multi-accelerator pipelining
        q_ane: queue.Queue = queue.Queue(maxsize=8)
        q_vl_in: queue.Queue = queue.Queue(maxsize=32) if self.voxlock_backend == "onnx_cpu" else None
        q_vl_out: queue.Queue = queue.Queue(maxsize=32) if self.voxlock_backend == "onnx_cpu" else None

        timings = {
            "inw_chunks_ms": [],
            "vl_chunks_ms": [],
            "moss_chunks_ms": [],
            "hifi_chunks_ms": [],
            "t_inw": 0.0,
            "t_vl": 0.0,
            "t_moss": 0.0,
            "t_hifi": 0.0,
        }

        # --- ANE Worker Thread (Stage 2B: HiFi-GAN on Apple Neural Engine + vDSP_biquad 7.2kHz Crossover) ---
        ane_out_chunks: list[np.ndarray] = []
        sos_lp = np.ascontiguousarray(signal.butter(4, crossover_hz, btype="lowpass", fs=sr, output="sos"), dtype=np.float64)
        sos_hp = np.ascontiguousarray(signal.butter(4, crossover_hz, btype="highpass", fs=sr, output="sos"), dtype=np.float64)
        bq_lp = _NATIVE_BIQUAD.biquad_create(sos_lp.ctypes.data, sos_lp.shape[0]) if _NATIVE_BIQUAD else None
        bq_hp = _NATIVE_BIQUAD.biquad_create(sos_hp.ctypes.data, sos_hp.shape[0]) if _NATIVE_BIQUAD else None
        zi_lp = np.zeros((sos_lp.shape[0], 2), dtype=np.float64)
        zi_hp = np.zeros((sos_hp.shape[0], 2), dtype=np.float64)

        def _emit_crossover(up_seg: np.ndarray, hi_seg: np.ndarray):
            nonlocal zi_lp, zi_hp
            n = len(up_seg)
            if n == 0:
                return
            if bq_lp is not None:
                up_c = np.ascontiguousarray(up_seg, dtype=np.float32)
                nh_c = np.ascontiguousarray(hi_seg, dtype=np.float32)
                low = np.empty(n, dtype=np.float32)
                high = np.empty(n, dtype=np.float32)
                out_c = np.empty(n, dtype=np.float32)
                _NATIVE_BIQUAD.biquad_process(bq_lp, up_c.ctypes.data, low.ctypes.data, n)
                _NATIVE_BIQUAD.biquad_process(bq_hp, nh_c.ctypes.data, high.ctypes.data, n)
                _NATIVE_BIQUAD.neon_add_f32(low.ctypes.data, high.ctypes.data, out_c.ctypes.data, n)
                ane_out_chunks.append(out_c)
            else:
                low, zi_lp = signal.sosfilt(sos_lp, up_seg, zi=zi_lp)
                high, zi_hp = signal.sosfilt(sos_hp, hi_seg, zi=zi_hp)
                ane_out_chunks.append((low + high).astype(np.float32))

        def _ane_worker():
            # Time-aligned overlap crossfade: hold back trailing `xf` samples of chunk k-1
            # and blend them with the EXACT same timestamps `rest[valid_new_start-xf : valid_new_start]`
            # from chunk k (zero phase discontinuity, zero boundary clicks!).
            held_rest = None
            held_up = None
            while True:
                item = q_ane.get()
                if item is None:
                    if held_rest is not None:
                        _emit_crossover(held_up, held_rest)
                    break
                ym_obj, up_hop, valid_new_start, valid_hop_len = item
                t0 = time.perf_counter()
                if self.hifi_coreml is not None:
                    rest = self.hifi_coreml.predict(ym_obj)
                else:
                    yg = self.restorer._c_hifi(ym_obj)
                    mx.eval(yg)
                    rest = np.array(yg, copy=False)
                dt = time.perf_counter() - t0
                timings["t_hifi"] += dt
                timings["hifi_chunks_ms"].append(dt * 1000.0)

                if held_rest is not None:
                    if valid_new_start >= xf and len(rest) >= valid_new_start:
                        curr_ov = rest[valid_new_start - xf : valid_new_start]
                        blended = held_rest * fade_out + curr_ov * fade_in
                    else:
                        blended = held_rest
                    _emit_crossover(held_up, blended)
                    held_rest = None
                    held_up = None

                new_hop = rest[valid_new_start : valid_new_start + valid_hop_len].copy()
                if len(new_hop) < valid_hop_len:
                    new_hop = np.pad(new_hop, (0, valid_hop_len - len(new_hop)))

                if valid_hop_len > xf:
                    _emit_crossover(up_hop[:-xf], new_hop[:-xf])
                    held_up = up_hop[-xf:].copy()
                    held_rest = new_hop[-xf:].copy()
                else:
                    _emit_crossover(up_hop, new_hop)
            if bq_lp is not None:
                _NATIVE_BIQUAD.biquad_destroy(bq_lp)
                _NATIVE_BIQUAD.biquad_destroy(bq_hp)

        ane_thread = threading.Thread(target=_ane_worker, daemon=True)
        ane_thread.start()

        # --- Optional CPU VoxLock Worker Thread (runs concurrently with GPU!) ---
        vl_streams = [self.voxlock.stream(enroll_16k) for _ in range(max(1, voxlock_passes))]
        vl_stream = vl_streams[0]

        def _vl_push_chain(x16: np.ndarray) -> np.ndarray:
            y = x16
            for st in vl_streams:
                if len(y) == 0:
                    return y
                y = st.push(y)
            return y

        def _vl_flush_chain() -> np.ndarray:
            parts = []
            y = vl_streams[0].flush()
            for st in vl_streams[1:]:
                p1 = st.push(y) if len(y) else np.zeros(0, dtype=np.float32)
                p2 = st.flush()
                y = np.concatenate([p1, p2]) if (len(p1) or len(p2)) else np.zeros(0, dtype=np.float32)
            return y

        def _cpu_vl_worker():
            while True:
                item = q_vl_in.get()
                if item is None:
                    t0 = time.perf_counter()
                    tail = vl_stream.flush()
                    dt = time.perf_counter() - t0
                    timings["t_vl"] += dt
                    if len(tail):
                        q_vl_out.put(tail)
                    q_vl_out.put(None)
                    break
                t0 = time.perf_counter()
                y16 = vl_stream.push(item)
                dt = time.perf_counter() - t0
                timings["t_vl"] += dt
                timings["vl_chunks_ms"].append(dt * 1000.0)
                if len(y16):
                    q_vl_out.put(y16)

        cpu_vl_thread = None
        if self.voxlock_backend == "onnx_cpu":
            cpu_vl_thread = threading.Thread(target=_cpu_vl_worker, daemon=True)
            cpu_vl_thread.start()

        # --- Helper to feed Stage-1B 16kHz output into Stage 2A (MossFormer MLX) -> ANE queue ---
        up_resampler = soxr.ResampleStream(16000, sr, 1, dtype="float32", quality="LQ")
        sr_buf = np.zeros(0, dtype=np.float32)
        sr_emitted_pos = 0
        stage1b_parts: list[np.ndarray] = []

        def _feed_stage2(y16_chunk: np.ndarray, is_last: bool = False):
            nonlocal sr_buf, sr_emitted_pos
            if len(y16_chunk):
                stage1b_parts.append(y16_chunk)
                up = up_resampler.resample_chunk(y16_chunk, last=False)
                if len(up):
                    sr_buf = np.concatenate([sr_buf, up])
            if is_last:
                up = up_resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if len(up):
                    sr_buf = np.concatenate([sr_buf, up])

            # Emit whenever we have `hop_s` new samples (or flush tail at end)
            while (len(sr_buf) - sr_emitted_pos) >= hop_s or (is_last and len(sr_buf) > sr_emitted_pos):
                avail = len(sr_buf) - sr_emitted_pos
                this_hop = min(hop_s, avail)
                end = sr_emitted_pos + this_hop
                win_start = max(0, sr_emitted_pos - ctx_s)
                seg_in = sr_buf[win_start:end]
                pad_len = max(0, win_s - len(seg_in))
                seg_in_p = np.pad(seg_in, (0, pad_len)) if pad_len else seg_in

                t0 = time.perf_counter()
                mel = self.restorer._compute_mel_mlx(seg_in_p)
                ym = self.restorer._c_moss(mel)
                if self.hifi_coreml is not None:
                    ym_f = ym[0].astype(mx.float32)
                    mx.eval(ym_f)
                    ym_payload = np.ascontiguousarray(np.array(ym_f, copy=False).T[None])
                else:
                    mx.eval(ym)
                    ym_payload = ym
                dt = time.perf_counter() - t0
                timings["t_moss"] += dt
                timings["moss_chunks_ms"].append(dt * 1000.0)

                up_hop = sr_buf[sr_emitted_pos:end].copy()
                q_ane.put((ym_payload, up_hop, sr_emitted_pos - win_start, this_hop))
                sr_emitted_pos = end

                # Trim old samples beyond context window
                keep_from = max(0, sr_emitted_pos - ctx_s)
                if keep_from > 0:
                    sr_buf = sr_buf[keep_from:]
                    sr_emitted_pos -= keep_from

        # --- MAIN STREAMING LOOP (TRUE WALL-CLOCK CONCURRENT!) ---
        inw_stream = self.inworld.stream(block_frames=self.inworld_bf, cache_frames=self.inworld_cf)
        down_resampler = soxr.ResampleStream(sr, 16000, 1, dtype="float32", quality="LQ")
        stage1a_parts: list[np.ndarray] = []

        t_wall0 = time.perf_counter()
        for i in range(0, len(mix_48k), chunk_48):
            t0 = time.perf_counter()
            y48 = inw_stream.push(mix_48k[i : i + chunk_48])
            dt = time.perf_counter() - t0
            timings["t_inw"] += dt
            timings["inw_chunks_ms"].append(dt * 1000.0)

            if len(y48):
                stage1a_parts.append(y48)
                y16_in = down_resampler.resample_chunk(y48, last=False)
                if len(y16_in):
                    if self.voxlock_backend == "onnx_cpu":
                        q_vl_in.put(y16_in)
                        # Drain any ready VoxLock CPU outputs without blocking GPU!
                        while True:
                            try:
                                out16 = q_vl_out.get_nowait()
                                if out16 is not None:
                                    _feed_stage2(out16, is_last=False)
                            except queue.Empty:
                                break
                    else:
                        t_v0 = time.perf_counter()
                        y16_out = _vl_push_chain(y16_in)
                        dt_v = time.perf_counter() - t_v0
                        timings["t_vl"] += dt_v
                        if len(y16_out):
                            timings["vl_chunks_ms"].append(dt_v * 1000.0)
                            _feed_stage2(y16_out, is_last=False)

        # Flush Stage 1A
        t0 = time.perf_counter()
        y48_tail = inw_stream.flush()
        timings["t_inw"] += time.perf_counter() - t0
        if len(y48_tail):
            stage1a_parts.append(y48_tail)
            y16_tail = down_resampler.resample_chunk(y48_tail, last=True)
        else:
            y16_tail = down_resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)

        # Flush Stage 1B
        if self.voxlock_backend == "onnx_cpu":
            if len(y16_tail):
                q_vl_in.put(y16_tail)
            q_vl_in.put(None)
            while True:
                out16 = q_vl_out.get()
                if out16 is None:
                    break
                _feed_stage2(out16, is_last=False)
            cpu_vl_thread.join()
        else:
            t_v0 = time.perf_counter()
            if len(y16_tail):
                o1 = _vl_push_chain(y16_tail)
                if len(o1):
                    _feed_stage2(o1, is_last=False)
            o2 = _vl_flush_chain()
            timings["t_vl"] += time.perf_counter() - t_v0
            if len(o2):
                _feed_stage2(o2, is_last=False)

        # Flush Stage 2A & wait for ANE Stage 2B to finish!
        _feed_stage2(np.zeros(0, dtype=np.float32), is_last=True)
        q_ane.put(None)
        ane_thread.join()

        # Stage 3: Studio Master (vDSP_biquad 80 Hz HPF + -1 dBFS)
        restored_48 = np.ascontiguousarray(
            np.concatenate(ane_out_chunks) if ane_out_chunks else np.zeros(len(mix_48k), dtype=np.float32),
            dtype=np.float32,
        )
        sos_hp_m = np.ascontiguousarray(signal.butter(2, 80.0, btype="highpass", fs=sr, output="sos"), dtype=np.float64)
        if _NATIVE_BIQUAD is not None:
            bq_m = _NATIVE_BIQUAD.biquad_create(sos_hp_m.ctypes.data, sos_hp_m.shape[0])
            master_48 = np.empty_like(restored_48)
            _NATIVE_BIQUAD.biquad_process(bq_m, restored_48.ctypes.data, master_48.ctypes.data, len(restored_48))
            _NATIVE_BIQUAD.biquad_destroy(bq_m)
        else:
            master_48 = signal.sosfilt(sos_hp_m, restored_48).astype(np.float32)
        peak = float(np.max(np.abs(master_48)) + 1e-9)
        master_48 = (master_48 / peak) * 0.891

        wall_s = time.perf_counter() - t_wall0
        stage1a_48 = np.concatenate(stage1a_parts)
        stage1b_16 = np.concatenate(stage1b_parts)

        timings["wall_s"] = wall_s
        timings["mlx_mem_mb"] = mx.get_active_memory() / (1024 ** 2)
        timings["inw_p50_ms"] = float(np.percentile(timings["inw_chunks_ms"], 50)) if timings["inw_chunks_ms"] else 0.0
        timings["inw_p95_ms"] = float(np.percentile(timings["inw_chunks_ms"], 95)) if timings["inw_chunks_ms"] else 0.0
        timings["vl_p50_ms"] = float(np.percentile(timings["vl_chunks_ms"], 50)) if timings["vl_chunks_ms"] else 0.0
        timings["moss_p50_ms"] = float(np.percentile(timings["moss_chunks_ms"], 50)) if timings["moss_chunks_ms"] else 0.0
        timings["hifi_p50_ms"] = float(np.percentile(timings["hifi_chunks_ms"], 50)) if timings["hifi_chunks_ms"] else 0.0
        return stage1a_48, stage1b_16, master_48, timings
