"""ClearMic Studio — Real-Time Apple M5 All-Silicon Audio Daemon.

Streams 48 kHz mono float32 PCM over a local Unix domain socket with ~74-95 ms
end-to-end latency, zero PyTorch, and zero coremltools:
  - Inworld Realtime MLX (BF16 backbone + Q8 decoder + Q8 KV + float4 Metal scan + vDSP FFT)
  - VoxLock MLX (1024-thread char4 INT8 Metal LSTM + STFT-Fused 1/2/3-Pass @ constant 24 ms delay)
  - Native C++ libnative_dsp.dylib (vDSP_biquad 7.2 kHz crossover + causal target-ratio gated
    real 48 kHz highs + 0 ms Broadcast Studio EQ + causal soft-knee vocal limiter)

Protocol:
  - Control: newline-delimited JSON on stdin (commands) and stdout (events/metrics).
  - Audio:   Unix domain socket (path passed via --socket). Each message is:
               uint32_le n_samples + float32[n_samples]
             Daemon reads one input chunk and immediately writes one output chunk of
             the exact same length `n_samples` (zero drift, sample-accurate loopback!).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import socket
import struct
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

STUDIO_DIR = pathlib.Path(__file__).resolve().parent
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import mlx.core as mx
import numpy as np
from scipy import signal
import soundfile as sf
import soxr

import inworld_mlx
import low_latency_48k as ll
import voxlock_mlx


PRESETS = {
    "studio_call": {
        "label": "Studio Call (~95 ms)",
        "lookahead": 2,
        "block_frames": 2,
        "passes": 3,
        "use_voxlock": True,
        "algo_ms": 85.3,
    },
    "ultra_low": {
        "label": "Ultra-Low (~74 ms)",
        "lookahead": 0,
        "block_frames": 2,
        "passes": 2,
        "use_voxlock": True,
        "algo_ms": 65.3,
    },
    "natural": {
        "label": "Natural 1-Pass (~95 ms)",
        "lookahead": 2,
        "block_frames": 2,
        "passes": 1,
        "use_voxlock": True,
        "algo_ms": 85.3,
    },
    "denoise_only": {
        "label": "Denoise Only (~62 ms)",
        "lookahead": 2,
        "block_frames": 2,
        "passes": 0,
        "use_voxlock": False,
        "algo_ms": 61.3,
    },
}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class LiveBroadcastProcessor:
    """Stateful real-time 48 kHz mono chunk-in / chunk-out processor."""

    def __init__(self, enroll_path: pathlib.Path):
        self.sr = 48000
        self.lock = threading.Lock()
        self.preset_id = "studio_call"
        self.studio_eq = True
        self.bypass = False
        self.gate_gamma = 1.6
        self.output_gain = 1.0
        self.enroll_path = enroll_path

        emit({"event": "loading", "message": "Loading Inworld 48 kHz MLX (la=2 & la=0)..."})
        self.inw_la2 = inworld_mlx.MLXVoiceIsolator(
            precision="hybrid", lookahead_frames=2, block_frames=2, cache_frames=576, kv_bits=8
        )
        self.inw_la0 = inworld_mlx.MLXVoiceIsolator(
            precision="hybrid", lookahead_frames=0, block_frames=2, cache_frames=576, kv_bits=8
        )

        emit({"event": "loading", "message": "Loading VoxLock Metal INT8 LSTM..."})
        self.voxlock = voxlock_mlx.VoxLockMLX(precision="hybrid")

        # Load initial enrollment audio (or fallback silent buffer until user enrolls)
        self.enroll_16k = self._load_enroll(enroll_path)

        emit({"event": "loading", "message": "Warming up M5 Metal kernels..."})
        self._warmup_all()

        # Streaming state (initialized in _reset_streams)
        self._dsp_objs = []
        self._reset_streams()
        emit({
            "event": "ready",
            "preset": self.preset_id,
            "has_enrollment": self.enroll_path.is_file(),
            "enroll_path": str(self.enroll_path),
            "mlx_mem_mb": round(mx.get_active_memory() / (1024 ** 2), 1),
        })

    def _load_enroll(self, path: pathlib.Path) -> np.ndarray:
        if path.is_file():
            x, sr = sf.read(str(path), dtype="float32")
            if x.ndim == 2:
                x = x.mean(axis=1)
            if sr != 16000:
                x = soxr.resample(x, sr, 16000).astype(np.float32)
            r = float(np.sqrt(np.mean(x * x) + 1e-12))
            if r > 1e-5:
                x = np.clip(x * (0.10 / r), -0.95, 0.95).astype(np.float32)
            return x
        return np.zeros(16000 * 4, dtype=np.float32)

    def enroll_from_wav(self, wav_48k_path: pathlib.Path, save_to: pathlib.Path, label: str | None = None) -> dict:
        """Cleans a newly recorded or dropped 48 kHz voice sample with Inworld and updates enrollment."""
        x, sr = sf.read(str(wav_48k_path), dtype="float32")
        if x.ndim == 2:
            x = x.mean(axis=1)
        if len(x) < max(int(sr * 0.25), 1600):
            raise ValueError("Audio clip is too short to build a voice print.")
        if sr != 48000:
            x = soxr.resample(x, sr, 48000).astype(np.float32)

        # For long Voice Memos (>20s), pick the highest-energy 15-second contiguous speech window
        # so enrollment stays fast (~1.5s) and skips leading silence.
        max_samples = 48000 * 20
        win_samples = 48000 * 15
        if len(x) > max_samples:
            hop = 48000
            best_i, best_e = 0, -1.0
            for i in range(0, len(x) - win_samples + 1, hop):
                seg = x[i : i + win_samples]
                e = float(np.mean(seg * seg))
                if e > best_e:
                    best_e = e
                    best_i = i
            x = x[best_i : best_i + win_samples]

        r = float(np.sqrt(np.mean(x * x) + 1e-12))
        if r > 1e-5:
            x = np.clip(x * (0.10 / r), -0.92, 0.92).astype(np.float32)
        with self.lock:
            clean_48 = self.inw_la2.isolate(x, block_frames=4)
            enr_16 = soxr.resample(clean_48, 48000, 16000).astype(np.float32)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_to), enr_16, 16000)
            self.enroll_path = save_to
            self.enroll_16k = enr_16
            self._reset_streams()
        res = {"ok": True, "duration_s": round(len(enr_16) / 16000.0, 2), "path": str(save_to)}
        if label:
            res["label"] = label
        return res

    def _warmup_all(self) -> None:
        dummy48 = np.zeros(480 * 2 * 6, dtype=np.float32)
        for iso in (self.inw_la2, self.inw_la0):
            s = iso.stream(block_frames=2, cache_frames=576)
            for i in range(0, len(dummy48), 960):
                _ = s.push(dummy48[i : i + 960])
            _ = s.flush()
        dummy16 = np.zeros(320 * 6, dtype=np.float32)
        for passes in (1, 2, 3):
            vs = self.voxlock.stream(self.enroll_16k, passes=passes)
            for i in range(0, len(dummy16), 320):
                _ = vs.push(dummy16[i : i + 320])
            _ = vs.flush()

    def _close_dsp(self) -> None:
        for obj in self._dsp_objs:
            if isinstance(obj, ll._Biquad):
                obj.close()
            else:
                ll._DSP.gated_highs_destroy(obj)
        self._dsp_objs = []

    def _reset_streams(self) -> None:
        self._close_dsp()
        cfg = PRESETS[self.preset_id]
        iso = self.inw_la2 if cfg["lookahead"] == 2 else self.inw_la0
        self.inw_stream = iso.stream(block_frames=cfg["block_frames"], cache_frames=576)
        self.use_voxlock = cfg["use_voxlock"]
        self.vl_stream = (
            self.voxlock.stream(self.enroll_16k, passes=cfg["passes"]) if self.use_voxlock else None
        )
        self.down_rs = soxr.ResampleStream(self.sr, 16000, 1, dtype="float32", quality="LQ")
        self.up_rs = soxr.ResampleStream(16000, self.sr, 1, dtype="float32", quality="LQ")

        sr = float(self.sr)
        self.bq_lp_t = ll._Biquad(signal.butter(4, 7200.0, btype="lowpass", fs=sr, output="sos"))
        self.bq_bp_t = ll._Biquad(signal.butter(2, [2000.0, 7200.0], btype="bandpass", fs=sr, output="sos"))
        self.bq_bp_a = ll._Biquad(signal.butter(2, [2000.0, 7200.0], btype="bandpass", fs=sr, output="sos"))
        self.bq_hp_a = ll._Biquad(signal.butter(4, 7200.0, btype="highpass", fs=sr, output="sos"))

        sos_master = signal.butter(2, 80.0, btype="highpass", fs=sr, output="sos")
        if self.studio_eq:
            sos_master = np.vstack([
                sos_master,
                ll._peaking_sos(140.0, +2.0, 0.85, sr),
                ll._peaking_sos(340.0, -2.5, 1.25, sr),
                ll._peaking_sos(3800.0, +2.2, 0.95, sr),
                ll._highshelf_sos(9500.0, +1.5, sr),
            ])
        self.bq_master = ll._Biquad(sos_master)
        self.gh = ll._DSP.gated_highs_create(sr, 6.0, 2.5, 35.0, float(self.gate_gamma), 1.0)
        self._dsp_objs = [self.bq_lp_t, self.bq_bp_t, self.bq_bp_a, self.bq_hp_a, self.bq_master, self.gh]

        self.clean48_fifo = np.zeros(0, dtype=np.float32)
        self.out_fifo = np.zeros(0, dtype=np.float32)
        self.in_agc_env = 0.08  # gentle causal input leveler for quiet mics
        self.comp_env = 0.08    # gentle causal broadcast compressor/limiter

    def configure(
        self,
        preset: str | None = None,
        studio_eq: bool | None = None,
        gate_gamma: float | None = None,
        output_gain: float | None = None,
        bypass: bool | None = None,
    ) -> None:
        with self.lock:
            need_reset = False
            if preset is not None and preset in PRESETS and preset != self.preset_id:
                self.preset_id = preset
                need_reset = True
            if studio_eq is not None and studio_eq != self.studio_eq:
                self.studio_eq = studio_eq
                need_reset = True
            if gate_gamma is not None and abs(gate_gamma - self.gate_gamma) > 1e-3:
                self.gate_gamma = float(np.clip(gate_gamma, 0.5, 3.0))
                need_reset = True
            if output_gain is not None:
                self.output_gain = float(np.clip(output_gain, 0.2, 3.0))
            if bypass is not None:
                self.bypass = bool(bypass)
            if need_reset:
                self._reset_streams()

    def process_chunk(self, x_in: np.ndarray) -> tuple[np.ndarray, dict]:
        """Processes one 48 kHz mono chunk (e.g. 960 samples = 20 ms) and returns same-length output."""
        n_req = len(x_in)
        in_rms = float(np.sqrt(np.mean(x_in * x_in) + 1e-12))

        with self.lock:
            if self.bypass:
                y_out = (x_in * self.output_gain).astype(np.float32)
                return y_out, {
                    "in_rms": in_rms,
                    "out_rms": float(np.sqrt(np.mean(y_out * y_out) + 1e-12)),
                    "compute_ms": 0.1,
                    "algo_ms": 0.0,
                    "e2e_ms": 20.0,
                    "bypass": True,
                }

            t0 = time.perf_counter()

            # Gentle causal input AGC (only boosts quiet mics when voice is active, never boosts silence)
            if in_rms > 0.008:
                self.in_agc_env = 0.94 * self.in_agc_env + 0.06 * in_rms
            in_scale = float(np.clip(0.10 / max(self.in_agc_env, 0.015), 0.5, 4.0))
            x_scaled = np.clip(x_in * in_scale, -0.95, 0.95).astype(np.float32)

            # Stage 1A: Inworld Realtime MLX 48 kHz
            y48 = self.inw_stream.push(x_scaled)

            if len(y48):
                if not self.use_voxlock:
                    out_seg = self.bq_master.process(y48)
                    self.out_fifo = np.concatenate([self.out_fifo, out_seg])
                else:
                    self.clean48_fifo = np.concatenate([self.clean48_fifo, y48])
                    y16_in = self.down_rs.resample_chunk(y48, last=False)
                    if len(y16_in):
                        y16_out = self.vl_stream.push(y16_in)
                        if len(y16_out):
                            y48_t = self.up_rs.resample_chunk(y16_out, last=False)
                            n = len(y48_t)
                            if n > 0:
                                if len(self.clean48_fifo) < n:
                                    y48_a = np.pad(self.clean48_fifo, (0, n - len(self.clean48_fifo)))
                                    self.clean48_fifo = np.zeros(0, dtype=np.float32)
                                else:
                                    y48_a = self.clean48_fifo[:n]
                                    self.clean48_fifo = self.clean48_fifo[n:]

                                low_t = self.bq_lp_t.process(y48_t)
                                mid_t = self.bq_bp_t.process(y48_t)
                                mid_a = self.bq_bp_a.process(y48_a)
                                high_a = self.bq_hp_a.process(y48_a)
                                mixed = np.empty(n, dtype=np.float32)
                                ll._DSP.gated_highs_process(
                                    self.gh,
                                    mid_t.ctypes.data,
                                    mid_a.ctypes.data,
                                    high_a.ctypes.data,
                                    mixed.ctypes.data,
                                    n,
                                )
                                out_seg = self.bq_master.process(low_t + (mixed - mid_t))
                                self.out_fifo = np.concatenate([self.out_fifo, out_seg])

            # Emit exact `n_req` samples (pad leading startup latency with zeros)
            if len(self.out_fifo) >= n_req:
                y_raw = self.out_fifo[:n_req]
                self.out_fifo = self.out_fifo[n_req:]
            elif len(self.out_fifo) > 0:
                y_raw = np.pad(self.out_fifo, (n_req - len(self.out_fifo), 0))
                self.out_fifo = np.zeros(0, dtype=np.float32)
            else:
                y_raw = np.zeros(n_req, dtype=np.float32)

            # Gentle causal broadcast voice leveler + soft-knee limiter (-1 dBFS ceiling)
            y_rms = float(np.sqrt(np.mean(y_raw * y_raw) + 1e-12))
            if y_rms > 0.008:
                self.comp_env = 0.90 * self.comp_env + 0.10 * y_rms
            makeup = float(np.clip(0.14 / max(self.comp_env, 0.03), 0.7, 2.2)) * self.output_gain
            y_out = np.tanh((y_raw * makeup) / 0.89) * 0.89
            y_out = y_out.astype(np.float32)

            compute_ms = (time.perf_counter() - t0) * 1000.0
            algo_ms = PRESETS[self.preset_id]["algo_ms"]
            return y_out, {
                "in_rms": in_rms,
                "out_rms": float(np.sqrt(np.mean(y_out * y_out) + 1e-12)),
                "compute_ms": round(compute_ms, 2),
                "algo_ms": algo_ms,
                "e2e_ms": round(algo_ms + compute_ms, 1),
                "bypass": False,
            }


def main() -> None:
    import queue

    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True, help="Unix domain socket path for 48 kHz PCM audio")
    default_enroll = pathlib.Path.home() / "Library/Application Support/ClearMic Studio/voice_print_16k.wav"
    ap.add_argument(
        "--enroll",
        default=str(default_enroll),
        help="Default enrollment WAV path",
    )
    args = ap.parse_args()

    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    proc = LiveBroadcastProcessor(pathlib.Path(args.enroll))
    stop_ev = threading.Event()
    cmd_q: queue.Queue = queue.Queue()

    # Background stdin JSON reader -> feeds cmd_q so ALL MLX Metal ops stay strictly on Main Thread!
    def _stdin_loop():
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cmd_q.put(msg)
                if msg.get("cmd") == "quit":
                    stop_ev.set()
                    break
        except Exception:
            pass
        finally:
            stop_ev.set()

    stdin_th = threading.Thread(target=_stdin_loop, daemon=True)
    stdin_th.start()

    def _drain_cmds() -> bool:
        while True:
            try:
                msg = cmd_q.get_nowait()
            except queue.Empty:
                return not stop_ev.is_set()
            cmd = msg.get("cmd")
            if cmd == "configure":
                proc.configure(
                    preset=msg.get("preset"),
                    studio_eq=msg.get("studio_eq"),
                    gate_gamma=msg.get("gate_gamma"),
                    output_gain=msg.get("output_gain"),
                    bypass=msg.get("bypass"),
                )
                emit({"event": "configured", "preset": proc.preset_id, "studio_eq": proc.studio_eq, "bypass": proc.bypass})
            elif cmd == "enroll":
                wav_p = pathlib.Path(msg["wav_path"])
                save_p = pathlib.Path(msg.get("save_path", str(default_enroll)))
                try:
                    res = proc.enroll_from_wav(wav_p, save_p, label=msg.get("label"))
                    emit({"event": "enrolled", **res})
                except Exception as exc:
                    emit({"event": "enroll_error", "error": f"Voice enrollment failed: {exc}"})
            elif cmd == "reset":
                with proc.lock:
                    proc._reset_streams()
                emit({"event": "reset_ok"})
            elif cmd == "quit":
                stop_ev.set()
                return False

    # Audio socket server runs strictly on MAIN thread (100% MLX Metal stream thread-safe!)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    srv.settimeout(0.1)
    emit({"event": "socket_ready", "socket": sock_path})
    last_stat_t = 0.0
    try:
        while _drain_cmds():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(0.25)
            emit({"event": "audio_connected"})
            try:
                while _drain_cmds():
                    try:
                        hdr = recv_exact(conn, 4)
                    except socket.timeout:
                        continue
                    if hdr is None:
                        break
                    (n_samples,) = struct.unpack("<I", hdr)
                    if n_samples == 0 or n_samples > 48000:
                        break
                    payload = recv_exact(conn, n_samples * 4)
                    if payload is None:
                        break
                    x_in = np.frombuffer(payload, dtype=np.float32).copy()
                    y_out, st = proc.process_chunk(x_in)
                    conn.sendall(struct.pack("<I", len(y_out)) + y_out.tobytes())
                    now = time.perf_counter()
                    if now - last_stat_t >= 0.05:
                        last_stat_t = now
                        st["event"] = "stats"
                        emit(st)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            except Exception as exc:
                import traceback
                emit({"event": "audio_error", "error": f"{exc}\n{traceback.format_exc()}"})
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                emit({"event": "audio_disconnected"})
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        try:
            srv.close()
        except OSError:
            pass
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
