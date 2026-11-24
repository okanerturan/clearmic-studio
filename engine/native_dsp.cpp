// Native Apple Silicon (Accelerate vDSP + ARM64 NEON) DSP library for
// real-time 48 kHz voice isolation, overlap-add synthesis, and studio crossover/mastering.
//
// Build:
//   clang++ -std=c++20 -O3 -mcpu=native -ffast-math -shared -fPIC \
//       -framework Accelerate native_dsp.cpp -o libnative_dsp.dylib

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {

struct InworldDSP {
    int n_fft;
    int hop;
    int bins;
    float bound;
    FFTSetup fft_setup;
    vDSP_Length log2n;
    std::vector<float> win;
    std::vector<float> win_sq;
    std::vector<float> tmp_real;
    std::vector<float> tmp_imag;
    std::vector<float> tmp_time;
};

InworldDSP* inworld_dsp_create(int n_fft, int hop, float bound) {
    auto* d = new InworldDSP();
    d->n_fft = n_fft;
    d->hop = hop;
    d->bins = n_fft / 2 + 1;
    d->bound = bound;
    d->log2n = (vDSP_Length)std::round(std::log2((double)n_fft));
    d->fft_setup = vDSP_create_fftsetup(d->log2n, kFFTRadix2);
    d->win.resize(n_fft);
    d->win_sq.resize(n_fft);
    const double two_pi_over_n = 2.0 * M_PI / (double)n_fft;
    for (int i = 0; i < n_fft; ++i) {
        float w = (float)(0.5 - 0.5 * std::cos(two_pi_over_n * i));
        d->win[i] = w;
        d->win_sq[i] = w * w;
    }
    d->tmp_real.resize(n_fft / 2);
    d->tmp_imag.resize(n_fft / 2);
    d->tmp_time.resize(n_fft);
    return d;
}

void inworld_dsp_destroy(InworldDSP* d) {
    if (!d) return;
    vDSP_destroy_fftsetup(d->fft_setup);
    delete d;
}

// Compute `count` frames of real-to-complex STFT using Apple vDSP_fft_zrip.
// `audio` has length `(count - 1) * hop + n_fft`.
// `spec_ri_out` has shape `[count, 2 * bins]` (interleaved real, imag float32).
void inworld_dsp_stft(
    InworldDSP* d,
    const float* audio,
    int count,
    float* spec_ri_out
) {
    const int N = d->n_fft;
    const int half = N / 2;
    const int hop = d->hop;
    const int bins = d->bins;
    DSPSplitComplex split{d->tmp_real.data(), d->tmp_imag.data()};

    for (int f = 0; f < count; ++f) {
        const float* seg = audio + f * hop;
        // Apply Hann window via vDSP SIMD multiply
        vDSP_vmul(seg, 1, d->win.data(), 1, d->tmp_time.data(), 1, N);
        // Pack even/odd into split complex
        vDSP_ctoz((const DSPComplex*)d->tmp_time.data(), 2, &split, 1, half);
        // Forward real FFT (Apple vDSP scales by 2.0!)
        vDSP_fft_zrip(d->fft_setup, &split, 1, d->log2n, kFFTDirection_Forward);

        float* out = spec_ri_out + f * (2 * bins);
        // DC (bin 0): real = 0.5 * realp[0], imag = 0
        out[0] = 0.5f * split.realp[0];
        out[1] = 0.0f;
        // Nyquist (bin N/2): real = 0.5 * imagp[0], imag = 0
        out[2 * half] = 0.5f * split.imagp[0];
        out[2 * half + 1] = 0.0f;

        // Bins 1 .. half-1 (interleaved, scaled by 0.5)
        const float32x4_t half_v = vdupq_n_f32(0.5f);
        int k = 1;
        for (; k + 4 <= half; k += 4) {
            float32x4_t vr = vmulq_f32(vld1q_f32(split.realp + k), half_v);
            float32x4_t vi = vmulq_f32(vld1q_f32(split.imagp + k), half_v);
            float32x4x2_t zip = {vr, vi};
            vst2q_f32(out + 2 * k, zip);
        }
        for (; k < half; ++k) {
            out[2 * k]     = 0.5f * split.realp[k];
            out[2 * k + 1] = 0.5f * split.imagp[k];
        }
    }
}

// Compute complex mask + bounded residual + iSTFT + overlap-add for `count` frames.
// `raw`: `[count, bins, 4]` float32 (mask_r, mask_i, res_r, res_i)
// `spec_ri`: `[count, 2 * bins]` float32
// `ola`, `env`: length at least `start + (count - 1) * hop + n_fft`
void inworld_dsp_corr_istft_ola(
    InworldDSP* d,
    const float* raw,
    const float* spec_ri,
    int count,
    float* ola,
    float* env,
    int start
) {
    const int N = d->n_fft;
    const int half = N / 2;
    const int hop = d->hop;
    const int bins = d->bins;
    const float bound = d->bound;
    const float inv_n = 1.0f / (float)N;
    DSPSplitComplex split{d->tmp_real.data(), d->tmp_imag.data()};

    for (int f = 0; f < count; ++f) {
        const float* r_ptr = raw + f * (bins * 4);
        const float* s_ptr = spec_ri + f * (2 * bins);

        auto calc_bin = [&](int k, float& dr, float& di) {
            const float sr = s_ptr[2 * k];
            const float si = s_ptr[2 * k + 1];
            const float mr = 2.0f * std::tanh(r_ptr[4 * k + 0]);
            const float mi = 2.0f * std::tanh(r_ptr[4 * k + 1]);
            const float rr = std::tanh(r_ptr[4 * k + 2]);
            const float ri = std::tanh(r_ptr[4 * k + 3]);
            const float nrm = std::fmax(std::sqrt(rr * rr + ri * ri), 1.0f);
            const float mag = std::sqrt(sr * sr + si * si);
            const float sc = (bound * mag) / nrm;
            dr = (sr * mr - si * mi) + rr * sc;
            di = (sr * mi + si * mr) + ri * sc;
        };

        float dr0, di0, drn, din;
        calc_bin(0, dr0, di0);
        calc_bin(half, drn, din);
        split.realp[0] = dr0;
        split.imagp[0] = drn;

        for (int k = 1; k < half; ++k) {
            float dr, di;
            calc_bin(k, dr, di);
            split.realp[k] = dr;
            split.imagp[k] = di;
        }

        // Inverse real FFT (Apple vDSP_fft_zrip inverse returns N * x, scaled by 1/N via vDSP)
        vDSP_fft_zrip(d->fft_setup, &split, 1, d->log2n, kFFTDirection_Inverse);
        vDSP_ztoc(&split, 1, (DSPComplex*)d->tmp_time.data(), 2, half);

        // Apply (window / N) and accumulate into ola + win_sq into env
        const float scale_n = inv_n;
        float* o_dst = ola + start + f * hop;
        float* e_dst = env + start + f * hop;
        const float* w_ptr = d->win.data();
        const float* w2_ptr = d->win_sq.data();
        const float* t_ptr = d->tmp_time.data();

        const float32x4_t sv = vdupq_n_f32(scale_n);
        int i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t tv = vmulq_f32(vld1q_f32(t_ptr + i), sv);
            float32x4_t wv = vld1q_f32(w_ptr + i);
            float32x4_t ov = vld1q_f32(o_dst + i);
            vst1q_f32(o_dst + i, vfmaq_f32(ov, tv, wv));

            float32x4_t ev = vld1q_f32(e_dst + i);
            vst1q_f32(e_dst + i, vaddq_f32(ev, vld1q_f32(w2_ptr + i)));
        }
        for (; i < N; ++i) {
            o_dst[i] += (t_ptr[i] * scale_n) * w_ptr[i];
            e_dst[i] += w2_ptr[i];
        }
    }
}

// Ultra-fast NEON overlap-add for VoxLock (NFFT = 512, HOP = 128)
void voxlock_ola_neon(
    const float* frames,  // [k, 512]
    int k,
    const float* win2,    // [512]
    float* ola,           // len >= (k-1)*128 + 512
    float* env
) {
    for (int j = 0; j < k; ++j) {
        const float* f_ptr = frames + j * 512;
        float* o_ptr = ola + j * 128;
        float* e_ptr = env + j * 128;
        for (int i = 0; i < 512; i += 4) {
            vst1q_f32(o_ptr + i, vaddq_f32(vld1q_f32(o_ptr + i), vld1q_f32(f_ptr + i)));
            vst1q_f32(e_ptr + i, vaddq_f32(vld1q_f32(e_ptr + i), vld1q_f32(win2 + i)));
        }
    }
}

// Streaming Apple Accelerate vDSP_biquad filter for causal SOS crossover & mastering HPF.
// `sos` is scipy SOS format: [n_sections, 6] = [b0, b1, b2, a0(=1), a1, a2] in double.
struct BiquadStream {
    vDSP_biquad_Setup setup;
    int n_sections;
    std::vector<float> delay;
};

BiquadStream* biquad_create(const double* sos, int n_sections) {
    auto* b = new BiquadStream();
    b->n_sections = n_sections;
    // vDSP_biquad_CreateSetup expects [b0, b1, b2, a1, a2] per section (5 doubles/section)
    std::vector<double> coeffs(n_sections * 5);
    for (int s = 0; s < n_sections; ++s) {
        double a0 = sos[s * 6 + 3];
        coeffs[s * 5 + 0] = sos[s * 6 + 0] / a0;
        coeffs[s * 5 + 1] = sos[s * 6 + 1] / a0;
        coeffs[s * 5 + 2] = sos[s * 6 + 2] / a0;
        coeffs[s * 5 + 3] = sos[s * 6 + 4] / a0;
        coeffs[s * 5 + 4] = sos[s * 6 + 5] / a0;
    }
    b->setup = vDSP_biquad_CreateSetup(coeffs.data(), (vDSP_Length)n_sections);
    b->delay.assign(2 * n_sections + 2, 0.0f);
    return b;
}

void biquad_process(BiquadStream* b, const float* x, float* y, int n) {
    vDSP_biquad(b->setup, b->delay.data(), x, 1, y, 1, (vDSP_Length)n);
}

void biquad_destroy(BiquadStream* b) {
    if (!b) return;
    vDSP_biquad_DestroySetup(b->setup);
    delete b;
}

// Fused NEON crossover sum + optional mastering peak normalize
void neon_add_f32(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

// Zero-latency causal target-ratio envelope gate for Real-48k Gated-Highs mode.
// Computes causal RMS envelopes of `target_band` (VoxLock) and `total_band` (Inworld),
// forms Wiener-like gate = clamp(env_t / (env_tot + eps), 0, 1)^gamma with fast attack
// and smooth release, and outputs `out[i] = low_target[i] + (hf_gain * gate) * high_total[i]`.
struct GatedHighsState {
    float env_t;
    float env_tot;
    float gate_smooth;
    float alpha_env;
    float alpha_atk;
    float alpha_rel;
    float gamma;
    float hf_gain;
};

GatedHighsState* gated_highs_create(
    float sr,
    float env_ms,
    float atk_ms,
    float rel_ms,
    float gamma,
    float hf_gain
) {
    auto* s = new GatedHighsState();
    s->env_t = 0.0f;
    s->env_tot = 0.0f;
    s->gate_smooth = 0.0f;
    s->alpha_env = std::exp(-1.0f / std::fmax(1.0f, sr * env_ms * 0.001f));
    s->alpha_atk = std::exp(-1.0f / std::fmax(1.0f, sr * atk_ms * 0.001f));
    s->alpha_rel = std::exp(-1.0f / std::fmax(1.0f, sr * rel_ms * 0.001f));
    s->gamma = gamma;
    s->hf_gain = hf_gain;
    return s;
}

void gated_highs_process(
    GatedHighsState* s,
    const float* low_target,
    const float* low_total,
    const float* high_total,
    float* out,
    int n
) {
    float et = s->env_t;
    float ea = s->env_tot;
    float g = s->gate_smooth;
    const float ae = s->alpha_env;
    const float one_ae = 1.0f - ae;
    const float aa = s->alpha_atk;
    const float ar = s->alpha_rel;
    const float gam = s->gamma;
    const float hfg = s->hf_gain;
    const float eps = 1e-7f;

    for (int i = 0; i < n; ++i) {
        const float lt = low_target[i];
        const float la = low_total[i];
        et = ae * et + one_ae * (lt * lt);
        ea = ae * ea + one_ae * (la * la);
        float rt = std::sqrt(et > 0.0f ? et : 0.0f);
        float ra = std::sqrt(ea > 0.0f ? ea : 0.0f) + eps;
        float r = rt / ra;
        if (r > 1.0f) r = 1.0f;
        if (r < 0.0f) r = 0.0f;
        // Fast attack / smooth release on the raw ratio, then Wiener exponent (NaN-safe under -ffast-math!)
        float raw_g = (r > 1e-5f) ? std::pow(r, gam) : 0.0f;
        float coef = (raw_g > g) ? aa : ar;
        g = coef * g + (1.0f - coef) * raw_g;
        out[i] = lt + (hfg * g) * high_total[i];
    }
    s->env_t = et;
    s->env_tot = ea;
    s->gate_smooth = g;
}

void gated_highs_destroy(GatedHighsState* s) {
    delete s;
}

} // extern "C"
