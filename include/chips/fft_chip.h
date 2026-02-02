/*
 * FFT_CHIP — Frozen Fast Fourier Transform Primitive
 *
 * Why FFT in a neural network engine?
 *
 * CfC processes TIME SERIES. Sensor data arrives as raw samples.
 * Before the CfC cell sees it, you often need spectral features:
 *   - Vibration analysis: dominant frequencies for predictive maintenance
 *   - Audio: mel-frequency bands for keyword detection
 *   - IMU: periodic motion detection for gesture recognition
 *   - ECG/EEG: frequency-band power for health monitoring
 *
 * Pipeline: Sensor → FFT_CHIP → |magnitude|² → CfC_CELL → Classification
 *
 * This chip implements radix-2 Cooley-Tukey DIT FFT.
 * Power-of-2 lengths only. In-place. No memory allocation.
 *
 * For a 256-point FFT on M4 @ 180MHz: ~50us.
 * That's 5x faster than the CfC cell itself at hidden_dim=32.
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_FFT_CHIP_H
#define TRIX_FFT_CHIP_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * FFT_CHIP — In-place radix-2 DIT FFT
 *
 * Input:  real[N], imag[N]  (interleaved or separate arrays)
 * Output: real[N], imag[N]  (overwritten in-place)
 * N must be a power of 2.
 *
 * For real-only input, pass imag[] initialized to zeros.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * FFT_CHIP — Forward FFT, in-place, radix-2 Cooley-Tukey
 *
 * @param real   Real part [N] (modified in-place)
 * @param imag   Imaginary part [N] (modified in-place)
 * @param N      Length (must be power of 2)
 */
static inline void FFT_CHIP(float* real, float* imag, int N) {
    /* Bit-reversal permutation */
    int j = 0;
    for (int i = 0; i < N - 1; i++) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int k = N >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Butterfly stages */
    for (int stage = 1; stage < N; stage <<= 1) {
        float angle = -3.14159265358979323846f / (float)stage;
        float wr = cosf(angle);
        float wi = sinf(angle);

        for (int group = 0; group < N; group += stage << 1) {
            float twr = 1.0f;
            float twi = 0.0f;

            for (int pair = 0; pair < stage; pair++) {
                int a = group + pair;
                int b = a + stage;

                float tr = twr * real[b] - twi * imag[b];
                float ti = twr * imag[b] + twi * real[b];

                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;

                float new_twr = twr * wr - twi * wi;
                twi = twr * wi + twi * wr;
                twr = new_twr;
            }
        }
    }
}

/**
 * IFFT_CHIP — Inverse FFT, in-place
 *
 * Same as forward but conjugate twiddles and scale by 1/N.
 */
static inline void IFFT_CHIP(float* real, float* imag, int N) {
    /* Conjugate input */
    for (int i = 0; i < N; i++) imag[i] = -imag[i];

    /* Forward FFT */
    FFT_CHIP(real, imag, N);

    /* Conjugate and scale */
    float inv_n = 1.0f / (float)N;
    for (int i = 0; i < N; i++) {
        real[i] *= inv_n;
        imag[i] = -imag[i] * inv_n;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SPECTRAL FEATURE EXTRACTION
 *
 * These convert FFT output into features suitable for CfC input.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * FFT_MAGNITUDE_CHIP — |X[k]|² (power spectrum)
 *
 * @param real   FFT real output [N]
 * @param imag   FFT imag output [N]
 * @param mag    Output: magnitude squared [N/2+1] (one-sided)
 * @param N      FFT length
 *
 * Only computes N/2+1 bins (Nyquist symmetry for real input).
 */
static inline void FFT_MAGNITUDE_CHIP(
    const float* real, const float* imag,
    float* mag, int N
) {
    int half = N / 2 + 1;
    for (int i = 0; i < half; i++) {
        mag[i] = real[i] * real[i] + imag[i] * imag[i];
    }
}

/**
 * FFT_LOG_MAGNITUDE_CHIP — 10*log10(|X[k]|²) in dB
 *
 * Useful for audio/vibration features where dynamic range matters.
 */
static inline void FFT_LOG_MAGNITUDE_CHIP(
    const float* real, const float* imag,
    float* mag_db, int N
) {
    int half = N / 2 + 1;
    float floor_val = 1e-10f;
    for (int i = 0; i < half; i++) {
        float power = real[i] * real[i] + imag[i] * imag[i];
        if (power < floor_val) power = floor_val;
        mag_db[i] = 10.0f * log10f(power);
    }
}

/**
 * FFT_BAND_ENERGY_CHIP — Energy in frequency bands
 *
 * Sums power spectrum into coarse bands. Reduces N/2+1 FFT bins
 * down to num_bands features suitable as CfC input.
 *
 * Example: 256-point FFT at 1kHz → 4 bands = [0-125Hz, 125-250Hz, 250-375Hz, 375-500Hz]
 *
 * @param mag        Power spectrum [N/2+1]
 * @param N          FFT length
 * @param bands      Output: band energies [num_bands]
 * @param num_bands  Number of output bands
 */
static inline void FFT_BAND_ENERGY_CHIP(
    const float* mag, int N,
    float* bands, int num_bands
) {
    int half = N / 2 + 1;
    int bins_per_band = half / num_bands;
    if (bins_per_band < 1) bins_per_band = 1;

    for (int b = 0; b < num_bands; b++) {
        float sum = 0.0f;
        int start = b * bins_per_band;
        int end = start + bins_per_band;
        if (b == num_bands - 1) end = half; /* Last band gets remainder */
        for (int i = start; i < end; i++) {
            sum += mag[i];
        }
        bands[b] = sum;
    }
}

/**
 * FFT_DOMINANT_FREQ_CHIP — Find the dominant frequency bin
 *
 * Returns the index of the highest-energy bin (excluding DC).
 * Multiply by (sample_rate / N) to get frequency in Hz.
 */
static inline int FFT_DOMINANT_FREQ_CHIP(const float* mag, int N) {
    int half = N / 2 + 1;
    int max_idx = 1; /* Skip DC (bin 0) */
    float max_val = mag[1];
    for (int i = 2; i < half; i++) {
        if (mag[i] > max_val) {
            max_val = mag[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * WINDOWING
 *
 * Apply before FFT to reduce spectral leakage.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * FFT_HANN_WINDOW_CHIP — Apply Hann window in-place
 */
static inline void FFT_HANN_WINDOW_CHIP(float* x, int N) {
    for (int i = 0; i < N; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * (float)i / (float)(N - 1)));
        x[i] *= w;
    }
}

/**
 * FFT_HAMMING_WINDOW_CHIP — Apply Hamming window in-place
 */
static inline void FFT_HAMMING_WINDOW_CHIP(float* x, int N) {
    for (int i = 0; i < N; i++) {
        float w = 0.54f - 0.46f * cosf(2.0f * 3.14159265358979323846f * (float)i / (float)(N - 1));
        x[i] *= w;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_FFT_CHIP_H */
