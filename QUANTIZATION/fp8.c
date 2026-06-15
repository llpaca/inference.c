/// @file fp8.c  —  E4M3 FP8 arithmetic primitives
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fp8.h"

/*
 * Format recap (E4M3):
 *   E4M3 → small range / good precision  → activations
 *   E5M2 → large range / low  precision  → weights (future option)
 *
 *   |S|E3|E2|E1|E0|M2|M1|M0|
 *     sign = (fp8 >> 7) & 1
 *     exp  = (fp8 >> 3) & 0xF
 *     mant =  fp8       & 0x7
 */

/* ─────────────────────────── fp8_mul ───────────────────────────────── */
f8 fp8_mul(f8 a, f8 b) {
    int S = (a ^ b) & 0x80;                          /* result sign           */
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;/* exponent, rm one bias */
    int M = ((a & 0x7) | 8) * ((b & 0x7) | 8);      /* implicit-1 multiply   */

    int shift = M >> 7;   /* 1 if M ≥ 128 (result needs normalisation) */
    M >>= shift;
    E += shift;

    int mant = (M + 4) >> 3;   /* round to 3 mantissa bits */
    int overflow = mant >> 4;  /* 1 if mant spilled into bit 4 */
    mant &= 0xF;
    E += overflow;
    E = (E <= 0) ? 0 : ((E >= 15) ? 15 : E);

    return (f8)(S | (E << 3) | (mant & 0x7));
}

/* ─────────────────────────── fp32_to_fp8 ───────────────────────────── */
f8 fp32_to_fp8(float x) {
    if (isnan(x)) return FP8_NAN;

    uint32_t bits;
    memcpy(&bits, &x, 4);

    uint32_t sign = (bits >> 31) & 1u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127;   /* unbiased */
    uint32_t mant =  bits & 0x7FFFFFu;

    /* overflow → clamp to ±max-finite (±448) */
    if (exp > 8)  return (f8)((sign << 7) | 0x7Eu);

    /* underflow → flush to ±0 */
    if (exp < -9) return (f8)(sign << 7);

    /* ── subnormal result ───────────────────────────────────────────── */
    if (exp < -6) {
        int      shift    = -6 - exp;
        uint32_t full_m   = (1u << 23) | mant;
        uint32_t m8       = full_m >> (20 + shift);
        uint32_t round_b  = (full_m >> (19 + shift)) & 1u;
        uint32_t sticky   = (full_m  & ((1u << (19 + shift)) - 1u)) != 0u;
        m8 += (round_b & (m8 & 1u)) | (round_b & sticky);   /* RNE */
        if (m8 > 7u) m8 = 7u;
        return (f8)((sign << 7) | m8);
    }

    /* ── normal result ──────────────────────────────────────────────── */
    int32_t  e8     = exp + FP8_E4M3_BIAS;
    uint32_t m3     = mant >> 20;
    uint32_t g_bit  = (mant >> 19) & 1u;
    uint32_t sticky = (mant & 0x7FFFFu) != 0u;

    /* round-to-nearest-even */
    uint32_t round_up = g_bit & ((m3 & 1u) | sticky);
    m3 += round_up;

    if (m3 > 7u) {         /* mantissa overflow → carry into exponent  */
        m3 = 0u;
        e8++;
        if (e8 >= 15) return (f8)((sign << 7) | 0x7Eu);   /* clamp    */
    }

    return (f8)((sign << 7) | ((uint32_t)e8 << 3) | m3);
}

/* ─────────────────────────── fp8_to_fp32 ───────────────────────────── */
float fp8_to_fp32(f8 v) {
    /* NaN: e=15, m=7  (both sign variants) */
    if ((v & 0x7Fu) == 0x7Fu) return NAN;

    uint32_t sign = (v >> 7) & 1u;
    uint32_t e8   = (v >> 3) & 0xFu;
    uint32_t m3   =  v       & 0x7u;

    float result;
    if (e8 == 0u) {
        /* subnormal: value = (-1)^s × 2^(1-7) × (m3/8) = m3 × 2^(-9) */
        result = (float)m3 * (1.0f / 512.0f);
    } else {
        /* normal:    value = (-1)^s × 2^(e8-7) × (1 + m3/8) */
        result = ldexpf(1.0f, (int)e8 - FP8_E4M3_BIAS)
               * (1.0f + (float)m3 / 8.0f);
    }
    return sign ? -result : result;
}

f8 fp8_add(f8 a, f8 b) {
    return fp32_to_fp8(fp8_to_fp32(a) + fp8_to_fp32(b));
}

void fp32_buf_to_fp8(const float *src, f8 *dst, int n,
                                    float *scale_out) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    /* FP8 E4M3 max finite = 448 */
    float scale = (amax > 0.0f) ? (448.0f / amax) : 1.0f;
    if (scale_out) *scale_out = 1.0f / scale;   /* store dequant scale */
    for (int i = 0; i < n; i++)
        dst[i] = fp32_to_fp8(src[i] * scale);
}

void fp8_buf_to_fp32(const f8 *src, float *dst, int n,
                                    float scale) {
    for (int i = 0; i < n; i++)
        dst[i] = fp8_to_fp32(src[i]) * scale;
}