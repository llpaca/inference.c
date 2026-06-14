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
/*
 * Branchless integer multiply.
 * Implicit-1 bit is OR'd into the mantissa before multiply,
 * so mant is always in [8..15].  Product is in [64..225].
 */
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

/* ─────────────────────────── fp8_mul_precompt ──────────────────────── */
/*
 * Same result as fp8_mul, but the 8×8 mantissa product is read from a
 * 64-entry lookup table — one cache line on any modern CPU.
 * Slightly faster on scalar pipelines where multiply latency dominates.
 *
 * Index = (mantA << 3) | mantB   →  product including implicit-1 on both sides
 */
static const uint8_t mant_table[64] = {
/*        b=0   1    2    3    4    5    6    7          */
/* a=0 */   8,  16,  24,  32,  40,  48,  56,  64,
/* a=1 */  16,  18,  27,  36,  45,  54,  63,  72,
/* a=2 */  24,  27,  32,  48,  60,  72,  84,  96,
/* a=3 */  32,  36,  48,  64,  80,  96, 112, 128,
/* a=4 */  40,  45,  60,  80, 100, 120, 140, 160,
/* a=5 */  48,  54,  72,  96, 120, 144, 168, 192,
/* a=6 */  56,  63,  84, 112, 140, 168, 196, 224,
/* a=7 */  64,  72,  96, 128, 160, 192, 224, 255,
};

f8 fp8_mul_precompt(f8 a, f8 b) {
    int S = (a ^ b) & 0x80;
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;

    int idx  = ((a & 0x7) << 3) | (b & 0x7);
    int mant = mant_table[idx];

    int shift = mant >> 7;   /* 1 if mant ≥ 128 */
    mant >>= shift;
    E += shift;

    mant = (mant + 4) >> 3;
    int overflow = mant >> 4;
    mant &= 0x7;
    E += overflow;
    E = (E <= 0) ? 0 : ((E >= 15) ? 15 : E);

    return (f8)(S | (E << 3) | mant);
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