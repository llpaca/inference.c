#pragma once
/*
 * fp8.h  —  E4M3 FP8 type for Llama inference
 *
 * Layout: |S|E3|E2|E1|E0|M2|M1|M0|
 *   sign = bit 7
 *   exp  = bits 6:3   (4 bits, bias = 7)
 *   mant = bits 2:0   (3 bits, implicit leading 1 for normals)
 *
 * Special values (E4M3 convention):
 *   e=0,  m=0   → ±0
 *   e=0,  m≠0   → subnormal
 *   e=15, m=7   → NaN  (0x7F / 0xFF)
 *   e=15, m<7   → max finite ±448  (0x7E / 0xFE)
 *
 * Phase 1  (now):   weights stored as f8, dequant→f32 before BLAS calls
 * Phase 2  (TODO):  fp8_matmul_vec with AVX2/AVX-512 — all arithmetic in f8
 *                   target: 8× memory bandwidth, 4× compute vs f32
 *
 * SIMD roadmap (fp8_matmul.c, to be added):
 *   - AVX2:    _mm256_cvtepu8_epi32 + scaled integer dot
 *   - AVX-512: _mm512_cvtne2ps_pbh / VNNI tricks with uint8 accumulation
 *   - plan: pack 8 f8 weights per 64-bit lane, dequant inline in the inner loop
 */

#include <stdint.h>
#include <math.h>    /* isnan, NAN, ldexpf */
#include <string.h>  /* memcpy */

/* ── type alias ─────────────────────────────────────────────────────── */
typedef uint8_t f8;
typedef uint8_t fp8_t;   /* alias used in some older call sites */

/* ── constants ───────────────────────────────────────────────────────── */
#define FP8_E4M3_BIAS   7
#define FP8_NAN         ((f8)0x7Fu)   /* canonical NaN  (e=15, m=7) */
#define FP8_MAX         ((f8)0x7Eu)   /* +448            (e=15, m=6) */
#define FP8_ZERO        ((f8)0x00u)
#define FP8_NEG_ZERO    ((f8)0x80u)

/* ── core arithmetic ─────────────────────────────────────────────────── */

/* E4M3 multiply — branchless, no lookup */
f8    fp8_mul(f8 a, f8 b);

/* E4M3 multiply — lookup-table mantissa (faster on scalar pipelines) */
f8    fp8_mul_precompt(f8 a, f8 b);

/* E4M3 add — dequant → f32 add → requant
 * Used by dot-product loops until the SIMD path lands. */
static inline f8 fp8_add(f8 a, f8 b);   /* defined below */

/* ── conversion ──────────────────────────────────────────────────────── */
f8    fp32_to_fp8(float x);
float fp8_to_fp32(f8 v);

/* ── inline fp8_add (dequant path, phase-1 only) ─────────────────────
 * This will be replaced by an integer accumulator in the SIMD matmul.
 * For now it keeps all call-sites in fp8 types without needing a
 * separate float accumulator visible in the caller.
 */
static inline f8 fp8_add(f8 a, f8 b) {
    return fp32_to_fp8(fp8_to_fp32(a) + fp8_to_fp32(b));
}

/* ── quantisation helpers ────────────────────────────────────────────── */

/* Quantise a float array to fp8 in-place (per-tensor, max-abs scaling).
 * scale_out receives the f32 scale factor so the caller can dequant later.
 *
 * TODO (phase 2): per-channel / per-block scaling for better accuracy.
 */
static inline void fp32_buf_to_fp8(const float *src, f8 *dst, int n,
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

/* Dequantise an fp8 array back to float (use the scale returned above) */
static inline void fp8_buf_to_fp32(const f8 *src, float *dst, int n,
                                    float scale) {
    for (int i = 0; i < n; i++)
        dst[i] = fp8_to_fp32(src[i]) * scale;
}