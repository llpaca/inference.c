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
f8    fp8_mul_precompt(f8 a, f8 b);
f8 fp8_add(f8 a, f8 b);   /* defined below */

/* ── conversion ──────────────────────────────────────────────────────── */
f8    fp32_to_fp8(float x);
float fp8_to_fp32(f8 v);

void fp32_buf_to_fp8(const float *src, f8 *dst, int n, float *scale_out);
void fp8_buf_to_fp32(const f8 *src, float *dst, int n, float scale);