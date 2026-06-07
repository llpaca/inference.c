///@file: fp8.c
#include <stdio.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "fp8.h"
/**
@note:
    E4M3 has small range and good precision, so we will use this for activation
    -> |1bit sign|4bit exponent|3bit mantisa|
        sign = (fp8 >> 7) & 1
        exp  = (fp8 >> 3) & 0xF
        mant = fp8 & 0x7

    E5M2 has large range but low precision, this can be used in weights
    -> |1bit sign|5bit exponent|2bit mantisa|
*/

/*
branchless version
static const int mant_lookup[8] = {8,9,10,11,12,13,14,15};
int M = mant_lookup[a & 0x7] * mant_lookup[b & 0x7];
*/
/// @brief branchless version
/// @param a 
/// @param b 
/// @return fp8
f8 fp8_mul(f8 a, f8 b){
    int S = (a ^ b) & 0x80; //0x80 in binary is 10000000.
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;
    int M = ((a & 0x7) | 8) * ((b & 0x7) | 8); // 8 == 1<<3

    int shift = M >> 7;   // 1 if M >= 128
    M >>= shift;           
    E += shift;

    int mant = (M + 4) >> 3;
    // Handle overflow mantissa
    int overflow = mant >> 4;   // 1 if mant >= 16
    mant &= 0xF;                // clamp to 0..15
    E += overflow;
    E = (E <= 0) ? 0 : ((E >= 15) ? 15 : E);
    return S | (E << 3) | (mant & 0x7);
}

// Precompute all possible mantissa products with implicit 1 bit
// Index = (mantA << 3) | mantB = 0..7 << 3 | 0..7 => 0..63
static const uint8_t mant_table[64] = {
    8, 16, 24, 32, 40, 48, 56, 64,   // mantA=0
    16, 18, 27, 36, 45, 54, 63, 72,  // mantA=1
    24, 27, 32, 48, 60, 72, 84, 96,  // mantA=2
    32, 36, 48, 64, 80, 96,112,128,  // mantA=3
    40, 45, 60, 80,100,120,140,160,  // mantA=4
    48, 54, 72, 96,120,144,168,192,  // mantA=5
    56, 63, 84,112,140,168,196,224,  // mantA=6
    64, 72, 96,128,160,192,224,255   // mantA=7
};

f8 fp8_mul_precompt(f8 a, f8 b){
    // Sign
    int S = (a ^ b) & 0x80;

    // Exponent sum minus bias
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;

    // Mantissa product via lookup table
    int idx = ((a & 0x7) << 3) | (b & 0x7);
    int mant = mant_table[idx];

    // Normalize mantissa overflow
    int shift = mant >> 7; // 1 if mant>=128
    mant >>= shift;
    E += shift;

    // Round and clamp mantissa
    mant = (mant + 4) >> 3;
    int overflow = mant >> 4;
    mant &= 0x7;
    E += overflow;

    // Clamp exponent
    E = (E <= 0) ? 0 : ((E >= 15) ? 15 : E);

    return S | (E << 3) | mant;
}

f8 fp32_to_fp8(float x) {
    if (isnan(x))  return FP8_NAN;

    uint32_t bits;
    memcpy(&bits, &x, 4);

    uint32_t sign = (bits >> 31) & 1u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127;  /* unbiased FP32 */
    uint32_t mant = bits & 0x7FFFFFu;

    /* clamp overflow to max finite */
    if (exp > 8) {   /* > 448 in magnitude */
        return (fp8_t)((sign << 7) | 0x7E);  /* ±448 */
    }

    /* flush underflow (below min subnormal) to zero */
    if (exp < -9) return (fp8_t)(sign << 7);

    /* ── subnormal result ─── */
    if (exp < -6) {
        /* shift mantissa right, implicit leading 1 included */
        int shift = -6 - exp;          /* how far below normal boundary       */
        uint32_t full_mant = (1u << 23) | mant;
        uint32_t m8 = (full_mant >> (20 + shift));  /* keep 3 + guard bits   */
        uint32_t round_bit = (full_mant >> (19 + shift)) & 1u;
        uint32_t sticky    = (full_mant  & ((1u << (19 + shift)) - 1u)) != 0;
        m8 += (round_bit & (m8 & 1u)) | (round_bit & sticky);  /* RNE        */
        if (m8 > 7u) m8 = 7u;  /* subnormal overflow → clamp                 */
        return (fp8_t)((sign << 7) | m8);
    }

    /* ── normal result ─── */
    int32_t e8 = exp + FP8_E4M3_BIAS;  /* re-bias for E4M3                   */
    uint32_t m3     = mant >> 20;      /* top 3 bits of FP32 mantissa         */
    uint32_t g_bit  = (mant >> 19) & 1u;
    uint32_t sticky = (mant & 0x7FFFFu) != 0;

    /* round-to-nearest-even */
    uint32_t round_up = g_bit & ((m3 & 1u) | sticky);
    m3 += round_up;

    if (m3 > 7u) {           /* mantissa overflow → increment exponent        */
        m3 = 0u;
        e8++;
        if (e8 >= 15) {      /* NaN territory in E4M3 — clamp to max         */
            return (fp8_t)((sign << 7) | 0x7E);
        }
    }

    return (fp8_t)((sign << 7) | ((uint32_t)e8 << 3) | m3);
}

float fp8_to_fp32(f8 v) {
    if ((v & 0x7Fu) == 0x7Fu) return NAN;  /* E4M3 NaN                        */

    uint32_t sign = (v >> 7) & 1u;
    uint32_t e8   = (v >> 3) & 0xFu;
    uint32_t m3   =  v       & 0x7u;

    float result;

    if (e8 == 0u) {
        /* subnormal: value = (-1)^s * 2^(1-bias) * (m3/8) */
        result = (float)m3 * (1.0f / 512.0f);   /* 2^(-9) * m3               */
    } else {
        /* normal: (-1)^s * 2^(e8-bias) * (1 + m3/8) */
        float exp_val = ldexpf(1.0f, (int)e8 - FP8_E4M3_BIAS);
        result = exp_val * (1.0f + (float)m3 / 8.0f);
    }
    return sign ? -result : result;
}