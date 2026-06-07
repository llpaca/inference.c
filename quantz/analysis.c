/*
 * fp8_analysis.c
 * ─────────────────────────────────────────────────────────────────────────────
 * FP8 (E4M3) deep-dive:
 *   1. Data-loss % visualisation vs FP32
 *   2. Optimal scaling factor for summing N FP8 values
 *   3. FP8 summation with scaling
 *   4. FP8 multiplication
 *   5. Optimisations: LUT, block-scaling, vectorisation hints
 *
 * Format used: E4M3FN  (4-bit exponent, 3-bit mantissa, no infinities)
 *   • bias = 7
 *   • max  = 448.0   (0b_0_1111_110)
 *   • min normal     = 2^(-6)  ≈ 0.015625
 *   • min subnormal  = 2^(-9)  ≈ 0.001953125
 *   • NaN            = 0b_0_1111_111  (and all sign variants)
 *
 * Build:
 *   gcc -O2 -march=native -o fp8_analysis fp8_analysis.c -lm
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* ══════════════════════════════════════════════════════════════════════════════
 * § 1  —  FP8 E4M3 CORE PRIMITIVES
 * ══════════════════════════════════════════════════════════════════════════════*/

typedef uint8_t fp8_t;   /* raw 8-bit storage                                 */

#define FP8_E4M3_BIAS       7
#define FP8_E4M3_MAX_VAL    448.0f
#define FP8_E4M3_MIN_NORM   (1.0f / 64.0f)    /* 2^(1-bias) = 2^-6            */
#define FP8_E4M3_MIN_SUB    (1.0f / 512.0f)   /* 2^(1-bias-mant) = 2^-9       */
#define FP8_NAN             0x7Fu              /* 0 11111 111 — the only NaN   */

/* ── fp32 → fp8 (round-to-nearest-even, clamp to ±448) ───────────────────── */
fp8_t fp32_to_fp8(float x) {
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

/* ── fp8 → fp32 ───────────────────────────────────────────────────────────── */
float fp8_to_fp32(fp8_t v) {
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

/* ══════════════════════════════════════════════════════════════════════════════
 * § 2  —  DATA LOSS ANALYSIS
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * For each exponent bucket of FP32, compute:
 *   absolute error  = |fp32_val - fp8_roundtrip(fp32_val)|
 *   relative error  = absolute / |fp32_val|   (= data-loss %)
 *
 * We probe ~1M uniformly distributed values in FP8's representable range.
 */
void analyse_data_loss(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           FP8 E4M3  —  DATA LOSS ANALYSIS vs FP32           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── per-exponent-bucket stats ── */
    typedef struct { double sum_rel; double max_rel; long count; } Bucket;
    Bucket bucket[19] = {0};  /* exponents: -9 .. +8  (18 buckets) + 1 guard  */
    const int EXP_OFFSET = 9; /* bucket index = exp + 9                       */

    const int SAMPLES = 1 << 20;
    double total_rel = 0.0, max_rel_global = 0.0;
    long   total_samples = 0;

    srand(42);

    for (int i = 0; i < SAMPLES; i++) {
        /* uniform random in [-448, +448] excluding tiny underflow range       */
        float x = ((float)rand() / (float)RAND_MAX) * 896.0f - 448.0f;
        if (fabsf(x) < FP8_E4M3_MIN_SUB) continue;

        fp8_t   q  = fp32_to_fp8(x);
        float   xq = fp8_to_fp32(q);

        double abs_err = fabs((double)x - (double)xq);
        double rel_err = abs_err / fabs((double)x);  /* 0.0 = perfect         */

        int exponent = (int)floorf(log2f(fabsf(x)));
        int b = exponent + EXP_OFFSET;
        if (b < 0) b = 0;
        if (b > 18) b = 18;

        bucket[b].sum_rel += rel_err;
        if (rel_err > bucket[b].max_rel) bucket[b].max_rel = rel_err;
        bucket[b].count++;

        total_rel += rel_err;
        if (rel_err > max_rel_global) max_rel_global = rel_err;
        total_samples++;
    }

    /* ── print bucket table ── */
    printf("  %-12s  %-12s  %-14s  %-14s  %-10s\n",
           "Exp range", "Repr range", "Avg loss %", "Max loss %", "Samples");
    printf("  %s\n", "─────────────────────────────────────────────────────────────────");

    for (int b = 0; b <= 17; b++) {
        int e = b - EXP_OFFSET;
        if (bucket[b].count == 0) continue;
        double avg_loss = (bucket[b].sum_rel / bucket[b].count) * 100.0;
        double max_loss = bucket[b].max_rel * 100.0;
        float  lo = ldexpf(1.0f, e);
        float  hi = ldexpf(1.0f, e + 1);
        printf("  2^%-3d..2^%-3d  %6.4f..%-6.3f  %12.4f%%  %12.4f%%  %10ld\n",
               e, e+1, lo, hi, avg_loss, max_loss, bucket[b].count);
    }

    printf("  %s\n", "─────────────────────────────────────────────────────────────────");
    printf("  %-12s  %-12s  %12.4f%%  %12.4f%%  %10ld\n",
           "OVERALL", "",
           (total_rel / total_samples) * 100.0,
           max_rel_global * 100.0,
           total_samples);

    /* ── show representable density bar chart ── */
    printf("\n  Representable-value density  (# FP8 codes per exponent band)\n");
    printf("  %-14s  %-5s  %s\n", "Exponent", "Count", "Bar");
    printf("  %s\n", "──────────────────────────────────────────────");
    /* Count unique fp8 values per exponent */
    int density[19] = {0};
    for (int c = 0; c < 256; c++) {
        fp8_t v = (fp8_t)c;
        if ((v & 0x7F) == 0x7F) continue;  /* skip NaN                       */
        float fv = fp8_to_fp32(v);
        if (!isfinite(fv) || fv == 0.0f) continue;
        int e = (int)floorf(log2f(fabsf(fv)));
        int b = e + EXP_OFFSET;
        if (b >= 0 && b <= 18) density[b]++;
    }
    for (int b = 0; b <= 17; b++) {
        int e = b - EXP_OFFSET;
        if (density[b] == 0) continue;
        int bar = density[b] / 2;
        printf("  2^%3d          %3d    ", e, density[b]);
        for (int k = 0; k < bar; k++) putchar('█');
        putchar('\n');
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 * § 3  —  SCALING FACTOR FOR SUMMATION OF N FP8 VALUES
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *  Problem:  sum = x_0 + x_1 + ... + x_{N-1},  each x_i quantised to FP8.
 *
 *  Worst case (all values = +448):  sum = N × 448
 *  To keep the SUM representable in FP8 we need:
 *
 *      scale_factor  ≥  N              (integer ceil)
 *
 *  Strategy A — PRE-SCALE inputs (divide inputs by S before quantising):
 *      • Each x_i_scaled = x_i / S
 *      • sum_fp8 ≈  (1/S) × Σ x_i   →  stays in [-448, +448]
 *      • Final result = sum_fp8 × S  (dequantise after accumulation)
 *      • Best S = next power-of-2 ≥ N  (fast bit-shift dequant)
 *
 *  Strategy B — ACCUMULATE IN HIGHER PRECISION (float32 accumulator):
 *      • Zero quantisation overflow, but costs bandwidth.
 *      • Recommended for N > 128.
 *
 *  Strategy C — BLOCK SCALING (NVIDIA's approach in H100 Transformer Engine):
 *      • Divide tensor into blocks of B elements.
 *      • Compute per-block scale = max(|x|) / 448.
 *      • Quantise with that scale, store scale factor alongside.
 *      • Effective range ≫ raw FP8.
 *
 *  Theoretical loss increase with N (Strategy A, uniform input ∈ [0,1]):
 *      • Quantisation step size δ = 2^(-3) × 2^floor(log2(1/S))
 *      • Signal variance σ² ≈ 1/3
 *      • SQNR ≈ 6.02 × 3 + 1.76 - 20×log10(S)  dB
 * ══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int   N;
    int   S_pow2;          /* next power-of-2 ≥ N                              */
    float S_exact;         /* exact minimum scale = N                          */
    float expected_loss;   /* % loss relative to float32 sum, empirical        */
} ScaleInfo;

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

ScaleInfo compute_scale(int N) {
    ScaleInfo s;
    s.N       = N;
    s.S_exact = (float)N;
    s.S_pow2  = next_pow2(N);

    /* empirical: sum N uniform-random fp8 values, compare to fp32 sum        */
    srand(42);
    double total_err = 0.0;
    const int TRIALS = 4096;
    for (int t = 0; t < TRIALS; t++) {
        float fp32_sum = 0.0f;
        float fp8_sum  = 0.0f;
        float inv_S    = 1.0f / (float)s.S_pow2;
        for (int i = 0; i < N; i++) {
            float v  = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            fp32_sum += v;
            /* pre-scale → quantise → accumulate (still in scaled space)      */
            float vs = v * inv_S;
            fp8_sum += fp8_to_fp32(fp32_to_fp8(vs));
        }
        float fp8_result = fp8_sum * (float)s.S_pow2;  /* dequantise          */
        total_err += fabs((double)(fp8_result - fp32_sum)) /
                     (fabs((double)fp32_sum) + 1e-9);
    }
    s.expected_loss = (float)(total_err / TRIALS * 100.0);
    return s;
}

void analyse_scaling(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     SCALING FACTOR FOR SUMMING N FP8 VALUES (E4M3)          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  FP8 E4M3 max = 448.0,  min subnormal ≈ 0.00195\n\n");

    printf("  %-8s  %-10s  %-10s  %-14s  %s\n",
           "N", "S_exact", "S_pow2", "Exp.loss (%)", "Recommendation");
    printf("  %s\n", "──────────────────────────────────────────────────────────────────");

    int ns[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384};
    for (int i = 0; i < (int)(sizeof(ns)/sizeof(ns[0])); i++) {
        ScaleInfo s = compute_scale(ns[i]);
        const char *rec = (ns[i] <= 16)  ? "Pre-scale FP8"   :
                          (ns[i] <= 256) ? "Block-scale (B=64)" :
                                           "FP32 accumulator";
        printf("  %-8d  %-10.0f  %-10d  %-14.4f  %s\n",
               s.N, s.S_exact, s.S_pow2, s.expected_loss, rec);
    }

    printf("\n  Formula:\n");
    printf("    S_min  = N                    (minimum scale to prevent overflow)\n");
    printf("    S_best = next_power_of_2(N)   (free dequant via bit-shift)\n");
    printf("    x_scaled[i]  = fp32_to_fp8(x[i] / S_best)\n");
    printf("    result       = fp8_sum(x_scaled) * S_best\n");
}

/* ══════════════════════════════════════════════════════════════════════════════
 * § 4  —  FP8 SUMMATION  (three variants)
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * Variant A — Naive: convert each to fp32, accumulate, re-quantise.
 *   Error model: N × ε_quant  where ε_quant ≈ 2^(-3) × 2^(exp of element)
 */
fp8_t fp8_sum_naive(const fp8_t *a, int N) {
    float acc = 0.0f;
    for (int i = 0; i < N; i++)
        acc += fp8_to_fp32(a[i]);
    return fp32_to_fp8(acc);
}

/*
 * Variant B — Pre-scaled: caller supplies scale S (power of 2).
 *   1. Each input already stored pre-scaled (x / S).
 *   2. Accumulate in fp32 (no overflow possible since each |x/S| ≤ 448/S).
 *   3. Return fp8 result in scaled space; caller multiplies by S.
 *   This is what you'd use in inference kernels (similar to INT8 gemm scaling).
 */
float fp8_sum_prescaled(const fp8_t *a, int N, int S_pow2) {
    float acc = 0.0f;
    for (int i = 0; i < N; i++)
        acc += fp8_to_fp32(a[i]);
    return acc * (float)S_pow2;   /* dequantise: returns fp32 result           */
}

/*
 * Variant C — LUT-accelerated block sum (fastest on CPU without SIMD).
 *   Precompute fp8→fp32 table once. Inner loop becomes table lookup + add.
 *   This is exactly how Transformer Engine / cuDNN do it on non-FP8 hardware.
 *
 *   Memory: 256 × 4 bytes = 1 KB  →  fits in L1D every time.
 */
static float FP8_LUT[256];
static int   lut_initialised = 0;

void fp8_init_lut(void) {
    if (lut_initialised) return;
    for (int i = 0; i < 256; i++)
        FP8_LUT[i] = fp8_to_fp32((fp8_t)i);
    lut_initialised = 1;
}

float fp8_sum_lut(const fp8_t *a, int N) {
    fp8_init_lut();
    float acc = 0.0f;
    /* unroll 8× — compiler will likely auto-vectorise with -O2 -march=native */
    int i = 0;
    for (; i <= N - 8; i += 8) {
        acc += FP8_LUT[a[i+0]];
        acc += FP8_LUT[a[i+1]];
        acc += FP8_LUT[a[i+2]];
        acc += FP8_LUT[a[i+3]];
        acc += FP8_LUT[a[i+4]];
        acc += FP8_LUT[a[i+5]];
        acc += FP8_LUT[a[i+6]];
        acc += FP8_LUT[a[i+7]];
    }
    for (; i < N; i++) acc += FP8_LUT[a[i]];
    return acc;
}

/*
 * Variant D — Pairwise (tree) summation in fp32 (Kahan-like error bound).
 *   Standard pairwise reduces rounding error from O(N·ε) to O(log2(N)·ε).
 *   Works in-place on a scratch buffer.
 */
float fp8_sum_pairwise(const fp8_t *a, int N) {
    /* Convert to fp32 scratch */
    float *buf = (float *)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) buf[i] = FP8_LUT[a[i]];

    /* Binary tree reduction */
    int n = N;
    while (n > 1) {
        int half = n / 2;
        for (int i = 0; i < half; i++)
            buf[i] += buf[i + half];
        if (n & 1) buf[0] += buf[n - 1];
        n = half;
    }
    float result = buf[0];
    free(buf);
    return result;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * § 5  —  FP8 MULTIPLICATION  (three variants)
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * Variant A — Via fp32 (reference, always correct).
 */
fp8_t fp8_mul_fp32(fp8_t a, fp8_t b) {
    return fp32_to_fp8(fp8_to_fp32(a) * fp8_to_fp32(b));
}

/*
 * Variant B — Direct bit manipulation (no float hardware).
 *   Useful on MCUs without FPU or for FPGA soft-core emulation.
 *
 *   sign  = s_a XOR s_b
 *   exp   = e_a + e_b - BIAS
 *   mant  = (1.m_a × 1.m_b) → needs 1-bit normalisation
 */
fp8_t fp8_mul_bits(fp8_t a, fp8_t b) {
    /* handle special cases */
    uint8_t a7 = a & 0x7Fu, b7 = b & 0x7Fu;
    if (a7 == 0x7Fu || b7 == 0x7Fu) return FP8_NAN;
    if (a7 == 0u   || b7 == 0u  )  return (fp8_t)((a ^ b) & 0x80u);  /* ±0  */

    uint8_t sign = ((a ^ b) & 0x80u);
    int32_t ea   = (a >> 3) & 0xF;
    int32_t eb   = (b >> 3) & 0xF;
    uint32_t ma  = a & 0x7u;
    uint32_t mb  = b & 0x7u;

    /* subnormal promotion: treat as 0.mantissa × 2^(1-bias) */
    if (ea == 0) { /* subnormal a */
        /* normalise: shift left until implicit 1 appears */
        while (!(ma & 8u)) { ma <<= 1; ea--; }
        ma &= 7u;
        ea++;   /* account for removed implicit 1 */
    }
    if (eb == 0) {
        while (!(mb & 8u)) { mb <<= 1; eb--; }
        mb &= 7u;
        eb++;
    }

    /* 4-bit × 4-bit mantissa product (1.mmm × 1.mmm = 1x.xxxxxx) */
    uint32_t fa = (1u << 3) | ma;   /* 4-bit: 1.mmm                          */
    uint32_t fb = (1u << 3) | mb;
    uint32_t prod = fa * fb;        /* 8-bit product                          */

    /* result exponent (unbiased in E4M3 space) */
    int32_t er = ea + eb - FP8_E4M3_BIAS;

    /* normalise: product is either 0b1xxx_xxxx (bit6=1) or 0b01xx_xxxx       */
    if (prod & (1u << 6)) {   /* leading 1 at bit 6 → shift right 1          */
        er++;
        prod >>= 1;
    }
    /* now leading 1 at bit 5; extract top 3 mantissa bits with rounding      */
    uint32_t mr = (prod >> 2) & 0x7u;
    uint32_t r  = (prod >> 1) & 1u;  /* round bit                            */
    uint32_t s  = (prod & 1u);       /* sticky bit                            */
    mr += (r & (mr & 1u)) | (r & s); /* RNE round                            */
    if (mr > 7u) { mr = 0u; er++; }

    /* overflow / underflow */
    if (er >= 15) return (fp8_t)(sign | 0x7Eu);          /* clamp to ±448    */
    if (er <= 0)  return (fp8_t)(sign);                   /* underflow → 0    */

    return (fp8_t)(sign | ((uint32_t)er << 3) | mr);
}

/*
 * Variant C — LUT-based 256×256 multiply table (FASTEST for repeated ops).
 *   256×256 = 65536 entries × 1 byte = 64 KB (fits in L2/L3 cache).
 *   Precompute once; lookup in 1 cycle per multiply.
 *   This is the approach used in early TPU software stacks.
 *
 *   NOTE: 64KB may evict other working data; benchmark on your target.
 */
static fp8_t  MUL_LUT[256][256];
static int    mul_lut_init = 0;

void fp8_init_mul_lut(void) {
    if (mul_lut_init) return;
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            MUL_LUT[i][j] = fp8_mul_fp32((fp8_t)i, (fp8_t)j);
    mul_lut_init = 1;
}

static inline fp8_t fp8_mul_lut(fp8_t a, fp8_t b) {
    return MUL_LUT[a][b];
}

/* ══════════════════════════════════════════════════════════════════════════════
 * § 6  —  BENCHMARK  (wall-clock comparison of sum variants)
 * ══════════════════════════════════════════════════════════════════════════════*/

#define BENCH_N   (1 << 16)   /* 65536 elements                               */
#define BENCH_REP 200

void benchmark_sum(void) {
    fp8_init_lut();

    fp8_t *data = (fp8_t *)malloc(BENCH_N);
    srand(1337);
    for (int i = 0; i < BENCH_N; i++) {
        float v = ((float)rand()/(float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = fp32_to_fp8(v);
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     SUMMATION BENCHMARK  (N=%d, %d reps)              ║\n",
           BENCH_N, BENCH_REP);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    struct timespec t0, t1;
    volatile float sink = 0.0f;   /* prevent dead-code elimination            */

    #define TIME_IT(label, expr)                                              \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                  \
        for (int r = 0; r < BENCH_REP; r++) { sink += (expr); }              \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                  \
        {                                                                      \
            double us = ((t1.tv_sec - t0.tv_sec)*1e9 +                        \
                         (t1.tv_nsec - t0.tv_nsec)) / (1e3 * BENCH_REP);     \
            printf("  %-30s %8.1f µs/call  (sum≈%.2f)\n",                    \
                   label, us, (double)sink/BENCH_REP);                        \
            sink = 0.0f;                                                       \
        }

    TIME_IT("Naive (fp32 path/elem):",    fp8_sum_naive(data, BENCH_N))
    TIME_IT("LUT + 8× unroll:",           fp8_sum_lut(data, BENCH_N))
    TIME_IT("Pairwise (log2 error):",     fp8_sum_pairwise(data, BENCH_N))
    TIME_IT("Pre-scaled S=65536:",        fp8_sum_prescaled(data, BENCH_N, 65536))

    free(data);
    (void)sink;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * § 7  —  BENCHMARK: multiplication variants
 * ══════════════════════════════════════════════════════════════════════════════*/

#define MUL_N (1 << 16)

void benchmark_mul(void) {
    fp8_init_mul_lut();
    fp8_t *a = (fp8_t*)malloc(MUL_N);
    fp8_t *b = (fp8_t*)malloc(MUL_N);
    srand(42);
    for (int i = 0; i < MUL_N; i++) {
        a[i] = fp32_to_fp8(((float)rand()/(float)RAND_MAX)*100.0f - 50.0f);
        b[i] = fp32_to_fp8(((float)rand()/(float)RAND_MAX)*100.0f - 50.0f);
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     MULTIPLICATION BENCHMARK  (N=%d, elem-wise)        ║\n", MUL_N);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    struct timespec t0, t1;
    volatile fp8_t sink8 = 0;

    #define TIME_MUL(label, expr)                                             \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                  \
        for (int i = 0; i < MUL_N; i++) sink8 ^= (expr);                     \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                  \
        {                                                                      \
            double ns = ((t1.tv_sec - t0.tv_sec)*1e9 +                        \
                         (t1.tv_nsec - t0.tv_nsec));                          \
            printf("  %-30s %6.2f ns/op\n", label, ns / MUL_N);              \
            sink8 = 0;                                                         \
        }

    TIME_MUL("Via FP32 (reference):",    fp8_mul_fp32(a[i], b[i]))
    TIME_MUL("Bit-manipulation:",        fp8_mul_bits(a[i], b[i]))
    TIME_MUL("LUT 256×256 (fastest):",   fp8_mul_lut(a[i], b[i]))

    free(a); free(b);
    (void)sink8;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * § 8  —  OPTIMISATION SUMMARY
 * ══════════════════════════════════════════════════════════════════════════════*/

void print_optimisation_guide(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                 OPTIMISATION CHEAT-SHEET                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    puts(
    "  ┌── THROUGHPUT ──────────────────────────────────────────────────────┐\n"
    "  │  • LUT (1 KB for fp8→fp32):  eliminates all decode logic,          │\n"
    "  │    pure table-lookup + add.  GCC/Clang auto-vectorises this.        │\n"
    "  │  • 256×256 MUL LUT (64 KB): 1-cycle multiply, but watch L2 evict.  │\n"
    "  │  • Unroll 8× or 16× to keep FP32 adder pipe full.                  │\n"
    "  └────────────────────────────────────────────────────────────────────┘\n"
    "\n"
    "  ┌── PRECISION ──────────────────────────────────────────────────────┐\n"
    "  │  • Use FP32 accumulator — NEVER accumulate in FP8 (too noisy).    │\n"
    "  │  • Pairwise summation → O(log N·ε) vs O(N·ε) error.               │\n"
    "  │  • Kahan compensated sum for N > 1M.                               │\n"
    "  └───────────────────────────────────────────────────────────────────┘\n"
    "\n"
    "  ┌── SCALING ────────────────────────────────────────────────────────┐\n"
    "  │  • S = next_pow2(N): free dequant with bit-shift (*S = <<log2(S)) │\n"
    "  │  • Block scaling (block B=64..256): per-block scale stored as fp8  │\n"
    "  │    exponent; matches NVIDIA FP8 Tensor Core semantics.             │\n"
    "  │  • For training: dynamic loss scaling, track overflow/underflow.   │\n"
    "  └───────────────────────────────────────────────────────────────────┘\n"
    "\n"
    "  ┌── HARDWARE ───────────────────────────────────────────────────────┐\n"
    "  │  • NVIDIA H100/H200: native FP8 GEMM via Tensor Cores.            │\n"
    "  │    cuBLAS cublasLtMatmul with CUDA_R_8F_E4M3 dtype.               │\n"
    "  │  • RTX 3050 (yours): no native FP8, use LUT path + CUDA int8      │\n"
    "  │    intrinsics (dp4a) for 4-element dot products as proxy.          │\n"
    "  │  • FPGA: implement LUT as BRAM; multiply via log-space addition.   │\n"
    "  └───────────────────────────────────────────────────────────────────┘\n"
    "\n"
    "  ┌── CUDA KERNEL HINT (RTX 3050 without native FP8 hw) ─────────────┐\n"
    "  │  __device__ float fp8_sum_warp(uint8_t *ptr, int n) {             │\n"
    "  │    float acc = 0.f;                                                │\n"
    "  │    for (int i = tid; i < n; i += 32)                              │\n"
    "  │      acc += __ldg(&LUT[ptr[i]]);  // cached texture load          │\n"
    "  │    // warp-reduce with __shfl_down_sync                            │\n"
    "  │    for (int m=16;m;m>>=1) acc+=__shfl_down_sync(0xFFFFFFFF,acc,m);│\n"
    "  │    return acc;                                                      │\n"
    "  │  }                                                                  │\n"
    "  └───────────────────────────────────────────────────────────────────┘\n"
    );
}

/* ══════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════════════════*/

int main(void) {
    fp8_init_lut();

    /* § 2 — data loss */
    analyse_data_loss();

    /* § 3 — scaling */
    analyse_scaling();

    /* § 4/5 — benchmarks */
    benchmark_sum();
    benchmark_mul();

    /* § 6 — guide */
    print_optimisation_guide();

    /* ── quick correctness sanity ── */
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║               QUICK CORRECTNESS CHECK                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    float tests[] = {1.0f, -1.0f, 0.5f, 448.0f, 0.001f, 123.456f, -0.03125f};
    for (int i = 0; i < (int)(sizeof(tests)/sizeof(tests[0])); i++) {
        float x  = tests[i];
        fp8_t q  = fp32_to_fp8(x);
        float xq = fp8_to_fp32(q);
        printf("  fp32 = %12.6f  →  fp8 = 0x%02X  →  fp32 = %12.6f"
               "  (err = %.4f%%)\n",
               x, q, xq, fabsf(x - xq) / fabsf(x) * 100.0f);
    }

    /* multiply check */
    printf("\n  Multiply: 2.0 × 3.0 = %.1f (fp8: 0x%02X → %.1f)\n",
           2.0f*3.0f,
           fp8_mul_bits(fp32_to_fp8(2.0f), fp32_to_fp8(3.0f)),
           fp8_to_fp32(fp8_mul_bits(fp32_to_fp8(2.0f), fp32_to_fp8(3.0f))));
    printf("  Multiply: 1.5 × 4.0 = %.1f (fp8: 0x%02X → %.1f)\n",
           1.5f*4.0f,
           fp8_mul_bits(fp32_to_fp8(1.5f), fp32_to_fp8(4.0f)),
           fp8_to_fp32(fp8_mul_bits(fp32_to_fp8(1.5f), fp32_to_fp8(4.0f))));

    return 0;
}