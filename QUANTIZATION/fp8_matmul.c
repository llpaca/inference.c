/// @file fp8_matmul.c  —  Pure FP8 matrix-vector multiply (E4M3)
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "fp8.h"

/// ── Optional feature gates ───────────────────────────────────────────────
/// Compile with -DUSE_SIMD   to enable AVX2 / NEON paths
/// Compile with -DUSE_THREADS to enable pthreads row-parallel path
/// ─────────────────────────────────────────────────────────────────────────

#ifdef USE_SIMD
#  if defined(__AVX2__)
#    include <immintrin.h>
#    define HAVE_AVX2
#  elif defined(__ARM_NEON)
#    include <arm_neon.h>
#    define HAVE_NEON
#  endif
#endif

#ifdef USE_THREADS
#  include <pthread.h>
#endif


/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  SCALAR BASELINE
 *     out[i]  = Σ_j  W[i*in_dim + j]  *  x[j]        (FP8 × FP8 → fp32 acc)
 *     out[i] += Σ_j  W[i*in_dim + j]  *  x[j]        (accumulate variant)
 *
 *     Accumulation is done in fp32 to avoid catastrophic cancellation;
 *     we only stay in FP8 for the per-element multiply.
 * ═══════════════════════════════════════════════════════════════════════════ */

void fp8_matmul_scalar(float *out, const f8 *x, const f8 *W,
                       int out_dim, int in_dim)
{
    for (int i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        const f8 *row = W + (size_t)i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            /* fp8_mul → FP8 product, then widen to fp32 for accumulation   */
            acc += fp8_to_fp32(fp8_mul_precompt(row[j], x[j]));
        }
        out[i] = acc;
    }
}

void fp8_matmul_scalar_acc(float *out, const f8 *x, const f8 *W,
                            int out_dim, int in_dim)
{
    for (int i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        const f8 *row = W + (size_t)i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            acc += fp8_to_fp32(fp8_mul_precompt(row[j], x[j]));
        }
        out[i] += acc;   /* += matches beta=1 behaviour */
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  DEQUANTISE-THEN-SGEMV  (fast-path scalar)
 *
 *     Widening the full row to fp32 up-front lets the inner loop use plain
 *     fp32 FMA, which is far cheaper than per-pair fp8_mul + fp8_to_fp32.
 *     Stack-allocate a row buffer so there is no heap traffic per call.
 *
 *     Threshold: break-even vs pure-FP8 path is ≈ in_dim > 32 on most uarches.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Maximum stack buffer (bytes). Rows wider than this fall back to scalar.   */
#define FP8_ROW_STACK_LIMIT  4096

void fp8_matmul_deq(float *out, const f8 *x, const f8 *W,
                    int out_dim, int in_dim)
{
    /* Widen x once — reused for every output row */
    float x_f32_stack[FP8_ROW_STACK_LIMIT];
    float *x_f32 = (in_dim <= FP8_ROW_STACK_LIMIT)
                    ? x_f32_stack
                    : (float *)malloc((size_t)in_dim * sizeof(float));

    for (int j = 0; j < in_dim; j++)
        x_f32[j] = fp8_to_fp32(x[j]);

    for (int i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        const f8 *row = W + (size_t)i * in_dim;
        for (int j = 0; j < in_dim; j++)
            acc += fp8_to_fp32(row[j]) * x_f32[j];
        out[i] = acc;
    }

    if (x_f32 != x_f32_stack) free(x_f32);
}

void fp8_matmul_deq_acc(float *out, const f8 *x, const f8 *W,
                         int out_dim, int in_dim)
{
    float x_f32_stack[FP8_ROW_STACK_LIMIT];
    float *x_f32 = (in_dim <= FP8_ROW_STACK_LIMIT)
                    ? x_f32_stack
                    : (float *)malloc((size_t)in_dim * sizeof(float));

    for (int j = 0; j < in_dim; j++)
        x_f32[j] = fp8_to_fp32(x[j]);

    for (int i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        const f8 *row = W + (size_t)i * in_dim;
        for (int j = 0; j < in_dim; j++)
            acc += fp8_to_fp32(row[j]) * x_f32[j];
        out[i] += acc;
    }

    if (x_f32 != x_f32_stack) free(x_f32);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  SIMD PATHS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ───────────────────────────────────────────────────────────────────────────
 * 3a.  AVX2  (x86-64)
 *
 *  Strategy:
 *   • Dequantise 8 FP8 bytes → 8 fp32 lanes using _mm256_cvtepu8_epi32 +
 *     a manual unpack (no hardware FP8→FP32 until AVX-512-FP8 / H100 class).
 *   • Inner loop processes 8 elements per iteration via 256-bit FMA.
 *   • Tail (in_dim % 8) handled by scalar.
 *
 *  The deq helper converts 8 packed E4M3 bytes held in the low 64 bits of
 *  an __m128i into an __m256 of fp32 values.
 * ─────────────────────────────────────────────────────────────────────────── */
#ifdef HAVE_AVX2

/* Expand 8-bit E4M3 → fp32  (8 elements at a time) */
static inline __m256 deq8_avx2(const f8 *p)
{
    /* Load 8 bytes */
    __m128i v8 = _mm_loadl_epi64((const __m128i *)p);

    /* Unpack to 32-bit lanes */
    __m256i v32 = _mm256_cvtepu8_epi32(v8);

    /* Extract sign / exponent / mantissa with integer SIMD */
    __m256i vsign = _mm256_and_si256(_mm256_srli_epi32(v32, 7),
                                     _mm256_set1_epi32(1));
    __m256i vexp  = _mm256_and_si256(_mm256_srli_epi32(v32, 3),
                                     _mm256_set1_epi32(0xF));
    __m256i vmant = _mm256_and_si256(v32, _mm256_set1_epi32(0x7));

    /* ── subnormal lane: e8==0 → value = mant * 2^-9 ── */
    __m256  sub_val = _mm256_mul_ps(
                          _mm256_cvtepi32_ps(vmant),
                          _mm256_set1_ps(1.0f / 512.0f));

    /* ── normal lane: value = 2^(e8-7) × (1 + mant/8) ── */
    /* fp32 biased exponent = (e8 - 7) + 127 = e8 + 120 */
    __m256i vfexp  = _mm256_slli_epi32(
                         _mm256_add_epi32(vexp, _mm256_set1_epi32(120)),
                         23);
    __m256  vpow2  = _mm256_castsi256_ps(vfexp);   /* 2^(e8-7)           */
    __m256  vnorm  = _mm256_mul_ps(
                         vpow2,
                         _mm256_add_ps(
                             _mm256_set1_ps(1.0f),
                             _mm256_mul_ps(_mm256_cvtepi32_ps(vmant),
                                           _mm256_set1_ps(1.0f / 8.0f))));

    /* Select subnormal or normal */
    __m256i is_sub = _mm256_cmpeq_epi32(vexp, _mm256_setzero_si256());
    __m256  result = _mm256_blendv_ps(vnorm, sub_val, _mm256_castsi256_ps(is_sub));

    /* Apply sign */
    __m256  sign_mask = _mm256_castsi256_ps(
                            _mm256_slli_epi32(vsign, 31));
    return _mm256_xor_ps(result, sign_mask);
}

static void fp8_matmul_avx2_row(float *out, const float *x_f32,
                                 const f8 *W, int out_dim, int in_dim)
{
    int tail = in_dim & 7;
    int body = in_dim - tail;

    for (int i = 0; i < out_dim; i++) {
        const f8 *row = W + (size_t)i * in_dim;
        __m256 vacc = _mm256_setzero_ps();

        for (int j = 0; j < body; j += 8) {
            __m256 vw = deq8_avx2(row + j);
            __m256 vx = _mm256_loadu_ps(x_f32 + j);
            vacc = _mm256_fmadd_ps(vw, vx, vacc);
        }

        /* Horizontal sum of 8 lanes */
        __m128 lo  = _mm256_castps256_ps128(vacc);
        __m128 hi  = _mm256_extractf128_ps(vacc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float acc = _mm_cvtss_f32(sum);

        /* Scalar tail */
        for (int j = body; j < in_dim; j++)
            acc += fp8_to_fp32(row[j]) * x_f32[j];

        out[i] = acc;
    }
}

#endif /* HAVE_AVX2 */


/* ───────────────────────────────────────────────────────────────────────────
 * 3b.  ARM NEON  (AArch64)
 *
 *  Strategy identical to AVX2 but uses 4-wide float32x4_t lanes.
 *  Inner loop: 8 FP8 bytes → two float32x4_t → two vfmaq_f32 per iteration.
 * ─────────────────────────────────────────────────────────────────────────── */
#ifdef HAVE_NEON

/* Expand 4 packed E4M3 bytes (low 4 of an uint8x8_t) → float32x4_t */
static inline float32x4_t deq4_neon(uint8x8_t v8, int lane_base)
{
    /* Extract 4 bytes as uint32 */
    uint32x4_t v32;
    v32 = vsetq_lane_u32(vget_lane_u8(v8, lane_base+0), v32, 0);
    v32 = vsetq_lane_u32(vget_lane_u8(v8, lane_base+1), v32, 1);
    v32 = vsetq_lane_u32(vget_lane_u8(v8, lane_base+2), v32, 2);
    v32 = vsetq_lane_u32(vget_lane_u8(v8, lane_base+3), v32, 3);

    uint32x4_t vsign = vandq_u32(vshrq_n_u32(v32, 7), vdupq_n_u32(1));
    uint32x4_t vexp  = vandq_u32(vshrq_n_u32(v32, 3), vdupq_n_u32(0xF));
    uint32x4_t vmant = vandq_u32(v32,                  vdupq_n_u32(0x7));

    /* Subnormal: mant * 2^-9 */
    float32x4_t sub_val = vmulq_n_f32(vcvtq_f32_u32(vmant), 1.0f / 512.0f);

    /* Normal: 2^(e8-7) × (1 + mant/8).
     * fp32 exponent bits = (e8 + 120) << 23                              */
    uint32x4_t vfexp  = vshlq_n_u32(vaddq_u32(vexp, vdupq_n_u32(120)), 23);
    float32x4_t vpow2 = vreinterpretq_f32_u32(vfexp);
    float32x4_t vnorm = vmulq_f32(
                            vpow2,
                            vaddq_f32(vdupq_n_f32(1.0f),
                                      vmulq_n_f32(vcvtq_f32_u32(vmant),
                                                  1.0f / 8.0f)));

    /* Select sub / normal */
    uint32x4_t is_sub = vceqq_u32(vexp, vdupq_n_u32(0));
    float32x4_t result = vbslq_f32(is_sub, sub_val, vnorm);

    /* Apply sign */
    uint32x4_t sign_bits = vshlq_n_u32(vsign, 31);
    return vreinterpretq_f32_u32(
               veorq_u32(vreinterpretq_u32_f32(result), sign_bits));
}

static void fp8_matmul_neon_row(float *out, const float *x_f32,
                                 const f8 *W, int out_dim, int in_dim)
{
    int tail = in_dim & 7;
    int body = in_dim - tail;

    for (int i = 0; i < out_dim; i++) {
        const f8 *row = W + (size_t)i * in_dim;
        float32x4_t vacc0 = vdupq_n_f32(0.0f);
        float32x4_t vacc1 = vdupq_n_f32(0.0f);

        for (int j = 0; j < body; j += 8) {
            uint8x8_t v8 = vld1_u8(row + j);
            float32x4_t vw0 = deq4_neon(v8, 0);
            float32x4_t vw1 = deq4_neon(v8, 4);
            float32x4_t vx0 = vld1q_f32(x_f32 + j);
            float32x4_t vx1 = vld1q_f32(x_f32 + j + 4);
            vacc0 = vfmaq_f32(vacc0, vw0, vx0);
            vacc1 = vfmaq_f32(vacc1, vw1, vx1);
        }

        float32x4_t vacc = vaddq_f32(vacc0, vacc1);
        float32x2_t s2   = vadd_f32(vget_low_f32(vacc), vget_high_f32(vacc));
        float acc = vget_lane_f32(vpadd_f32(s2, s2), 0);

        for (int j = body; j < in_dim; j++)
            acc += fp8_to_fp32(row[j]) * x_f32[j];

        out[i] = acc;
    }
}

#endif /* HAVE_NEON */


/* ═══════════════════════════════════════════════════════════════════════════
 * 4.  THREADED WRAPPER  (pthreads, row-parallel)
 *
 *  Each thread owns a contiguous stripe of output rows.
 *  The inner kernel per thread is whichever SIMD path compiled in.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifdef USE_THREADS

#define FP8_MAX_THREADS 16

typedef struct {
    float       *out;
    const float *x_f32;      /* pre-widened activation vector              */
    const f8    *W;
    int          out_dim;
    int          in_dim;
    int          row_start;
    int          row_end;
} Fp8MatmulJob;

static void *fp8_matmul_thread(void *arg)
{
    Fp8MatmulJob *j = (Fp8MatmulJob *)arg;
    int  slice = j->row_end - j->row_start;
    float *out_slice = j->out + j->row_start;
    const f8 *W_slice = j->W + (size_t)j->row_start * j->in_dim;

#if defined(HAVE_AVX2)
    fp8_matmul_avx2_row(out_slice, j->x_f32, W_slice, slice, j->in_dim);
#elif defined(HAVE_NEON)
    fp8_matmul_neon_row(out_slice, j->x_f32, W_slice, slice, j->in_dim);
#else
    /* scalar deq fallback */
    for (int i = 0; i < slice; i++) {
        float acc = 0.0f;
        const f8 *row = W_slice + (size_t)i * j->in_dim;
        for (int k = 0; k < j->in_dim; k++)
            acc += fp8_to_fp32(row[k]) * j->x_f32[k];
        out_slice[i] = acc;
    }
#endif
    return NULL;
}

void fp8_matmul_threaded(float *out, const f8 *x, const f8 *W,
                          int out_dim, int in_dim, int n_threads)
{
    if (n_threads < 1)  n_threads = 1;
    if (n_threads > FP8_MAX_THREADS) n_threads = FP8_MAX_THREADS;

    /* Widen x once, shared across all threads (read-only) */
    float *x_f32 = (float *)malloc((size_t)in_dim * sizeof(float));
    for (int j = 0; j < in_dim; j++)
        x_f32[j] = fp8_to_fp32(x[j]);

    pthread_t    tids[FP8_MAX_THREADS];
    Fp8MatmulJob jobs[FP8_MAX_THREADS];

    int rows_per = out_dim / n_threads;
    int extra    = out_dim % n_threads;
    int start    = 0;

    for (int t = 0; t < n_threads; t++) {
        int rows       = rows_per + (t < extra ? 1 : 0);
        jobs[t].out     = out;
        jobs[t].x_f32   = x_f32;
        jobs[t].W       = W;
        jobs[t].out_dim = out_dim;
        jobs[t].in_dim  = in_dim;
        jobs[t].row_start = start;
        jobs[t].row_end   = start + rows;
        start += rows;
        pthread_create(&tids[t], NULL, fp8_matmul_thread, &jobs[t]);
    }
    for (int t = 0; t < n_threads; t++)
        pthread_join(tids[t], NULL);

    free(x_f32);
}

void fp8_matmul_threaded_acc(float *out, const f8 *x, const f8 *W,
                              int out_dim, int in_dim, int n_threads)
{
    /* Compute into a temp buffer, then accumulate — avoids false-sharing  */
    float *tmp = (float *)malloc((size_t)out_dim * sizeof(float));
    fp8_matmul_threaded(tmp, x, W, out_dim, in_dim, n_threads);
    for (int i = 0; i < out_dim; i++) out[i] += tmp[i];
    free(tmp);
}

#endif /* USE_THREADS */


/* ═══════════════════════════════════════════════════════════════════════════
 * 5.  UNIFIED DISPATCH  —  fp8_matmul / fp8_matmul_acc
 *
 *  Mirrors your original matmul / matmul_acc signature exactly.
 *  Picks the best available backend at compile-time.
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef USE_THREADS
#  ifndef FP8_N_THREADS
#    define FP8_N_THREADS 4
#  endif
#endif

void fp8_matmul(float *out, const f8 *x, const f8 *W,
                int out_dim, int in_dim)
{
#if defined(USE_THREADS)
    fp8_matmul_threaded(out, x, W, out_dim, in_dim, FP8_N_THREADS);

#elif defined(HAVE_AVX2) || defined(HAVE_NEON)
    /* SIMD deq path — widen x once, then dispatch */
    float x_f32_stack[FP8_ROW_STACK_LIMIT];
    float *x_f32 = (in_dim <= FP8_ROW_STACK_LIMIT)
                    ? x_f32_stack
                    : (float *)malloc((size_t)in_dim * sizeof(float));
    for (int j = 0; j < in_dim; j++) x_f32[j] = fp8_to_fp32(x[j]);

#  if defined(HAVE_AVX2)
    fp8_matmul_avx2_row(out, x_f32, W, out_dim, in_dim);
#  else
    fp8_matmul_neon_row(out, x_f32, W, out_dim, in_dim);
#  endif
    if (x_f32 != x_f32_stack) free(x_f32);

#else
    fp8_matmul_deq(out, x, W, out_dim, in_dim);
#endif
}

void fp8_matmul_acc(float *out, const f8 *x, const f8 *W,
                    int out_dim, int in_dim)
{
#if defined(USE_THREADS)
    fp8_matmul_threaded_acc(out, x, W, out_dim, in_dim, FP8_N_THREADS);

#elif defined(HAVE_AVX2) || defined(HAVE_NEON)
    float x_f32_stack[FP8_ROW_STACK_LIMIT];
    float *x_f32 = (in_dim <= FP8_ROW_STACK_LIMIT)
                    ? x_f32_stack
                    : (float *)malloc((size_t)in_dim * sizeof(float));
    for (int j = 0; j < in_dim; j++) x_f32[j] = fp8_to_fp32(x[j]);

    float *tmp = (float *)malloc((size_t)out_dim * sizeof(float));
#  if defined(HAVE_AVX2)
    fp8_matmul_avx2_row(tmp, x_f32, W, out_dim, in_dim);
#  else
    fp8_matmul_neon_row(tmp, x_f32, W, out_dim, in_dim);
#  endif
    for (int i = 0; i < out_dim; i++) out[i] += tmp[i];
    free(tmp);
    if (x_f32 != x_f32_stack) free(x_f32);

#else
    fp8_matmul_deq_acc(out, x, W, out_dim, in_dim);
#endif
}