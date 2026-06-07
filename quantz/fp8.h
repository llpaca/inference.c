///@file: fp8.c
#include <stdio.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
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

typedef uint8_t f8;

f8 mul(f8 a, f8 b){
    int S = ((a ^ b) >> 7) & 1;
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;
    int M = ((1 << 3) | (a & 0x7)) * ((1 << 3) | (b & 0x7));

    if(M >= 128){
        M >>= 1;
        E++;
    }

    int mant = (M+4) >> 3;
    if(mant >= 16){
        mant = 0;
        E++;
    }else{
        mant &= 0x7;
    }
    if (E <= 0) return 0;
    if (E >= 15) return (S<<7) | (15<<3);

    return (S<<7) | (E<<3) | mant;
}

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

/// Timing function for benchmarking
long benchmark_ns(f8 (*func)(f8,f8), f8 a, f8 b, long iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    f8 result = 0;
    for(long i = 0; i < iterations; i++) {
        result ^= func(a,b); // prevent compiler optimizing away
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L
                    + (end.tv_nsec - start.tv_nsec);
    printf("Dummy result: %d (to avoid optimization)\n", result);
    return elapsed_ns;
}

long benchmark_ns_random(f8 (*func)(f8,f8), long iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    f8 result = 0;
    for(long i = 0; i < iterations; i++) {
        f8 a = rand() & 0xFF;  // random 8-bit value
        f8 b = rand() & 0xFF;  // random 8-bit value
        result ^= func(a,b);    // prevent compiler optimizing away
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L
                    + (end.tv_nsec - start.tv_nsec);
    printf("Dummy result: %d\n", result);
    return elapsed_ns;
}

int main() {
    srand(time(NULL));   // seed RNG

    long iters = 10000000;  // 10 million
    long t1 = benchmark_ns_random(mul, iters);
    printf("mul took %ld ns for %ld iterations (avg %.2f ns)\n",
           t1, iters, (double)t1/iters);

    long t2 = benchmark_ns_random(fp8_mul, iters);
    printf("fp8_mul took %ld ns for %ld iterations (avg %.2f ns)\n",
           t2, iters, (double)t2/iters);

    long t3 = benchmark_ns_random(fp8_mul_precompt, iters);
    printf("fp8_mul_precompt took %ld ns for %ld iterations (avg %.2f ns)\n",
           t3, iters, (double)t3/iters);

    return 0;
}