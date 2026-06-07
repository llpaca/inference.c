#pragma once

typedef uint8_t f8;
f8 fp8_mul(f8 a, f8 b);
f8 fp32_to_fp8(float x);
float fp8_to_fp32(f8 v);