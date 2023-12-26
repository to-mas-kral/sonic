#ifndef PT_SPECTRUM_CONSTS_H
#define PT_SPECTRUM_CONSTS_H

__device__ constexpr u32 LAMBDA_MIN = 360;
__device__ constexpr u32 LAMBDA_MAX = 830;
__device__ constexpr u32 LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN + 1;

__device__ constexpr f32 CIE_Y_INTEGRAL = 106.856895;

#endif // PT_SPECTRUM_CONSTS_H
