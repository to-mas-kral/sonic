#ifndef PT_SPECTRUM_CONSTS_H
#define PT_SPECTRUM_CONSTS_H

#include "../utils/basic_types.h"

constexpr u32 LAMBDA_MIN = 360;
constexpr u32 LAMBDA_MAX = 830;
constexpr u32 LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN + 1;

constexpr f32 CIE_Y_INTEGRAL = 106.856895;

#endif // PT_SPECTRUM_CONSTS_H
