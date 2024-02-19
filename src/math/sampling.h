#ifndef PT_SAMPLING_H
#define PT_SAMPLING_H

#include "../math/math_utils.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

/*
 * Most sampling code was taken from Physically Based Rendering, 4th edition
 * */

vec2
sample_uniform_disk_concentric(const vec2 &u);

// z-up
norm_vec3
sample_cosine_hemisphere(const vec2 &sample);

// z-up
vec3
sample_uniform_sphere(const vec2 &sample);

// z-up
vec3
sample_uniform_hemisphere(const vec2 &sample);

/// Taken from PBRT - UniformSampleTriangle.
/// Return barycentric coordinates that can be used to sample any triangle.
vec3
sample_uniform_triangle(const vec2 &sample);

/// Samples a CMF, return an index into the CMF slice.
/// Expects a normalized CMF.
u32
sample_discrete_cmf(Span<f32> cmf, f32 sample);

/// Samples a CMF, return a value in [0, 1), and an index into the CDF slice.
Tuple<f32, u32>
sample_continuous_cmf(Span<f32> cdf, f32 sample);

#endif // PT_SAMPLING_H
