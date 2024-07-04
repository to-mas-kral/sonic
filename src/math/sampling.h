#ifndef PT_SAMPLING_H
#define PT_SAMPLING_H

#include "../math/vecmath.h"

/*
 * Most sampling code was taken from Physically Based Rendering, 4th edition
 * */

vec2
sample_uniform_disk_concentric(const vec2 &u);

// z-up
norm_vec3
sample_cosine_hemisphere(const vec2 &sample);

// z-up
norm_vec3
sample_uniform_sphere(const vec2 &sample);

// z-up
norm_vec3
sample_uniform_hemisphere(const vec2 &sample);

/// Taken from PBRT - UniformSampleTriangle.
/// Return barycentric coordinates that can be used to sample any triangle.
vec3
sample_uniform_triangle(const vec2 &sample);

#endif // PT_SAMPLING_H
