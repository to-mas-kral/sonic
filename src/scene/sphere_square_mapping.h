
#ifndef SPHERE_SQUARE_MAPPING_H
#define SPHERE_SQUARE_MAPPING_H

#include "../math/vecmath.h"

vec2
sphere_to_square(const norm_vec3 &arg_dir);

norm_vec3
square_to_sphere(const vec2 &uv);

#endif // SPHERE_SQUARE_MAPPING_H
