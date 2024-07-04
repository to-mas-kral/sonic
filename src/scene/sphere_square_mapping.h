
#ifndef SPHERE_SQUARE_MAPPING_H
#define SPHERE_SQUARE_MAPPING_H

#include "../math/vecmath.h"

/// Coordinate system:
/// Returns xy-coords ranging 0-1
///
/// XY coords are top to bottom!:
///
/// 0,0 |               x
///   --.--------------->
///     |
///     |
///     |
///     |
///     |
///     |
///  y  |
vec2
sphere_to_square(const norm_vec3 &arg_dir);

/// Coordinate system: see above
norm_vec3
square_to_sphere(const vec2 &xy);

#endif // SPHERE_SQUARE_MAPPING_H
