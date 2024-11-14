#ifndef PT_TRANSFORM_H
#define PT_TRANSFORM_H

#include "../utils/basic_types.h"
#include "vecmath.h"

#include <array>

class SquareMatrix4 {
public:
    // clang-format off
    SquareMatrix4(f32 m00, f32 m01, f32 m02, f32 m03,
                    f32 m10, f32 m11, f32 m12, f32 m13,
                    f32 m20, f32 m21, f32 m22, f32 m23,
                    f32 m30, f32 m31, f32 m32, f32 m33);

    static SquareMatrix4
    from_columns(const tuple4 &c0, const tuple4 &c1, const tuple4 &c2, const tuple4 &c3);

    /// Column-major order !
    static SquareMatrix4
    from_elements(const std::array<f32, 16> &columns);

    SquareMatrix4
    transpose() const;

    /// code for inverse() adapted from GLM.
    SquareMatrix4
    inverse() const;

    SquareMatrix4
    operator*(f32 mul)const;

    /// *this* transform is performed first
    SquareMatrix4
    compose(const SquareMatrix4 &other) const;

    SquareMatrix4
    operator*(const SquareMatrix4 &other) const;

    static SquareMatrix4
    identity();

    /// Angle is in radians!
    static SquareMatrix4
    from_euler_x(f32 a);

    /// Angle is in radians!
    static SquareMatrix4
    from_euler_y(f32 a);

    /// Angle is in radians!
    static SquareMatrix4
    from_euler_z(f32 a);

    static SquareMatrix4
    from_translate(f32 x, f32 y, f32 z);

    static SquareMatrix4
    from_scale(f32 x, f32 y, f32 z);

    static SquareMatrix4
    from_lookat(vec3 eye, vec3 look, vec3 up);

    vec3
    transform_vec(const vec3 &v) const;

    point3
    transform_point(const point3 &p) const;

private:
    SquareMatrix4() = default;

    /// Column-major matrix
    std::array<std::array<f32, 4>, 4> mat;
};

class SquareMatrix3 {
public:
    // clang-format off
    SquareMatrix3(f32 m00, f32 m01, f32 m02,
                    f32 m10, f32 m11, f32 m12,
                    f32 m20, f32 m21, f32 m22);

    static SquareMatrix3
    from_columns(const tuple3 &c0, const tuple3 &c1, const tuple3 &c2)noexcept;

    /// Column-major order !
    static SquareMatrix3
    from_elements(const std::array<f32, 9> &columns);

    SquareMatrix3
    transpose() const;

    /*/// code for inverse() adapted from GLM.
     SquareMatrix3
    inverse() const {

    }*/

    SquareMatrix3
    operator*(f32 mul)const;

    /// *this* transform is performed first
    SquareMatrix3
    compose(const SquareMatrix3 &other) const;

    SquareMatrix3
    operator*(const SquareMatrix3 &other) const;

    tuple3
    operator*(const tuple3 &other) const;

    static SquareMatrix3
    identity();

private:
    SquareMatrix3() = default;

    /// Column-major matrix
    std::array<std::array<f32, 3>, 3> mat;
};

using mat4 = SquareMatrix4;
using mat3 = SquareMatrix3;

#endif // PT_TRANSFORM_H
