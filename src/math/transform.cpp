#include "transform.h"

SquareMatrix4::
SquareMatrix4(f32 m00, f32 m01, f32 m02, f32 m03, f32 m10, f32 m11, f32 m12, f32 m13,
              f32 m20, f32 m21, f32 m22, f32 m23, f32 m30, f32 m31, f32 m32, f32 m33) {
    // clang-format on
    mat[0][0] = m00;
    mat[0][1] = m01;
    mat[0][2] = m02;
    mat[0][3] = m03;

    mat[1][0] = m10;
    mat[1][1] = m11;
    mat[1][2] = m12;
    mat[1][3] = m13;

    mat[2][0] = m20;
    mat[2][1] = m21;
    mat[2][2] = m22;
    mat[2][3] = m23;

    mat[3][0] = m30;
    mat[3][1] = m31;
    mat[3][2] = m32;
    mat[3][3] = m33;
}

SquareMatrix4
SquareMatrix4::from_columns(const tuple4 &c0, const tuple4 &c1, const tuple4 &c2,
                            const tuple4 &c3) {
    // clang-format off
    return SquareMatrix4(c0.x,  c0.y,  c0.z, c0.w,
                         c1.x,  c1.y,  c1.z, c1.w,
                         c2.x,  c2.y,  c2.z, c2.w,
                         c3.x,  c3.y,  c3.z, c3.w);
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_elements(const Array<f32, 16> &columns) {
    // clang-format off
    return SquareMatrix4(columns[0],  columns[1],  columns[2],  columns[3],
                         columns[4],  columns[5],  columns[6],  columns[7],
                         columns[8],  columns[9],  columns[10], columns[11],
                         columns[12], columns[13], columns[14], columns[15]);
    // clang-format on
}

SquareMatrix4
SquareMatrix4::transpose() const {
    SquareMatrix4 new_mat{};
    new_mat.mat[0][0] = mat[0][0];
    new_mat.mat[0][1] = mat[1][0];
    new_mat.mat[0][2] = mat[2][0];
    new_mat.mat[0][3] = mat[3][0];

    new_mat.mat[1][0] = mat[0][1];
    new_mat.mat[1][1] = mat[1][1];
    new_mat.mat[1][2] = mat[2][1];
    new_mat.mat[1][3] = mat[3][1];

    new_mat.mat[2][0] = mat[0][2];
    new_mat.mat[2][1] = mat[1][2];
    new_mat.mat[2][2] = mat[2][2];
    new_mat.mat[2][3] = mat[3][2];

    new_mat.mat[3][0] = mat[0][3];
    new_mat.mat[3][1] = mat[1][3];
    new_mat.mat[3][2] = mat[2][3];
    new_mat.mat[3][3] = mat[3][3];

    return new_mat;
}

SquareMatrix4
SquareMatrix4::inverse() const {
    f32 coef00 = mat[2][2] * mat[3][3] - mat[3][2] * mat[2][3];
    f32 coef02 = mat[1][2] * mat[3][3] - mat[3][2] * mat[1][3];
    f32 coef03 = mat[1][2] * mat[2][3] - mat[2][2] * mat[1][3];

    f32 coef04 = mat[2][1] * mat[3][3] - mat[3][1] * mat[2][3];
    f32 coef06 = mat[1][1] * mat[3][3] - mat[3][1] * mat[1][3];
    f32 coef07 = mat[1][1] * mat[2][3] - mat[2][1] * mat[1][3];

    f32 coef08 = mat[2][1] * mat[3][2] - mat[3][1] * mat[2][2];
    f32 coef10 = mat[1][1] * mat[3][2] - mat[3][1] * mat[1][2];
    f32 coef11 = mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2];

    f32 coef12 = mat[2][0] * mat[3][3] - mat[3][0] * mat[2][3];
    f32 coef14 = mat[1][0] * mat[3][3] - mat[3][0] * mat[1][3];
    f32 coef15 = mat[1][0] * mat[2][3] - mat[2][0] * mat[1][3];

    f32 coef16 = mat[2][0] * mat[3][2] - mat[3][0] * mat[2][2];
    f32 coef18 = mat[1][0] * mat[3][2] - mat[3][0] * mat[1][2];
    f32 coef19 = mat[1][0] * mat[2][2] - mat[2][0] * mat[1][2];

    f32 coef20 = mat[2][0] * mat[3][1] - mat[3][0] * mat[2][1];
    f32 coef22 = mat[1][0] * mat[3][1] - mat[3][0] * mat[1][1];
    f32 coef23 = mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1];

    tuple4 fac0(coef00, coef00, coef02, coef03);
    tuple4 fac1(coef04, coef04, coef06, coef07);
    tuple4 fac2(coef08, coef08, coef10, coef11);
    tuple4 fac3(coef12, coef12, coef14, coef15);
    tuple4 fac4(coef16, coef16, coef18, coef19);
    tuple4 fac5(coef20, coef20, coef22, coef23);

    tuple4 vec_0 = tuple4(mat[1][0], mat[0][0], mat[0][0], mat[0][0]);
    tuple4 vec_1 = tuple4(mat[1][1], mat[0][1], mat[0][1], mat[0][1]);
    tuple4 vec_2 = tuple4(mat[1][2], mat[0][2], mat[0][2], mat[0][2]);
    tuple4 vec_3 = tuple4(mat[1][3], mat[0][3], mat[0][3], mat[0][3]);

    tuple4 sign_a(+1.f, -1.f, +1.f, -1.f);
    tuple4 sign_b(-1.f, +1.f, -1.f, +1.f);

    tuple4 inv0 = tuple4(vec_1 * fac0 - vec_2 * fac1 + vec_3 * fac2) * sign_a;
    tuple4 inv1 = tuple4(vec_0 * fac0 - vec_2 * fac3 + vec_3 * fac4) * sign_b;
    tuple4 inv2 = tuple4(vec_0 * fac1 - vec_1 * fac3 + vec_3 * fac5) * sign_a;
    tuple4 inv3 = tuple4(vec_0 * fac2 - vec_1 * fac4 + vec_2 * fac5) * sign_b;

    tuple4 row0 = tuple4(inv0.x, inv1.x, inv2.x, inv3.x);

    tuple4 dot0 = tuple4(tuple4(mat[0][0], mat[0][1], mat[0][2], mat[0][3]) * row0);
    f32 dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);

    f32 one_over_determinant = 1.f / dot1;

    auto inverse_mat = SquareMatrix4::from_columns(inv0, inv1, inv2, inv3);
    return inverse_mat * one_over_determinant;
}

SquareMatrix4
SquareMatrix4::operator*(f32 mul) const {
    auto new_mat = *this;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            new_mat.mat[i][j] *= mul;
        }
    }

    return new_mat;
}

SquareMatrix4
SquareMatrix4::compose(const SquareMatrix4 &other) const {
    return *this * other;
}

SquareMatrix4
SquareMatrix4::operator*(const SquareMatrix4 &other) const {
    SquareMatrix4 new_mat{};

    for (u32 i = 0; i < 4; i++) {
        for (u32 j = 0; j < 4; j++) {
            new_mat.mat[i][j] = mat[0][j] * other.mat[i][0] +
                                mat[1][j] * other.mat[i][1] +
                                mat[2][j] * other.mat[i][2] + mat[3][j] * other.mat[i][3];
        }
    }

    return new_mat;
}

SquareMatrix4
SquareMatrix4::identity() {
    // clang-format off
    return SquareMatrix4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    );
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_euler_x(f32 a) {
    // clang-format off
    return SquareMatrix4(
        1.f,  0.f,    0.f,    0.f,
        0.f,  std::cos(a), std::sin(a), 0.f,
        0.f, -std::sin(a), std::cos(a), 0.f,
        0.f,  0.f,    0.f,    1.f
    );
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_euler_y(f32 a) {
    // clang-format off
    return SquareMatrix4(
        std::cos(a), 0.f, -std::sin(a), 0.f,
        0.f,    1.f,  0.f,    0.f,
        std::sin(a), 0.f,  std::cos(a), 0.f,
        0.f,    0.f,  0.f,    1.f
    );
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_euler_z(f32 a) {
    // clang-format off
    return SquareMatrix4(
        std::cos(a), std::sin(a), 0.f, 0.f,
        -std::sin(a), std::cos(a), 0.f, 0.f,
        0.f,    0.f,    1.f, 0.f,
        0.f,    0.f,    0.f, 1.f
    );
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_translate(f32 x, f32 y, f32 z) {
    // clang-format off
    return SquareMatrix4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        x,   y,   z,   1.f
    );
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_scale(f32 x, f32 y, f32 z) {
    // clang-format off
    return SquareMatrix4(
        x,   0.f, 0.f, 0.f,
        0.f, y,   0.f, 0.f,
        0.f, 0.f, z,   0.f,
        0.f, 0.f, 0.f, 1.f
    );
    // clang-format on
}

SquareMatrix4
SquareMatrix4::from_lookat(vec3 eye, vec3 look, vec3 up) {
    // Taken from GLM
    const auto dir = (look - eye).normalized();
    const auto right = vec3::cross(up, dir).normalized();
    const auto new_up = vec3::cross(dir, right);

    // clang-format off
    return SquareMatrix4(
        right.x, new_up.x, dir.x, 0.f,
        right.y, new_up.y, dir.y, 0.f,
        right.z, new_up.z, dir.z, 0.f,
        -vec3::dot(right, eye), -vec3::dot(new_up, eye), -vec3::dot(dir, eye), 1.f
    );
    // clang-format on
}

vec3
SquareMatrix4::transform_vec(const vec3 &v) const {
    f32 x = v.x;
    f32 y = v.y;
    f32 z = v.z;

    f32 xt = mat[0][0] * x + mat[1][0] * y + mat[2][0] * z;
    f32 yt = mat[0][1] * x + mat[1][1] * y + mat[2][1] * z;
    f32 zt = mat[0][2] * x + mat[1][2] * y + mat[2][2] * z;

    return vec3(xt, yt, zt);
}

point3
SquareMatrix4::transform_point(const point3 &p) const {
    f32 x = p.x;
    f32 y = p.y;
    f32 z = p.z;

    f32 xt = mat[0][0] * x + mat[1][0] * y + mat[2][0] * z + mat[3][0];
    f32 yt = mat[0][1] * x + mat[1][1] * y + mat[2][1] * z + mat[3][1];
    f32 zt = mat[0][2] * x + mat[1][2] * y + mat[2][2] * z + mat[3][2];
    f32 wt = mat[0][3] * x + mat[1][3] * y + mat[2][3] * z + mat[3][3];

    if (wt == 1.f) {
        return point3(xt, yt, zt);
    } else {
        return point3(xt, yt, zt) / wt;
    }
}

SquareMatrix3::
SquareMatrix3(f32 m00, f32 m01, f32 m02, f32 m10, f32 m11, f32 m12, f32 m20, f32 m21,
              f32 m22) {
    // clang-format on
    mat[0][0] = m00;
    mat[0][1] = m01;
    mat[0][2] = m02;

    mat[1][0] = m10;
    mat[1][1] = m11;
    mat[1][2] = m12;

    mat[2][0] = m20;
    mat[2][1] = m21;
    mat[2][2] = m22;
}

SquareMatrix3
SquareMatrix3::from_columns(const tuple3 &c0, const tuple3 &c1, const tuple3 &c2) {
    // clang-format off
    return SquareMatrix3(c0.x,  c0.y,  c0.z,
                         c1.x,  c1.y,  c1.z,
                         c2.x,  c2.y,  c2.z);
    // clang-format on
}

SquareMatrix3
SquareMatrix3::from_elements(const Array<f32, 9> &columns) {
    // clang-format off
    return SquareMatrix3(columns[0],  columns[1],  columns[2],
                         columns[3],  columns[4],  columns[5],
                         columns[6],  columns[7],  columns[8]);
    // clang-format on
}

SquareMatrix3
SquareMatrix3::transpose() const {
    SquareMatrix3 new_mat{};
    new_mat.mat[0][0] = mat[0][0];
    new_mat.mat[0][1] = mat[1][0];
    new_mat.mat[0][2] = mat[2][0];

    new_mat.mat[1][0] = mat[0][1];
    new_mat.mat[1][1] = mat[1][1];
    new_mat.mat[1][2] = mat[2][1];

    new_mat.mat[2][0] = mat[0][2];
    new_mat.mat[2][1] = mat[1][2];
    new_mat.mat[2][2] = mat[2][2];

    return new_mat;
}

SquareMatrix3
SquareMatrix3::operator*(f32 mul) {
    auto new_mat = *this;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            new_mat.mat[i][j] *= mul;
        }
    }

    return new_mat;
}

SquareMatrix3
SquareMatrix3::compose(const SquareMatrix3 &other) const {
    return *this * other;
}

SquareMatrix3
SquareMatrix3::operator*(const SquareMatrix3 &other) const {
    SquareMatrix3 new_mat{};

    for (u32 i = 0; i < 3; i++) {
        for (u32 j = 0; j < 3; j++) {
            new_mat.mat[i][j] = mat[0][j] * other.mat[i][0] +
                                mat[1][j] * other.mat[i][1] + mat[2][j] * other.mat[i][2];
        }
    }

    return new_mat;
}

tuple3
SquareMatrix3::operator*(const tuple3 &other) const {
    tuple3 res = tuple3(0.f);

    for (u32 i = 0; i < 3; i++) {
        res[i] = mat[0][i] * other[0] + mat[1][i] * other[1] + mat[2][i] * other[2];
    }

    return res;
}

SquareMatrix3
SquareMatrix3::identity() {
    // clang-format off
    return SquareMatrix3(
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f
    );
    // clang-format on
}
