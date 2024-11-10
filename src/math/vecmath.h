#ifndef PT_VECMATH_H
#define PT_VECMATH_H

#include "../math/math_utils.h"
#include "../utils/basic_types.h"

#include <algorithm>
#include <cassert>

// Vector3, Point3 and Tuple3 is inspired by PBRTv4...

template <template <typename> typename Child, typename T> struct Tuple2Base {
    explicit
    Tuple2Base(T val)
        : x{val}, y{val} {}

    Tuple2Base(T x, T y) : x{x}, y{y} {}

    Tuple2Base() : x{0.f}, y{0.f} {}

    T
    max_component() const {
        return std::max(x, y);
    }

    bool
    any_nan() {
        return std::isnan(x) || std::isnan(y);
    }

    Child<T>
    pow(T val) {
        return Child(std::pow(x, val), std::pow(y, val));
    }

    Child<T>
    operator*(T mul) const {
        return Child(x * mul, y * mul);
    }

    Child<T>
    operator*(const Child<T> &other) const {
        return Child(x * other.x, y * other.y);
    }

    Child<T> &
    operator*=(T div) {
        x *= div;
        y *= div;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator*=(const Child<T> &other) {
        x *= other.x;
        y *= other.y;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator/(T div) const {
        return Child(x / div, y / div);
    }

    Child<T>
    operator/(const Child<T> &other) const {
        return Child(x / other.x, y / other.y);
    }

    Child<T> &
    operator/=(T div) {
        x /= div;
        y /= div;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator/=(const Child<T> &other) {
        x /= other.x;
        y /= other.y;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator+(T add) const {
        return Child(x + add, y + add);
    }

    Child<T> &
    operator+=(T add) {
        x += add;
        y += add;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator+=(const Child<T> &other) {
        x += other.x;
        y += other.y;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator+(const Child<T> &other) const {
        return Child(x + other.x, y + other.y);
    }

    Child<T>
    operator-(T sub) const {
        return Child(x - sub, y - sub);
    }

    Child<T> &
    operator-=(T sub) {
        x -= sub;
        y -= sub;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator-=(const Child<T> &other) {
        x -= other.x;
        y -= other.y;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator-(const Child<T> &other) const {
        return Child(x - other.x, y - other.y);
    }

    Child<T>
    operator-() const {
        return Child(-x, -y);
    }

    friend Child<T>
    operator*(T mul, const Child<T> &vec) {
        return Child(vec.x * mul, vec.y * mul);
    }

    friend Child<T>
    operator+(T add, const Child<T> &vec) {
        return Child(vec.x + add, vec.y + add);
    }

    T
    operator[](const u32 index) const {
        assert(index < 2);
        if (index == 0) {
            return x;
        } else {
            return y;
        }
    }

    T &
    operator[](const u32 index) {
        assert(index < 2);
        if (index == 0) {
            return x;
        } else {
            return y;
        }
    }

    bool
    operator==(const Tuple2Base &other) const {
        return x == other.x && y == other.y;
    }

    bool
    approx_eq(const Tuple2Base &other) const {
        // This comparison could be better but works for now...
        constexpr f32 EPS = 0.00001;
        return std::abs(x - other.x) < EPS && std::abs(y - other.y) < EPS;
    }

    T x;
    T y;
};

template <typename T> struct Vector2 : Tuple2Base<Vector2, T> {
    using Tuple2Base<Vector2, T>::x;
    using Tuple2Base<Vector2, T>::y;

    explicit
    Vector2(T val)
        : Tuple2Base<Vector2, T>(val) {}

    Vector2(T x, T y) : Tuple2Base<Vector2, T>(x, y) {}

    Vector2() : Tuple2Base<Vector2, T>() {}

    T
    length_squared() const {
        return sqr(x) + sqr(y);
    }

    f32
    length() const {
        return sqrt(length_squared());
    }

    static T
    dot(const Vector2 &a, const Vector2 &b) {
        return a.x * b.x + a.y * b.y;
    }
};

struct NormalizedVector3;

template <template <typename> typename Child, typename T> struct Tuple3Base {
    explicit
    Tuple3Base(T val)
        : x{val}, y{val}, z{val} {}

    Tuple3Base(T x, T y, T z) : x{x}, y{y}, z{z} {}

    Tuple3Base() : x{0.f}, y{0.f}, z{0.f} {}

    T
    max_component() const {
        return std::max(std::max(x, y), z);
    }

    Child<T>
    clamp_negative() {
        return Child<T>(std::max(x, 0.f), std::max(y, 0.f), std::max(z, 0.f));
    }

    bool
    any_nan() {
        return std::isnan(x) || std::isnan(y) || std::isnan(z);
    }

    Child<T>
    pow(T val) {
        return Child(std::pow(x, val), std::pow(y, val), std::pow(z, val));
    }

    Child<T>
    operator*(T mul) const {
        return Child<T>(x * mul, y * mul, z * mul);
    }

    Child<T>
    operator*(const Child<T> &other) const {
        return Child(x * other.x, y * other.y, z * other.z);
    }

    Child<T> &
    operator*=(T div) {
        x *= div;
        y *= div;
        z *= div;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator*=(const Child<T> &other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator/(T div) const {
        return Child<T>(x / div, y / div, z / div);
    }

    Child<T>
    operator/(const Child<T> &other) const {
        return Child<T>(x / other.x, y / other.y, z / other.z);
    }

    Child<T> &
    operator/=(T div) {
        x /= div;
        y /= div;
        z /= div;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator/=(const Child<T> &other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator+(T add) const {
        return Child(x + add, y + add, z + add);
    }

    Child<T> &
    operator+=(T add) {
        x += add;
        y += add;
        z += add;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator+=(const Child<T> &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator+(const Child<T> &other) const {
        return Child<T>(x + other.x, y + other.y, z + other.z);
    }

    Child<T>
    operator-(T sub) const {
        return Child(x - sub, y - sub, z - sub);
    }

    Child<T> &
    operator-=(T sub) {
        x -= sub;
        y -= sub;
        z -= sub;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> &
    operator-=(const Child<T> &other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return static_cast<Child<T> &>(*this);
    }

    Child<T>
    operator-(const Child<T> &other) const {
        return Child(x - other.x, y - other.y, z - other.z);
    }

    Child<T>
    operator-() const {
        return Child<T>(-x, -y, -z);
    }

    friend Child<T>
    operator*(T mul, const Child<T> &vec) {
        return Child(vec.x * mul, vec.y * mul, vec.z * mul);
    }

    friend Child<T>
    operator+(T add, const Child<T> &vec) {
        return Child(vec.x + add, vec.y + add, vec.z + add);
    }

    T
    operator[](const u32 index) const {
        assert(index < 3);
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }

    T &
    operator[](const u32 index) {
        assert(index < 3);
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }

    bool
    operator==(const Tuple3Base &other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool
    approx_eq(const Tuple3Base &other) const {
        // This comparison could be better, but works for now...
        constexpr f32 EPS = 0.00001;
        return std::abs(x - other.x) < EPS && std::abs(y - other.y) < EPS &&
               std::abs(z - other.z) < EPS;
    }

    T x;
    T y;
    T z;
};

template <typename T> struct Vector3 : Tuple3Base<Vector3, T> {
    using Tuple3Base<Vector3, T>::x;
    using Tuple3Base<Vector3, T>::y;
    using Tuple3Base<Vector3, T>::z;

    explicit
    Vector3(T val)
        : Tuple3Base<Vector3, T>(val) {}

    Vector3(T x, T y, T z) : Tuple3Base<Vector3, T>(x, y, z) {}

    Vector3() : Tuple3Base<Vector3, T>(0.f) {}

    T
    length_squared() const {
        return sqr(x) + sqr(y) + sqr(z);
    }

    f32
    length() const {
        return sqrt(length_squared());
    }

    NormalizedVector3
    normalized() const;

    static f32
    dot(const Vector3 &a, const Vector3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static Vector3
    cross(const Vector3 &a, const Vector3 &b) {
        return Vector3((a.y * b.z - a.z * b.y), (a.z * b.x - a.x * b.z),
                       (a.x * b.y - a.y * b.x));
    }

    static Vector3
    reflect(const Vector3 &vec, const NormalizedVector3 &normal);
};

template <typename T> struct Tuple3 : Tuple3Base<Tuple3, T> {
    explicit
    Tuple3(f32 val)
        : Tuple3Base<Tuple3, T>(val) {}

    Tuple3(T x, T y, T z) : Tuple3Base<Tuple3, T>(x, y, z) {}
};

struct NormalizedVector3 : Vector3<f32> {
    NormalizedVector3() : Vector3(0.f, 0.f, 1.f) {}

    explicit
    NormalizedVector3(const f32 val)
        : Vector3(val) {
        assert(std::abs(Vector3::length() - 1.f) < 0.0001f);
    }

    NormalizedVector3(const f32 x, const f32 y, const f32 z) : Vector3(x, y, z) {
        assert(std::abs(Vector3::length() - 1.f) < 0.0001f);
    }

    f32
    length() const {
        return 1.f;
    }

    static NormalizedVector3
    halfway(const NormalizedVector3 &a, const NormalizedVector3 &b) {
        return (a + b).normalized();
    }

    NormalizedVector3
    operator-() const {
        return NormalizedVector3(-x, -y, -z);
    }
};

template <typename T>
NormalizedVector3
Vector3<T>::normalized() const {
    auto v = *this / length();
    return NormalizedVector3(v.x, v.y, v.z);
}

template <typename T>
Vector3<T>
Vector3<T>::reflect(const Vector3 &vec, const NormalizedVector3 &normal) {
    return -vec + normal * dot(normal, vec) * 2.f;
}

template <typename T = f32> struct Point3 : Tuple3Base<Point3, f32> {
    explicit
    Point3(const f32 val)
        : Tuple3Base(val) {}

    Point3(const f32 x, const f32 y, const f32 z) : Tuple3Base(x, y, z) {}

    explicit
    Point3(Vector3<T> vec)
        : Tuple3Base(vec.x, vec.y, vec.z) {}

    Point3() : Tuple3Base() {}

    Vector3<f32>
    operator-(const Point3 &other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    Point3
    operator-(const Vector3<f32> &other) const {
        return Point3(x - other.x, y - other.y, z - other.z);
    }

    Point3
    operator+(const Vector3<f32> &other) const {
        return Point3(x + other.x, y + other.y, z + other.z);
    }

    Point3
    operator+(const Point3 &other) const {
        return Point3(x + other.x, y + other.y, z + other.z);
    }
};

template <typename T> struct Tuple4 {
    explicit Tuple4<T>(T val) : x{val}, y{val}, z{val}, w{val} {}

    Tuple4<T>(T x, T y, T z, T w) : x{x}, y{y}, z{z}, w{w} {}

    Tuple4
    operator*(T mul) const {
        return Tuple4(x * mul, y * mul, z * mul, w * mul);
    }

    Tuple4
    operator*(const Tuple4 &other) const {
        return Tuple4(x * other.x, y * other.y, z * other.z, w * other.w);
    }

    Tuple4 &
    operator*=(T mul) {
        x *= mul;
        y *= mul;
        z *= mul;
        w *= mul;
        return this;
    }

    Tuple4 &
    operator*=(const Tuple4 &other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        w *= other.w;
        return this;
    }

    Tuple4
    operator/(T div) const {
        return Tuple4(x / div, y / div, z / div, w / div);
    }

    Tuple4
    operator/(const Tuple4 &other) const {
        return Tuple4(x / other.x, y / other.y, z / other.z, w / other.w);
    }

    Tuple4 &
    operator/=(T div) {
        x /= div;
        y /= div;
        z /= div;
        w /= div;
        return this;
    }

    Tuple4 &
    operator/=(const Tuple4 &other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        w /= other.w;
        return this;
    }

    Tuple4
    operator+(T add) const {
        return Tuple4(x + add, y + add, z + add, w + add);
    }

    Tuple4 &
    operator+=(T add) {
        x += add;
        y += add;
        z += add;
        w += add;
        return this;
    }

    Tuple4 &
    operator+=(const Tuple4 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return this;
    }

    Tuple4
    operator+(const Tuple4 &other) const {
        return Tuple4(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    Tuple4
    operator-(T sub) const {
        return Tuple4(x - sub, y - sub, z - sub, w - sub);
    }

    Tuple4 &
    operator-=(T sub) {
        x -= sub;
        y -= sub;
        z -= sub;
        w -= sub;
        return this;
    }

    Tuple4 &
    operator-=(const Tuple4 &other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return this;
    }

    Tuple4
    operator-(const Tuple4 &other) const {
        return Tuple4(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    Tuple4
    operator-() const {
        return Tuple4(-x, -y, -z, -w);
    }

    friend Tuple4
    operator*(T mul, const Tuple4 &vec) {
        return Tuple4(vec.x * mul, vec.y * mul, vec.z * mul, vec.w * mul);
    }

    friend Tuple4
    operator+(T add, const Tuple4 &vec) {
        return Tuple4(vec.x + add, vec.y + add, vec.z + add, vec.w + add);
    }

    T x;
    T y;
    T z;
    T w;
};

using vec2 = Vector2<f32>;
using vec3 = Vector3<f32>;

using norm_vec3 = NormalizedVector3;
using point3 = Point3<>;

using uvec2 = Vector2<u32>;
using uvec3 = Vector3<u32>;

using ivec2 = Vector2<i32>;
using ivec3 = Vector3<i32>;

// TODO: refactor some variables from vec3 to tuple3
using tuple3 = Tuple3<f32>;
using tuple4 = Tuple4<f32>;

template <typename T>
T
barycentric_interp(const vec3 &bar, const T &x, const T &y, const T &z) {
    return (bar.x * x) + (bar.y * y) + (bar.z * z);
}

#endif // PT_VECMATH_H
