#ifndef PT_VECMATH_H
#define PT_VECMATH_H

#include "../math/math_utils.h"
#include "../utils/basic_types.h"

// Vector3, Point3 and Tuple3 is inspired by PBRTv4...

template <template <typename> typename Child, typename T> struct Tuple2 {
    __host__ __device__ explicit Tuple2(T val) : x{val}, y{val} {}
    __host__ __device__
    Tuple2(T x, T y)
        : x{x}, y{y} {}

    __host__ __device__ T
    max_component() const {
        return max(x, y);
    }

    __host__ __device__ bool
    any_nan() {
        return isnan(x) || isnan(y);
    }

    __host__ __device__ Child<T>
    pow(T val) {
        return Child(std::pow(x, val), std::pow(y, val));
    }

    __host__ __device__ Child<T>
    operator*(T mul) const {
        return Child(x * mul, y * mul);
    }

    __host__ __device__ Child<T>
    operator*(const Child<T> &other) const {
        return Child(x * other.x, y * other.y);
    }

    __host__ __device__ Child<T> &
    operator*=(T div) {
        x *= div;
        y *= div;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator*=(const Child<T> &other) {
        x *= other.x;
        y *= other.y;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator/(T div) const {
        return Child(x / div, y / div);
    }

    __host__ __device__ Child<T>
    operator/(const Child<T> &other) const {
        return Child(x / other.x, y / other.y);
    }

    __host__ __device__ Child<T> &
    operator/=(T div) {
        x /= div;
        y /= div;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator/=(const Child<T> &other) {
        x /= other.x;
        y /= other.y;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator+(T add) const {
        return Child(x + add, y + add);
    }

    __host__ __device__ Child<T> &
    operator+=(T add) {
        x += add;
        y += add;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator+=(const Child<T> &other) {
        x += other.x;
        y += other.y;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator+(const Child<T> &other) const {
        return Child(x + other.x, y + other.y);
    }

    __host__ __device__ Child<T>
    operator-(T sub) const {
        return Child(x - sub, y - sub);
    }

    __host__ __device__ Child<T> &
    operator-=(T sub) {
        x -= sub;
        y -= sub;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator-=(const Child<T> &other) {
        x -= other.x;
        y -= other.y;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator-(const Child<T> &other) const {
        return Child(x - other.x, y - other.y);
    }

    __host__ __device__ Child<T>
    operator-() const {
        return Child(-x, -y);
    }

    __host__ __device__ friend Child<T>
    operator*(T mul, const Child<T> &vec) {
        return Child(vec.x * mul, vec.y * mul);
    }

    __host__ __device__ friend Child<T>
    operator+(T add, const Child<T> &vec) {
        return Child(vec.x + add, vec.y + add);
    }

    __host__ __device__ T
    operator[](u32 index) const {
        assert(index < 2);
        if (index == 0) {
            return x;
        } else {
            return y;
        }
    }

    __host__ __device__ T &
    operator[](u32 index) {
        assert(index < 2);
        if (index == 0) {
            return x;
        } else {
            return y;
        }
    }

    T x;
    T y;
};

template <typename T> struct Vector2 : public Tuple2<Vector2, T> {
    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;

    __host__ __device__ explicit Vector2(T val) : Tuple2<Vector2, T>(val) {}
    __host__ __device__
    Vector2(T x, T y)
        : Tuple2<Vector2, T>(x, y) {}

    __host__ __device__ T
    length_squared() const {
        return sqr(x) + sqr(y);
    }

    __host__ __device__ f32
    length() const {
        return sqrt(length_squared());
    }

    static __host__ __device__ T
    dot(const Vector2 &a, const Vector2 &b) {
        return a.x * b.x + a.y * b.y;
    }
};

struct NormalizedVector3;

template <template <typename> typename Child, typename T> struct Tuple3 {
    __host__ __device__ explicit Tuple3(T val) : x{val}, y{val}, z{val} {}
    __host__ __device__
    Tuple3(T x, T y, T z)
        : x{x}, y{y}, z{z} {}

    __device__ float3
    as_float3() const {
        return float3(x, y, z);
    }

    __host__ __device__ T
    max_component() const {
        return max(max(x, y), z);
    }

    __host__ __device__ bool
    any_nan() {
        return isnan(x) || isnan(y) || isnan(z);
    }

    __host__ __device__ Child<T>
    pow(T val) {
        return Child(std::pow(x, val), std::pow(y, val), std::pow(z, val));
    }

    __host__ __device__ Child<T>
    operator*(T mul) const {
        return Child<T>(x * mul, y * mul, z * mul);
    }

    __host__ __device__ Child<T>
    operator*(const Child<T> &other) const {
        return Child(x * other.x, y * other.y, z * other.z);
    }

    __host__ __device__ Child<T> &
    operator*=(T div) {
        x *= div;
        y *= div;
        z *= div;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator*=(const Child<T> &other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator/(T div) const {
        return Child<T>(x / div, y / div, z / div);
    }

    __host__ __device__ Child<T>
    operator/(const Child<T> &other) const {
        return Child<T>(x / other.x, y / other.y, z / other.z);
    }

    __host__ __device__ Child<T> &
    operator/=(T div) {
        x /= div;
        y /= div;
        z /= div;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator/=(const Child<T> &other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator+(T add) const {
        return Child(x + add, y + add, z + add);
    }

    __host__ __device__ Child<T> &
    operator+=(T add) {
        x += add;
        y += add;
        z += add;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator+=(const Child<T> &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator+(const Child<T> &other) const {
        return Child<T>(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Child<T>
    operator-(T sub) const {
        return Child(x - sub, y - sub, z - sub);
    }

    __host__ __device__ Child<T> &
    operator-=(T sub) {
        x -= sub;
        y -= sub;
        z -= sub;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T> &
    operator-=(const Child<T> &other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return static_cast<Child<T> &>(*this);
    }

    __host__ __device__ Child<T>
    operator-(const Child<T> &other) const {
        return Child(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Child<T>
    operator-() const {
        return Child<T>(-x, -y, -z);
    }

    __host__ __device__ friend Child<T>
    operator*(T mul, const Child<T> &vec) {
        return Child(vec.x * mul, vec.y * mul, vec.z * mul);
    }

    __host__ __device__ friend Child<T>
    operator+(T add, const Child<T> &vec) {
        return Child(vec.x + add, vec.y + add, vec.z + add);
    }

    __host__ __device__ T
    operator[](u32 index) const {
        assert(index < 3);
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }

    __host__ __device__ T &
    operator[](u32 index) {
        assert(index < 3);
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }

    T x;
    T y;
    T z;
};

template <typename T> struct Vector3 : Tuple3<Vector3, T> {
    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::z;

    __host__ __device__ explicit Vector3(T val) : Tuple3<Vector3, T>(val) {}
    __host__ __device__
    Vector3(T x, T y, T z)
        : Tuple3<Vector3, T>(x, y, z) {}

    __host__ __device__ T
    length_squared() const {
        return sqr(x) + sqr(y) + sqr(z);
    }

    __host__ __device__ f32
    length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ inline NormalizedVector3
    normalized();

    static __host__ __device__ f32
    dot(const Vector3 &a, const Vector3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static __host__ __device__ Vector3
    cross(const Vector3 &a, const Vector3 &b) {
        return Vector3((a.y * b.z - a.z * b.y), (a.z * b.x - a.x * b.z),
                       (a.x * b.y - a.y * b.x));
    }

    __host__ __device__ static Vector3
    reflect(const Vector3 &vec, const NormalizedVector3 &normal);
};

struct NormalizedVector3 : Vector3<f32> {
    __host__ __device__ explicit NormalizedVector3(f32 val) : Vector3(val) {
        assert(length() == 1.f);
    }
    __host__ __device__
    NormalizedVector3(f32 x, f32 y, f32 z)
        : Vector3(x, y, z) {
        assert(length() == 1.f);
    }

    __host__ __device__ f32
    length() const {
        return 1.f;
    }

    __host__ __device__ NormalizedVector3
    operator-() const {
        return NormalizedVector3(-x, -y, -z);
    }
};

template <typename T>
__host__ __device__ NormalizedVector3
Vector3<T>::normalized() {
    auto v = *this / length();
    return NormalizedVector3(v.x, v.y, v.z);
}

template <typename T>
__host__ __device__ inline Vector3<T>
Vector3<T>::reflect(const Vector3<T> &vec, const NormalizedVector3 &normal) {
    return -vec + normal * dot(normal, vec) * 2.f;
}

template <typename T = f32> struct Point3 : Tuple3<Point3, f32> {
    __host__ __device__ explicit Point3(f32 val) : Tuple3(val) {}
    __host__ __device__
    Point3(f32 x, f32 y, f32 z)
        : Tuple3(x, y, z) {}

    __host__ __device__ Vector3<f32>
    operator-(const Point3 &other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Point3
    operator-(const Vector3<f32> &other) const {
        return Point3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Point3
    operator+(const Vector3<f32> &other) const {
        return Point3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Point3
    operator+(const Point3 &other) const {
        return Point3(x + other.x, y + other.y, z + other.z);
    }
};

template <typename T> struct Tuple4 {
    __host__ __device__ explicit Tuple4<T>(T val) : x{val}, y{val}, z{val}, w{val} {}
    __host__ __device__
    Tuple4<T>(T x, T y, T z, T w)
        : x{x}, y{y}, z{z}, w{w} {}

    __host__ __device__ Tuple4
    operator*(T mul) const {
        return Tuple4(x * mul, y * mul, z * mul, w * mul);
    }

    __host__ __device__ Tuple4
    operator*(const Tuple4 &other) const {
        return Tuple4(x * other.x, y * other.y, z * other.z, w * other.w);
    }

    __host__ __device__ Tuple4 &
    operator*=(T mul) {
        x *= mul;
        y *= mul;
        z *= mul;
        w *= mul;
        return this;
    }

    __host__ __device__ Tuple4 &
    operator*=(const Tuple4 &other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        w *= other.w;
        return this;
    }

    __host__ __device__ Tuple4
    operator/(T div) const {
        return Tuple4(x / div, y / div, z / div, w / div);
    }

    __host__ __device__ Tuple4
    operator/(const Tuple4 &other) const {
        return Tuple4(x / other.x, y / other.y, z / other.z, w / other.w);
    }

    __host__ __device__ Tuple4 &
    operator/=(T div) {
        x /= div;
        y /= div;
        z /= div;
        w /= div;
        return this;
    }

    __host__ __device__ Tuple4 &
    operator/=(const Tuple4 &other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        w /= other.w;
        return this;
    }

    __host__ __device__ Tuple4
    operator+(T add) const {
        return Tuple4(x + add, y + add, z + add, w + add);
    }

    __host__ __device__ Tuple4 &
    operator+=(T add) {
        x += add;
        y += add;
        z += add;
        w += add;
        return this;
    }

    __host__ __device__ Tuple4 &
    operator+=(const Tuple4 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return this;
    }

    __host__ __device__ Tuple4
    operator+(const Tuple4 &other) const {
        return Tuple4(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    __host__ __device__ Tuple4
    operator-(T sub) const {
        return Tuple4(x - sub, y - sub, z - sub, w - sub);
    }

    __host__ __device__ Tuple4 &
    operator-=(T sub) {
        x -= sub;
        y -= sub;
        z -= sub;
        w -= sub;
        return this;
    }

    __host__ __device__ Tuple4 &
    operator-=(const Tuple4 &other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return this;
    }

    __host__ __device__ Tuple4
    operator-(const Tuple4 &other) const {
        return Tuple4(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    __host__ __device__ Tuple4
    operator-() const {
        return Tuple4(-x, -y, -z, -w);
    }

    __host__ __device__ friend Tuple4
    operator*(T mul, const Tuple4 &vec) {
        return Tuple4(vec.x * mul, vec.y * mul, vec.z * mul, vec.w * mul);
    }

    __host__ __device__ friend Tuple4
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

using tuple4 = Tuple4<f32>;

template <typename T>
__device__ __forceinline__ T
barycentric_interp(const vec3 &bar, const T &x, const T &y, const T &z) {
    return (bar.x * x) + (bar.y * y) + (bar.z * z);
}

__device__ __forceinline__ vec3
float3_to_vec(float3 f) {
    return vec3(f.x, f.y, f.z);
}
__device__ __forceinline__ float3
vec_to_float3(const vec3 &v) {
    return make_float3(v.x, v.y, v.z);
}

/// Taken from: Building an Orthonormal Basis, Revisited
/// Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani,
/// and Ryusuke Villemin
__device__ inline CTuple<vec3, vec3, vec3>
coordinate_system(vec3 v1) {
    f32 sign = copysign(1.f, v1.z);
    f32 a = -1.f / (sign + v1.z);
    f32 b = v1.x * v1.y * a;

    vec3 v2 = vec3(1.f + sign * sqr(v1.x) * a, sign * b, -sign * v1.x);
    vec3 v3 = norm_vec3(b, sign + sqr(v1.y) * a, -v1.y);

    return {v1, v2, v3};
}

/// Transforms dir into the basis of the normal
__device__ inline norm_vec3
orient_dir(const vec3 &dir, const norm_vec3 &normal) {
    auto [_, b1, b2] = coordinate_system(normal);
    norm_vec3 sample_dir = (b1 * dir.x + b2 * dir.y + normal * dir.z).normalized();

    if (vec3::dot(normal, sample_dir) < 0.f) {
        // TODO: it's usually really close to 0, unsure what to do here...
        sample_dir = -sample_dir;
    }

    return sample_dir;
}

#endif // PT_VECMATH_H
