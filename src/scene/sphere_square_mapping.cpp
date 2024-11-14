#include "sphere_square_mapping.h"

/// Adapted from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD.
vec2
sphere_to_square(const norm_vec3 &arg_dir) {
    const auto dir = vec3(arg_dir.x, arg_dir.y, arg_dir.z);
    const auto x = std::abs(dir.x);
    const auto y = std::abs(dir.y);
    const auto z = std::abs(dir.z);

    // Compute the radius r
    const auto r = safe_sqrt(1.F - z);
    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    const auto a = std::max(x, y);
    f32 b = std::min(x, y);
    if (a == 0.F) {
        b = 0.F;
    } else {
        b /= a;
    }

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    constexpr auto T1 = 0.406758566246788489601959989e-5F;
    constexpr auto T2 = 0.636226545274016134946890922156F;
    constexpr auto T3 = 0.61572017898280213493197203466e-2F;
    constexpr auto T4 = -0.247333733281268944196501420480F;
    constexpr auto T5 = 0.881770664775316294736387951347e-1F;
    constexpr auto T6 = 0.419038818029165735901852432784e-1F;
    constexpr auto T7 = -0.251390972343483509333252996350e-1F;

    auto phi = std::fma(b, T7, T6);
    phi = std::fma(phi, b, T5);
    phi = std::fma(phi, b, T4);
    phi = std::fma(phi, b, T3);
    phi = std::fma(phi, b, T2);
    phi = std::fma(phi, b, T1);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y) {
        phi = 1.F - phi;
    }

    // Find (u,v) based on (r,phi)
    auto v = phi * r;
    auto u = r - v;

    if (dir.z < 0.F) {
        // southern hemisphere -> mirror u,v
        auto tmp = u;
        u = v;
        v = tmp;

        u = 1.F - u;
        v = 1.F - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = std::copysign(u, dir.x);
    v = std::copysign(v, dir.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return vec2(0.5F * (u + 1.F), 0.5F * (v + 1.F));
}

/// Taken from PBRTv4. The code (as is) from Clarberg doesn't have as much precision as is
/// needed for envmap sampling.
norm_vec3
square_to_sphere(const vec2 &xy) {
    assert(xy.x >= 0.F && xy.x <= 1.F);
    assert(xy.y >= 0.F && xy.y <= 1.F);

    const f32 u = 2 * xy.x - 1;
    const f32 v = 2 * xy.y - 1;
    const f32 up = std::abs(u);
    const f32 vp = std::abs(v);

    const f32 signed_distance = 1.F - (up + vp);
    const f32 d = std::abs(signed_distance);
    const f32 r = 1.F - d;

    const f32 phi = (r == 0.F ? 1.F : (vp - up) / r + 1.F) * M_PIf / 4.F;

    const f32 z = std::copysign(1.F - sqr(r), signed_distance);

    const f32 cos_phi = std::copysign(std::cos(phi), u);
    const f32 sin_phi = std::copysign(std::sin(phi), v);
    return norm_vec3(cos_phi * r * safe_sqrt(2.F - sqr(r)),
                     sin_phi * r * safe_sqrt(2.F - sqr(r)), z);
}
