#include "sphere_square_mapping.h"

/// Adapted from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD.
vec2
sphere_to_square(const norm_vec3 &arg_dir) {
    // Change coordinates from world-space to paper-space
    const auto dir = norm_vec3(arg_dir.x, -arg_dir.y, arg_dir.z);
    // assert(dir.is_normalized());
    assert(sqr(dir.length()) > 0.999 && sqr(dir.length()) < 1.001);
    const auto x = std::abs(dir.x);
    const auto y = std::abs(dir.y);
    const auto z = std::abs(dir.z);

    // Compute the radius r
    const auto r = safe_sqrt(1.f - z);
    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    const auto a = std::max(x, y);
    f32 b = std::min(x, y);
    if (a == 0.f) {
        b = 0.f;
    } else {
        b /= a;
    }

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    constexpr auto T1 = 0.406758566246788489601959989e-5f;
    constexpr auto T2 = 0.636226545274016134946890922156f;
    constexpr auto T3 = 0.61572017898280213493197203466e-2f;
    constexpr auto T4 = -0.247333733281268944196501420480f;
    constexpr auto T5 = 0.881770664775316294736387951347e-1f;
    constexpr auto T6 = 0.419038818029165735901852432784e-1f;
    constexpr auto T7 = -0.251390972343483509333252996350e-1f;

    auto phi = T6 + T7 * b;
    phi = T5 + phi * b;
    phi = T4 + phi * b;
    phi = T3 + phi * b;
    phi = T2 + phi * b;
    phi = T1 + phi * b;

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y) {
        phi = 1.f - phi;
    }

    // Find (u,v) based on (r,phi)
    auto v = phi * r;
    auto u = r - v;

    if (dir.z < 0.f) {
        // southern hemisphere -> mirror u,v
        auto tmp = u;
        u = v;
        v = tmp;

        u = 1.f - u;
        v = 1.f - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = std::copysign(u, dir.x);
    v = std::copysign(v, dir.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return vec2(0.5f * (u + 1.f), 0.5f * (v + 1.f));
}

/// Adapted from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD.
norm_vec3
square_to_sphere(const vec2 &uv) {
    // Transform p from [0,1] to [-1,1]
    f32 u = 2.f * uv.x - 1.f;
    f32 v = 2.f * uv.y - 1.f;

    // Store the sign bits of u,v for later use
    const f32 sign_u = std::copysign(1.f, u);
    const f32 sign_v = std::copysign(1.f, v);

    // Take the absolute values to move u,v to the first quadrant
    u = std::abs(u);
    v = std::abs(v);

    // Compute the radius based on the signed distance along the diagonal
    const f32 sd = 1.f - (u + v);
    f32 d = sd;
    d = std::abs(d);
    const f32 r = 1.f - d;

    // Comute phi*2/pi based on u, v and r (avoid div-by-zero if r=0)
    const f32 phi = r == 0.f ? 1.f : (v - u) / r + 1.f; // phi = [0,2)

    // Compute the z coordinate (flip sign based on signed distance)
    const f32 r2 = r * r;
    f32 z = 1.f - r2;
    z = std::copysign(z, sd);

    const f32 sin_theta = r * std::sqrt(2.f - r2);

    constexpr f32 S1 = 0.7853975892066955566406250000000000f;
    constexpr f32 S2 = -0.0807407423853874206542968750000000f;
    constexpr f32 S3 = 0.0024843954015523195266723632812500f;
    constexpr f32 S4 = -0.0000341485538228880614042282104492f;

    // Approximate sin/cos
    const f32 phi2 = phi * phi;
    f32 sp = S3 + S4 * phi2;
    sp = S2 + sp * phi2;
    sp = S1 + sp * phi2;
    f32 sin_phi = sp * phi;

    constexpr f32 C1 = 0.9999932952821962577665326692990000f;
    constexpr f32 C2 = -0.3083711259464511647371969120320000f;
    constexpr f32 C3 = 0.0157862649459062213825197189573000f;
    constexpr f32 C4 = -0.0002983708648233575495551227373110f;

    f32 cp = C3 + C4 * phi2;
    cp = C2 + cp * phi2;
    cp = C1 + cp * phi2;
    f32 cos_phi = cp;

    // Flip signs of sin/cos based on signs of u,v
    cos_phi = std::copysign(cos_phi, sign_u);
    sin_phi = std::copysign(sin_phi, sign_v);

    // Compute the x and y coordinates of the 3D vector
    const f32 x = sin_theta * cos_phi;
    const f32 y = sin_theta * sin_phi;

    return vec3(x, -y, z).normalized();
}
