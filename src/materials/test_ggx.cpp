#include "../utils/sampler.h"
#include "material.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

float
radicalInverse_VdC(u32 bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10f; // / 0x100000000
}

vec2
hammersley2d(uint i, uint N) {
    return vec2(float(i) / float(N), radicalInverse_VdC(i));
}

enum class Domain {
    Sphere,
    Hemisphere,
};

f64
integrate_spherical_function(Domain domain,
                             const std::function<f64(const norm_vec3 &)> &f) {
    const int ITER = 10000;
    f64 total_values = 0.f;
    for (int i = 0; i < ITER; i++) {
        vec2 sample = hammersley2d(i, ITER);

        norm_vec3 wi(0.f, 0.f, 1.f);
        if (domain == Domain::Sphere) {
            wi = sample_uniform_sphere(sample).normalized();
        } else {
            wi = sample_uniform_hemisphere(sample).normalized();
        }

        wi = norm_vec3(wi.x, wi.z, wi.y);
        if (domain == Domain::Hemisphere) {
            if (wi.y < 0.f) {
                wi.y = -wi.y;
            }
        }

        f64 value = f(wi);
        REQUIRE(value >= 0.f);
        REQUIRE(!std::isnan(value));
        REQUIRE(!std::isinf(value));

        total_values += value;
    }

    f64 area;
    if (domain == Domain::Sphere) {
        area = 4. * M_PI;
    } else {
        area = 2. * M_PI;
    }

    return area * total_values / ITER;
}

void
test_spherical_function(Domain domain, const std::function<f64(const norm_vec3 &)> &f) {
    constexpr int n_measurements = 1;
    std::array<f64, n_measurements> pdfs{};
    pdfs.fill(0.f);

    // TODO: this currently doesn't do anything as the LDS is the same every run...

    for (int i = 0; i < n_measurements; i++) {
        f64 res = integrate_spherical_function(domain, f);
        pdfs[i] = res;
    }

    f64 sum = 0.f;
    for (auto v : pdfs) {
        sum += v;
    }

    f64 mean = sum / n_measurements;

    f64 sum_squares = 0.f;
    for (auto v : pdfs) {
        sum_squares += sqr(mean - v);
    }

    f64 variance = sum_squares / n_measurements;
    f64 stdev = sqrt(variance);

    // TODO: more rigorous statistical test would be nice...
    REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(1., 0.01));
    REQUIRE_THAT(stdev, Catch::Matchers::WithinAbs(0., 0.02));
}

norm_vec3
generate_w() {
    std::random_device dev;
    std::mt19937 rng(dev());

    auto x1x = std::generate_canonical<f32, 23>(rng);
    auto x2y = std::generate_canonical<f32, 23>(rng);

    norm_vec3 wo = sample_uniform_hemisphere(vec2(x1x, x2y)).normalized();
    return norm_vec3(wo.x, wo.z, wo.y);
}

TEST_CASE("diffuse PDF", "[diffuse_pdf]") {
    Material mat = Material::make_diffuse(0);
    auto λ = SampledLambdas::new_mock();

    auto wo = generate_w();
    norm_vec3 normal = norm_vec3(0.f, 1.f, 0.f);

    test_spherical_function(Domain::Sphere, [&](const norm_vec3 &wi) {
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        f64 pdf = mat.pdf(sgeom, λ);
        return pdf;
    });
}

TEST_CASE("GGX VNDF PDF alpha=0.1", "[ggx_vndf_pdf_alpha_0_1]") {
    auto eta = Spectrum(RgbSpectrumUnbounded::make(tuple3(0.200438, 0.924033, 1.10221)));
    auto k = Spectrum(RgbSpectrumUnbounded::make(tuple3(3.91295, 2.45285, 2.14219)));
    Material mat = Material::make_rough_conductor(0.1f, eta, k);
    auto λ = SampledLambdas::new_mock();

    auto wo = generate_w();
    norm_vec3 normal = norm_vec3(0.f, 1.f, 0.f);

    test_spherical_function(Domain::Sphere, [&](const norm_vec3 &wi) {
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        f64 pdf = mat.pdf(sgeom, λ);
        return pdf;
    });
}

TEST_CASE("GGX VNDF PDF alpha=0.25", "[ggx_vndf_pdf_alpha_0_25]") {
    auto eta = Spectrum(RgbSpectrumUnbounded::make(tuple3(0.200438, 0.924033, 1.10221)));
    auto k = Spectrum(RgbSpectrumUnbounded::make(tuple3(3.91295, 2.45285, 2.14219)));
    Material mat = Material::make_rough_conductor(0.25f, eta, k);
    auto λ = SampledLambdas::new_mock();

    auto wo = generate_w();
    norm_vec3 normal = norm_vec3(0.f, 1.f, 0.f);

    test_spherical_function(Domain::Sphere, [&](const norm_vec3 &wi) {
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        f64 pdf = mat.pdf(sgeom, λ);
        return pdf;
    });
}

TEST_CASE("GGX VNDF PDF alpha=0.50", "[ggx_vndf_pdf_alpha_0_5]") {
    auto eta = Spectrum(RgbSpectrumUnbounded::make(tuple3(0.200438, 0.924033, 1.10221)));
    auto k = Spectrum(RgbSpectrumUnbounded::make(tuple3(3.91295, 2.45285, 2.14219)));
    Material mat = Material::make_rough_conductor(0.5f, eta, k);
    auto λ = SampledLambdas::new_mock();

    auto wo = generate_w();
    norm_vec3 normal = norm_vec3(0.f, 1.f, 0.f);

    test_spherical_function(Domain::Sphere, [&](const norm_vec3 &wi) {
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        f64 pdf = mat.pdf(sgeom, λ);
        return pdf;
    });
}

TEST_CASE("GGX VNDF PDF alpha=0.75", "[ggx_vndf_pdf_alpha_0_75]") {
    auto eta = Spectrum(RgbSpectrumUnbounded::make(tuple3(0.200438, 0.924033, 1.10221)));
    auto k = Spectrum(RgbSpectrumUnbounded::make(tuple3(3.91295, 2.45285, 2.14219)));
    Material mat = Material::make_rough_conductor(0.75f, eta, k);
    auto λ = SampledLambdas::new_mock();

    auto wo = generate_w();
    norm_vec3 normal = norm_vec3(0.f, 1.f, 0.f);

    test_spherical_function(Domain::Sphere, [&](const norm_vec3 &wi) {
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        f64 pdf = mat.pdf(sgeom, λ);
        return pdf;
    });
}

TEST_CASE("GGX VNDF PDF alpha=1", "[ggx_vndf_pdf_alpha_1]") {
    auto eta = Spectrum(RgbSpectrumUnbounded::make(tuple3(0.200438, 0.924033, 1.10221)));
    auto k = Spectrum(RgbSpectrumUnbounded::make(tuple3(3.91295, 2.45285, 2.14219)));
    Material mat = Material::make_rough_conductor(1.f, eta, k);
    auto λ = SampledLambdas::new_mock();

    auto wo = generate_w();
    norm_vec3 normal = norm_vec3(0.f, 1.f, 0.f);

    test_spherical_function(Domain::Sphere, [&](const norm_vec3 &wi) {
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        f64 pdf = mat.pdf(sgeom, λ);
        return pdf;
    });
}
