#ifndef SHADING_FRAME_H
#define SHADING_FRAME_H

#include "../math/coordinate_system.h"
#include "../math/vecmath.h"

class ShadingFrameIncomplete {
public:
    explicit
    ShadingFrameIncomplete(const norm_vec3 &z)
        : coord_sys{CoordinateSystem(z)} {}

    vec3
    to_local(const norm_vec3 &input) const {
        return coord_sys.to_local(input);
    }

    vec3
    from_local(const norm_vec3 &input) const {
        return coord_sys.from_local(input);
    }

    static f32
    cos_theta(const norm_vec3 &w) {
        return w.z;
    }

    static f32
    cos_2_theta(const norm_vec3 &w) {
        return sqr(w.z);
    }

    static f32
    sin_2_theta(const norm_vec3 &w) {
        return std::max(0.F, 1.F - sqr(cos_theta(w)));
    }

    static f32
    sin_theta(const norm_vec3 &w) {
        return sqrtf(sin_2_theta(w));
    }

    static norm_vec3
    reflect(const norm_vec3 &w) {
        return {-w.x, -w.y, w.z};
    }

    static bool
    same_hemisphere(const norm_vec3 &a, const norm_vec3 &b) {
        return a.z * b.z > 0;
    }

private:
    CoordinateSystem coord_sys;
};

class ShadingFrame {
public:
    /// wi and wo are assumed to not be in local space
    ShadingFrame(const norm_vec3 &normal, const norm_vec3 &p_wi, const norm_vec3 &p_wo)
        : m_sframe{ShadingFrameIncomplete(normal)},
          m_wi{m_sframe.to_local(p_wi).normalized()},
          m_wo{m_sframe.to_local(p_wo).normalized()},
          m_h{m_sframe.to_local(norm_vec3::halfway(p_wi, p_wo)).normalized()} {}

    /// wi and wo have to be in local space !
    ShadingFrame(const ShadingFrameIncomplete &sframe, const norm_vec3 &wi,
                 const norm_vec3 &wo, const bool refracts = false)
        : m_sframe{sframe}, m_wi{wi}, m_wo{wo}, m_h{norm_vec3::halfway(wi, wo)} {

        assert(nowo() >= 0.F);
        if (!refracts) {
            assert(nowi() >= 0.F);
        } else {
            assert(nowi() <= 0.F);
        }
    }

    vec3
    to_local(const norm_vec3 &input) const {
        return m_sframe.to_local(input);
    }

    vec3

    from_local(const norm_vec3 &input) const {
        return m_sframe.from_local(input);
    }

    bool
    is_degenerate() const {
        return nowi() == 0.F || nowo() == 0.F;
    }

    static f32
    cos_theta(const norm_vec3 &w) {
        return w.z;
    }

    static f32
    cos_2_theta(const norm_vec3 &w) {
        return sqr(w.z);
    }

    static f32
    sin_2_theta(const norm_vec3 &w) {
        return std::max(0.F, 1.F - sqr(cos_theta(w)));
    }

    static f32
    sin_theta(const norm_vec3 &w) {
        return sqrtf(sin_2_theta(w));
    }

    static norm_vec3
    reflect(const norm_vec3 &w) {
        return {-w.x, -w.y, w.z};
    }

    f32
    howo() const {
        return vec3::dot(m_h, m_wo);
    }

    f32
    nowo() const {
        return m_wo.z;
    }

    f32
    nowi() const {
        return m_wi.z;
    }

    f32
    abs_nowi() const {
        return std::abs(m_wi.z);
    }

    f32
    noh() const {
        return m_h.z;
    }

    /// Omega-in
    norm_vec3
    wi() const {
        return m_wi;
    }

    /// Omega-out
    norm_vec3
    wo() const {
        return m_wo;
    }

    /// The half-vector
    norm_vec3
    h() const {
        return m_h;
    }

    static bool
    same_hemisphere(const norm_vec3 &a, const norm_vec3 &b) {
        return a.z * b.z > 0;
    }

private:
    ShadingFrameIncomplete m_sframe;

    norm_vec3 m_wi;
    norm_vec3 m_wo;
    norm_vec3 m_h;
};

#endif // SHADING_FRAME_H
