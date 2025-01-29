#ifndef PT_INTEGRATOR_CONTEXT_H
#define PT_INTEGRATOR_CONTEXT_H

#include "camera.h"
#include "embree_accel.h"
#include "framebuffer.h"
#include "scene/scene.h"
#include "utils/basic_types.h"

/// IntegratorContext is a collection of data needed for the integrators to do their job.
class IntegratorContext {
public:
    static IntegratorContext
    init(Scene &&scene) {
        const f32 aspect = static_cast<f32>(scene.attribs.film.resx) /
                           static_cast<f32>(scene.attribs.film.resy);
        const auto cam = Camera(scene.attribs.camera.fov, aspect);
        auto framebuf = Framebuffer(scene.attribs.film.resx, scene.attribs.film.resy);

        spdlog::info("Creating Embree acceleration structure");
        return IntegratorContext(std::move(scene), std::move(framebuf), cam);
    }

    IntegratorContext(const IntegratorContext &other) = delete;

    IntegratorContext &
    operator=(const IntegratorContext &other) = delete;

    // Having to watch the pointer inside EmbreeAccel is brittle...
    // TODO: refactor EmbreeAccel to be inside the scene, which is very annoying though
    // because of circular includes.
    // It's also because of the design of the ThreadPool, which needs refactoring... see
    // the init order in renderer.h...
    IntegratorContext(IntegratorContext &&other) noexcept
        : m_scene(std::move(other.m_scene)), m_cam(other.m_cam),
          m_framebuf(std::move(other.m_framebuf)),
          m_embree_accel(std::move(other.m_embree_accel)) {
        m_embree_accel.set_scene(m_scene);
    }

    IntegratorContext &
    operator=(IntegratorContext &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        m_scene = std::move(other.m_scene);
        m_cam = other.m_cam;
        m_framebuf = std::move(other.m_framebuf);
        m_embree_accel = std::move(other.m_embree_accel);

        m_embree_accel.set_scene(m_scene);

        return *this;
    }

    ~IntegratorContext() = default;

    Scene &
    scene() {
        return m_scene;
    }

    Camera &
    cam() {
        return m_cam;
    }

    Framebuffer &
    framebuf() {
        return m_framebuf;
    }

    SceneAttribs &
    attribs() {
        return m_scene.attribs;
    }

    EmbreeAccel &
    accel() {
        return m_embree_accel;
    }

private:
    explicit IntegratorContext(Scene &&scene, Framebuffer &&framebuf, const Camera &cam)
        : m_scene{std::move(scene)}, m_cam(cam), m_framebuf(std::move(framebuf)),
          m_embree_accel(m_scene) {}

    Scene m_scene;
    Camera m_cam;
    Framebuffer m_framebuf;
    EmbreeAccel m_embree_accel;
};

#endif // PT_INTEGRATOR_CONTEXT_H
