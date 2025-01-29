#ifndef RENDERER_H
#define RENDERER_H

#include "integrator/integrator.h"
#include "integrator/mis_nee_integrator.h"
#include "integrator/path_guiding_integrator.h"
#include "scene/scene.h"
#include "utils/thread_pool.h"

class Renderer {
public:
    static Renderer
    init(Scene &&scene, const Settings &settings) {
        auto ictx = std::make_unique<IntegratorContext>(
            IntegratorContext::init(std::move(scene)));

        std::unique_ptr<Integrator> integrator;

        switch (settings.integrator_type) {
        case IntegratorType::Naive:
            integrator = std::make_unique<MisNeeIntegrator>(settings, ictx.get());
            break;
        case IntegratorType::MISNEE:
            integrator = std::make_unique<MisNeeIntegrator>(settings, ictx.get());
            break;
        case IntegratorType::PathGuiding:
            integrator = std::make_unique<PathGuidingIntegrator>(settings, ictx.get());
            break;
        default:
            panic("Erroneous integrator type.");
        }

        return Renderer(settings, std::move(ictx), std::move(integrator));
    }

    Renderer(const Settings &settings, std::unique_ptr<IntegratorContext> &&ictx,
             std::unique_ptr<Integrator> &&integrator)
        : m_settings(settings), m_ictx(std::move(ictx)),
          m_integrator(std::move(integrator)),
          m_thread_pool(m_ictx->attribs(), m_integrator.get(), m_settings) {}

    void
    compute_sample() {
        m_thread_pool.start_new_frame();
        m_integrator->next_sample();
    }

    Renderer(const Renderer &other) = delete;

    Renderer &
    operator=(const Renderer &other) = delete;

    Renderer(Renderer &&other) noexcept = delete;

    Renderer &
    operator=(Renderer &&other) noexcept = delete;

    ~Renderer() { m_thread_pool.stop(); }

    Scene &
    scene() const {
        return m_ictx->scene();
    }

    Framebuffer &
    framebuf() const {
        return m_ictx->framebuf();
    }

private:
    Settings m_settings;
    std::unique_ptr<IntegratorContext> m_ictx;
    std::unique_ptr<Integrator> m_integrator;
    ThreadPool m_thread_pool;
};

#endif // RENDERER_H
