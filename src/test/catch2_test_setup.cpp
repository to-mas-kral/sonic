#ifndef CATCH2_LISTENER_H
#define CATCH2_LISTENER_H

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <spdlog/spdlog.h>

#include "../math/samplers/halton_sampler.h"
#include "../scene/image.h"
#include "test_globals.h"

class TestRunListener final : public Catch::EventListenerBase {
public:
    using EventListenerBase::EventListenerBase;

    void
    testRunStarting(Catch::TestRunInfo const & /*testRunInfo*/) override {
        sonic::init_halton_permutations();
        sonic::ENVMAP_BIG_TEST_IMAGE =
            Image::from_filepath("../resources/test/abandoned_tank_farm_03_4k.exr");
    }
};

CATCH_REGISTER_LISTENER(TestRunListener)

#endif // CATCH2_LISTENER_H
