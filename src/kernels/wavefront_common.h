#ifndef PT_WAVEFRONT_COMMON_H
#define PT_WAVEFRONT_COMMON_H

#include <cuda/std/atomic>

#include "../geometry/wavefront_ray.h"
#include "../utils/numtypes.h"
#include "../utils/shared_vector.h"

class WavefrontState {
public:
    WavefrontState(const SceneAttribs &attribs) : ray_counter{0} {
        u64 size = attribs.resx * attribs.resy;
        rays = SharedVector<WavefrontRay>(size);
        rays.assume_all_init();
    }

    cuda::atomic<u32> ray_counter;
    SharedVector<WavefrontRay> rays;
};

#endif // PT_WAVEFRONT_COMMON_H
