# cuda-pt
CUDA Path-Tracing project for a Computer Graphics class.

Notes in Czech can be found in [docs/main.pdf](/docs/main.pdf).

# Progress report 1
- Naive path tracing
  - Importance sampling of the diffuse BRDF.
  - Lights are not explicitely sampled.
- Hardcoded Cornell Box scene.
- No acceleration structure.
- The image is split up into blocks of 8x8 pixels and a single CUDA thread computes one sample for one pixel.
- Performance is not great - rendering the Cornell Box at 1024x1024 at 2048 samples took ~157 seconds.
  - Interestingly, the performance issues seem to be exactly what Wavefront path-tracing aims to eliminate: thread divergence and high register consumption of the megakernel.

![alt text](/docs/ptout.png)
