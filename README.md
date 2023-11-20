# cuda-pt
CUDA Path-Tracing project for a Computer Graphics class.

Notes in Czech can be found in [docs/main.pdf](/docs/main.pdf).

# Progress report 1
- Naive path tracing
  - Importance sampling of the diffuse BRDF.
  - Lights are not explicitely sampled.
- Hardcoded Cornell Box scene.
- Bounding Volume Hierarchy acceleration structure built with Surface Area Heuristic (taken from PBRTv4).
- The image is split into blocks of 8x8 pixels and a single CUDA thread computes one sample for one pixel.
- Performance is not great - rendering the Cornell Box at 1024x1024 at 2048 samples took ~157 seconds.

![alt text](/docs/ptout.png)

# Progress report 2
- Loading scenes using Mitsuba's 3 format.
  - Only supports a subset of the Shapes, options... only the diffuse BRDF.
- Environment map lighting.

![alt text](/docs/house_render.png)

# Progress report 3
- Ray-tracing is now done using OptiX (can utilize hardware ray-tracing cores).
- Support for rendering with bitmap textures.

![alt text](/docs/textures.png)

# Progress report 4
- Monte Carlo integrator now uses multiple importance sampling (MIS).
- Spheres can now be added to the scene.

Cornell Box rendered at 4k samples
![alt text](/docs/cornell-box-mis-4k.png)
