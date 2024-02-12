# cuda-pt
GPU Path-Tracing project for a Computer Graphics class.

Notes in Czech can be found in [docs/main.pdf](/docs/main.pdf).

# Techniques / features
- Path tracing
  - Path is extended randomly at each intersection based on the PDF of the BRDF.
  - At each path vertex, a light is randomly sampled and its contribution is weighted using MIS.
- Scenes are loaded using [Mitsuba's format](https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html).
  - This format uses XML for the markup and separate files for the geometry / textures.
  - Only a small subset of shapes / materials can actually be loaded...
- Ray-tracing using Nvidia's OptiX 8 API.
  - Overall a nice API to work with.
  - Can utilize hardware ray-tracing (although I don't have an Nvidia GPU with RTX to try it out...).
- Support for multiple shapes: triangles and spheres.
- A few BxDFs are implemented:
  - Diffuse BRDF
  - Smooth and rough conductor (metal) BRDFs
  - Perfectly dielectric BSDF
  - Smooth and rough plastic BRDFs
    - Simple analytic 2-layer material
- Support for bitmap textures using CUDA's texture machinery.
- Environment map lighting.
- The renders are saved in HDR using the EXR file format.

# Gallery
All shown scenes were taken from [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/).

**Coffee**
![coffee](/docs/coffee.png)

**Staircase**
![staircase](/docs/staircase.png)

**Spaceship**
![Spaceship](/docs/spaceship.png)

**Kitchen**
![kitchen](/docs/kitchen.png)

**Cornell Box**
![cornell-box](/docs/cornell-box.png)

**Veach MIS**
![veach-mis](/docs/veach_mis.png)
