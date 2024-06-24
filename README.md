# Sonic
Path-Tracing project for a Computer Graphics class.

# Techniques / features
- Path tracing:
  - Path is extended randomly at each intersection based on the BRDF.
  - At each path vertex, a light is randomly sampled and its contribution is weighted using MIS.
- Spectral path-tracing, mainly using the techniques from PBRTv4.
- Scenes are loaded using PBRTv4's format
  - Many shapes, texture types etc are not supported
- Ray-tracing using Intel's Embree library.
- Support for multiple shapes: triangles and spheres.
- A few BxDFs are implemented:
  - Diffuse BRDF
  - Smooth and rough conductor (metal) BRDFs
  - Smooth dielectric BSDF
  - Smooth and rough plastic BRDFs
    - Simple analytic 2-layer material
- Support for bitmap textures.
- Environment map lighting with importance sampling.
- The renders are saved in HDR using the EXR file format.

# Gallery

**WaterColor**

This scene was made by [Angelo Ferretti](https://www.lucydreams.it/).

![WaterColor](docs/watercolor-4096spp.png)

**Coffee**

All scenes shown below were taken from [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/).

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
