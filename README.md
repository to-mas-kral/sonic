# cuda-pt
CUDA Path-Tracing project for a Computer Graphics class.

Notes in Czech can be found in [docs/main.pdf](/docs/main.pdf).

# Gallery
All shown scenes were taken from [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/).

**Bathroom, 8K samples.**
![bathroom](/docs/bathroom-8k.png)

**Living room, 8K samples.**
![living-room](/docs/living-room-8k.png)

**Staircase, 8K samples.**
![living-room](/docs/staircase-8k.png)

**Staircase, 4K samples.**
![living-room](/docs/cornell-box-mis-4k.png)

# Techniques
- Path tracing
  - Path is extended randomly at each intersection based on the PDF of the BRDF.
  - At each path vertex, a light is randomly sampled and its contribution is added using MIS.
- Scenes are loaded using [Mitsuba's format](https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html).
  - This format uses XML for the markup and separate files for the geometry / textures.
  - Only a small subset of shapes / materials can actually be loaded...
- Ray-tracing using Nvidia's OptiX 8 API.
  - Overall a nice API to work with.
  - Can utilize hardware ray-tracing (although I don't have an Nvidia GPU with RTX to try it out...).
- Support for multiple shapes: triangles and spheres.
- Support for bitmap textures using CUDA's texture machinery.
- Environment map lighting.
  - Tried to apply MIS to environment lighting as well, but would need some help to get it to work properly.
- The renders are saved in HDR using the EXR file format.
