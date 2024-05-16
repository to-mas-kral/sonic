
#include "texture.h"

f32
FloatScaleTexture::fetch(const vec2 &uv) const {
    return texture->fetch(uv) * scale;
}

Spectrum
SpectrumScaleTexture::fetch(const vec2 &uv) const {
    // TODO: figure out scaling Spectrum textures...
    return texture->fetch(uv) /** scale*/;
}
