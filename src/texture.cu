
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "color/rgb_spectrum.h"
#include "texture.h"

void
transform_rgb_to_spectrum(f32 *pixels, i32 width, i32 height, i32 num_channels) {
    for (i32 p = 0; p < width * height; p++) {
        f32 r = pixels[4 * p + 0];
        f32 g = pixels[4 * p + 1];
        f32 b = pixels[4 * p + 2];

        RgbSpectrum spectrum = RgbSpectrum::make(tuple3(r, g, b));
        pixels[4 * p + 0] = spectrum.sigmoid_coeff.x;
        pixels[4 * p + 1] = spectrum.sigmoid_coeff.y;
        pixels[4 * p + 2] = spectrum.sigmoid_coeff.z;
    }
}

Texture
load_exr_texture(const std::string &texture_path, bool is_rgb) {
    f32 *pixels = nullptr;
    cudaArray_t texture_storage_array = nullptr;
    cudaTextureObject_t tex_obj = 0;
    i32 width = 0;
    i32 height = 0;
    constexpr i32 num_channels = 4;

    const char *err = nullptr;
    i32 ret = LoadEXR(&pixels, &width, &height, texture_path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            throw std::runtime_error(fmt::format("EXR loading error: {}", err));
            FreeEXRErrorMessage(err);
        }
    }

    if (is_rgb) {
        transform_rgb_to_spectrum(pixels, width, height, num_channels);
    }

    cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    CUDA_CHECK(cudaMallocArray(&texture_storage_array, &channel_desc, width, height))

    size_t spitch = width * num_channels * sizeof(f32);
    CUDA_CHECK(cudaMemcpy2DToArray(texture_storage_array, 0, 0, pixels, spitch,
                                   width * num_channels * sizeof(f32), height,
                                   cudaMemcpyHostToDevice))

    free(pixels);

    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = texture_storage_array;

    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    tex_desc.sRGB = 0;

    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr))

    return Texture(tex_obj, texture_storage_array, width, height);
}

Texture
load_other_format_texture(const std::string &texture_path, bool is_rgb) {
    cudaArray_t texture_storage_array = nullptr;
    cudaTextureObject_t tex_obj = 0;
    i32 width = 0;
    i32 height = 0;
    i32 num_channels = 0;

    cudaTextureReadMode read_mode{};

    u8 *pixels = stbi_load(texture_path.c_str(), &width, &height, &num_channels, 0);

    if (is_rgb) {
        read_mode = cudaReadModeElementType;
        cudaChannelFormatDesc channel_desc =
            cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        if (num_channels < 3 || num_channels > 4) {
            throw std::runtime_error("Invalid RGB texture channel count");
        }

        constexpr int num_channels_converted = 4;

        // Transform u8s to f32s
        auto pixels_f32 = std::vector<f32>(width * height * num_channels_converted);
        for (i32 p = 0; p < width * height; p++) {
            pixels_f32[4 * p + 0] =
                static_cast<f32>(pixels[num_channels * p + 0]) / 255.f;
            pixels_f32[4 * p + 1] =
                static_cast<f32>(pixels[num_channels * p + 1]) / 255.f;
            pixels_f32[4 * p + 2] =
                static_cast<f32>(pixels[num_channels * p + 2]) / 255.f;
            pixels_f32[4 * p + 3] = 1.f;
        }

        stbi_image_free(pixels);
        transform_rgb_to_spectrum(pixels_f32.data(), width, height,
                                  num_channels_converted);

        CUDA_CHECK(cudaMallocArray(&texture_storage_array, &channel_desc, width, height))

        size_t spitch = width * num_channels_converted * sizeof(f32);
        CUDA_CHECK(cudaMemcpy2DToArray(
            texture_storage_array, 0, 0, pixels_f32.data(), spitch,
            width * num_channels_converted * sizeof(f32), height, cudaMemcpyHostToDevice))
    } else {
        read_mode = cudaReadModeNormalizedFloat;
        cudaChannelFormatDesc channel_desc{};

        std::vector<u8> pixels_rgba;
        u8 *pixels_to_load = pixels;

        if (num_channels == 3) {
            // CUDA Texture objects don't support 3-channel textures
            num_channels = 4;
            pixels_rgba = std::vector<u8>(width * height * num_channels);
            for (i32 p = 0; p < width * height; p++) {
                pixels_rgba[4 * p + 0] = pixels[3 * p + 0];
                pixels_rgba[4 * p + 1] = pixels[3 * p + 1];
                pixels_rgba[4 * p + 2] = pixels[3 * p + 2];
                pixels_rgba[4 * p + 3] = 255;
            }

            pixels_to_load = pixels_rgba.data();
        }

        switch (num_channels) {
        case 1: {
            channel_desc =
                cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
            break;
        }
        case 2: {
            channel_desc =
                cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
            break;
        }
        case 4: {
            channel_desc =
                cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            break;
        }
        default: {
            throw std::runtime_error("Invalid texture channel count");
            throw;
        }
        }

        CUDA_CHECK(cudaMallocArray(&texture_storage_array, &channel_desc, width, height))

        size_t spitch = width * num_channels * sizeof(u8);
        CUDA_CHECK(cudaMemcpy2DToArray(texture_storage_array, 0, 0, pixels_to_load,
                                       spitch, width * num_channels * sizeof(u8), height,
                                       cudaMemcpyHostToDevice))

        stbi_image_free(pixels);
    }

    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = texture_storage_array;

    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = 1;
    tex_desc.sRGB = 0;

    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    return Texture(tex_obj, texture_storage_array, width, height);
}

Texture
Texture::make(const std::string &texture_path, bool is_rgb) {
    if (texture_path.ends_with(".exr")) {
        return load_exr_texture(texture_path, is_rgb);
    } else {
        return load_other_format_texture(texture_path, is_rgb);
    }
}

Texture::Texture(Texture &&other) noexcept {
    tex_obj = other.tex_obj;
    texture_storage_array = other.texture_storage_array;
    width = other.width;
    height = other.height;

    other.tex_obj = 0;
    other.texture_storage_array = nullptr;
    other.width = 0;
    other.height = 0;
}

Texture &
Texture::operator=(Texture &&other) noexcept {
    width = other.width;
    height = other.height;
    tex_obj = other.tex_obj;
    texture_storage_array = other.texture_storage_array;

    other.tex_obj = 0;
    other.texture_storage_array = nullptr;
    other.width = 0;
    other.height = 0;

    return *this;
}

Texture::~Texture() {
    if (tex_obj != 0) {
        cudaDestroyTextureObject(tex_obj);
    }

    if (texture_storage_array != nullptr) {
        cudaFreeArray(texture_storage_array);
    }
}
