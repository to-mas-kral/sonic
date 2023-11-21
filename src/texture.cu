
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "texture.h"

Texture::Texture(const std::string &texture_path) {
    cudaChannelFormatDesc channel_desc{};
    size_t spitch;
    cudaTextureReadMode read_mode;

    // Some duplicated code, but its probably better to keep duplicated...
    if (texture_path.ends_with(".exr")) {
        float *pixels = nullptr;
        int num_channels = 4;
        read_mode = cudaReadModeElementType;

        const char *err = nullptr;
        int ret = LoadEXR(&pixels, &width, &height, texture_path.c_str(), &err);
        if (ret != TINYEXR_SUCCESS) {
            if (err) {
                spdlog::error("EXR loading error: {}", err);
                FreeEXRErrorMessage(err);
            }
        }

        channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        CUDA_CHECK(cudaMallocArray(&texture_storage_array, &channel_desc, width, height))

        spitch = width * num_channels * sizeof(f32);
        CUDA_CHECK(cudaMemcpy2DToArray(texture_storage_array, 0, 0, pixels, spitch,
                                       width * num_channels * sizeof(f32), height,
                                       cudaMemcpyHostToDevice))

        free(pixels);
    } else {
        int num_channels = 0;
        u8 *pixels = stbi_load(texture_path.c_str(), &width, &height, &num_channels, 0);
        read_mode = cudaReadModeNormalizedFloat;

        std::vector<u8> pixels_rgba;
        u8 *pixels_to_load = pixels;

        if (num_channels == 3) {
            // CUDA Texture objects don't support 3-channel textures
            num_channels = 4;
            pixels_rgba = std::vector<u8>(width * height * num_channels);
            for (int p = 0; p < width * height; p++) {
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
            spdlog::error("Invalid texture channel count");
            throw;
        }
        }

        CUDA_CHECK(cudaMallocArray(&texture_storage_array, &channel_desc, width, height))

        spitch = width * num_channels * sizeof(u8);
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
    tex_desc.sRGB = 1;

    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr))
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
