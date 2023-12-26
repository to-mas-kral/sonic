#ifndef PT_IMAGE_WRITER_H
#define PT_IMAGE_WRITER_H

#include <vector>

#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "../color/color_space.h"
#include "../framebuffer.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

namespace ImageWriter {

void
write_framebuffer(const std::string &filename, Framebuffer &fb, u32 num_samples) {
    auto width = fb.get_res_x();
    auto height = fb.get_res_y();

    UmVector<vec3> &pixels = fb.get_pixels();

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    for (int i = 0; i < width * height; i++) {
        vec3 xyz = pixels[i] / static_cast<float>(num_samples);
        tuple3 rgb = xyz_to_srgb(tuple3(xyz.x, xyz.y, xyz.z));

        images[0][i] = rgb.x;
        images[1][i] = rgb.y;
        images[2][i] = rgb.z;
    }

    float *image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char **)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels =
        (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be BGR(A) order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255);
    header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] =
            TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char *err;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        spdlog::error("Error when saving output image file: {}\n", err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

} // namespace ImageWriter

#endif // PT_IMAGE_WRITER_H
