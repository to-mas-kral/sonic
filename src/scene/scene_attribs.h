#ifndef SCENE_ATTRIBS_H
#define SCENE_ATTRIBS_H

#include <filesystem>

#include "../math/transform.h"
#include "../utils/basic_types.h"

struct CameraAttribs {
    f32 fov = 90.F;
    mat4 camera_to_world = mat4::identity();
};

struct FilmAttribs {
    u32 resx = 1280;
    u32 resy = 720;
    f32 iso = 100.F;
    std::filesystem::path filename{"out"};
};

struct SceneAttribs {
    CameraAttribs camera{};
    FilmAttribs film{};
    u32 max_depth = 5;
};

#endif // SCENE_ATTRIBS_H
