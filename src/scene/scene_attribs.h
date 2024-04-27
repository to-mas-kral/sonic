#ifndef SCENE_ATTRIBS_H
#define SCENE_ATTRIBS_H

#include "../math/transform.h"
#include "../utils/basic_types.h"

struct CameraAttribs {
    u32 resx = 1280;
    u32 resy = 720;
    f32 fov = 30.f;
    mat4 camera_to_world = mat4::identity();
};

struct SceneAttribs {
    CameraAttribs camera_attribs{};
    u32 max_depth = 0;
};

#endif // SCENE_ATTRIBS_H
