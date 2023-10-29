#ifndef PT_INTERSECTION_H
#define PT_INTERSECTION_H

#include "../utils/numtypes.h"
#include "../render_context_common.h"

class Intersection {
public:
    /// Position
    vec3 pos;
    vec3 normal;
    /// Intersection ray parameter
    f32 t;

    Mesh* mesh;
};

#endif // PT_INTERSECTION_H
