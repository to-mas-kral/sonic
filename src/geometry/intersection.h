#ifndef PT_INTERSECTION_H
#define PT_INTERSECTION_H

class Intersection {
public:
    /// Position
    vec3 pos;
    vec3 normal;
    /// Intersection ray parameter
    f32 t;

    u32 material_id;
    Mesh* mesh;
};

#endif // PT_INTERSECTION_H
