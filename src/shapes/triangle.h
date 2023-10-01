#ifndef PT_TRIANGLE_H
#define PT_TRIANGLE_H

#include "../utils/numtypes.h"
#include "mesh.h"

class Triangle {
public:
    Triangle(Mesh *mesh, u64 id) : mesh(mesh), id(id) {}

    Mesh *get_mesh() const { return mesh; }
    u64 get_id() const { return id; }

private:
    Mesh *mesh;
    /// Index into the Mesh arrays
    u64 id;
};

#endif // PT_TRIANGLE_H
