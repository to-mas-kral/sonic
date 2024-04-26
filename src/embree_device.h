#ifndef PT_EMBREE_DEVICE_H
#define PT_EMBREE_DEVICE_H

#include "integrator/intersection.h"
#include "scene/scene.h"

#include <embree4/rtcore.h>
#include <fmt/core.h>
#include <iostream>
#include <limits>

inline void
errorFunction(void *userPtr, enum RTCError error, const char *str) {
    spdlog::error(fmt::format("Embree error {}: {}", (i32)error, str));
}

class EmbreeDevice {
public:
    explicit EmbreeDevice(Scene &scene) : scene(&scene) {
        device = initialize_device();
        initialize_scene();
    }

    Intersection
    get_triangle_its(u32 mesh_index, u32 triangle_index, const vec2 &bary) {
        auto &mesh = scene->geometry.meshes.meshes[mesh_index];
        auto &meshes = scene->geometry.meshes;

        auto [i0, i1, i2] = meshes.get_tri_indices(mesh.indices_index, triangle_index);
        auto [p0, p1, p2] = meshes.get_tri_pos(mesh.pos_index, {i0, i1, i2});

        vec3 bar = vec3(1.f - bary.x - bary.y, bary.x, bary.y);

        point3 pos = barycentric_interp(bar, p0, p1, p2);

        norm_vec3 normal = meshes.calc_normal(mesh.has_normals, i0, i1, i2,
                                              mesh.normals_index, bar, p0, p1, p2);
        norm_vec3 geometric_normal = meshes.calc_normal(
            mesh.has_normals, i0, i1, i2, mesh.normals_index, bar, p0, p1, p2, true);
        vec2 uv = meshes.calc_uvs(mesh.has_uvs, i0, i1, i2, mesh.uvs_index, bar);

        return Intersection{
            .material_id = mesh.material_id,
            .light_id = mesh.lights_start_id + triangle_index,
            .has_light = mesh.has_light,
            .normal = normal,
            .geometric_normal = geometric_normal,
            .pos = pos,
            .uv = uv,
        };
    }

    Intersection
    get_sphere_its(u32 sphere_id, const point3 &pos) {
        auto &spheres = scene->geometry.spheres;

        auto &center = spheres.vertices[sphere_id].pos;
        auto normal = Spheres::calc_normal(pos, center);

        return Intersection{
            .material_id = spheres.material_ids[sphere_id],
            .light_id = spheres.light_ids[sphere_id],
            .has_light = spheres.has_light[sphere_id],
            .normal = normal,
            .geometric_normal = Spheres::calc_normal(pos, center, true),
            .pos = pos,
            .uv = Spheres::calc_uvs(normal),
        };
    }

    Option<Intersection>
    cast_ray(point3 orig, vec3 dir) {
        RTCRayHit rayhit {};
        rayhit.ray.org_x = orig.x;
        rayhit.ray.org_y = orig.y;
        rayhit.ray.org_z = orig.z;
        rayhit.ray.dir_x = dir.x;
        rayhit.ray.dir_y = dir.y;
        rayhit.ray.dir_z = dir.z;
        rayhit.ray.tnear = 0;
        rayhit.ray.tfar = std::numeric_limits<f32>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(rtc_scene, &rayhit);

        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            if (rayhit.hit.geomID < mesh_count) {
                return get_triangle_its(rayhit.hit.geomID, rayhit.hit.primID,
                                        vec2(rayhit.hit.u, rayhit.hit.v));
            } else {
                point3 pos = orig + rayhit.ray.tfar * dir;
                return get_sphere_its(rayhit.hit.primID, pos);
            }

        } else {
            return {};
        }
    }

    Option<Intersection>
    cast_ray(const Ray &ray) {
        return cast_ray(ray.o, ray.dir);
    }

    bool
    is_visible(point3 a, point3 b) {
        vec3 dir = b - a;
        point3 orig = a;

        // tfar is relative to the ray length
        f32 tfar = 0.999f;

        RTCRay rtc_ray {};
        rtc_ray.org_x = orig.x;
        rtc_ray.org_y = orig.y;
        rtc_ray.org_z = orig.z;
        rtc_ray.dir_x = dir.x;
        rtc_ray.dir_y = dir.y;
        rtc_ray.dir_z = dir.z;
        rtc_ray.tnear = 0.001f;
        rtc_ray.tfar = tfar;
        rtc_ray.mask = -1;
        rtc_ray.flags = 0;
        rtc_ray.time = 0;

        rtcOccluded1(rtc_scene, &rtc_ray);

        if (rtc_ray.tfar == -INFINITY) {
            return false;
        } else {
            return true;
        }
    }

    static RTCDevice
    initialize_device() {
        RTCDevice device = rtcNewDevice(nullptr);

        if (!device) {
            spdlog::error(fmt::format("Cannot create Embree device, error: {}\n",
                                      (i32)rtcGetDeviceError(nullptr)));
        }

        rtcSetDeviceErrorFunction(device, errorFunction, nullptr);
        return device;
    }

    RTCScene
    initialize_scene() {
        rtc_scene = rtcNewScene(device);

        initialize_meshes();
        initialize_spheres();

        rtcCommitScene(rtc_scene);

        return rtc_scene;
    }

    void
    initialize_meshes() {
        auto &pos = scene->geometry.meshes.pos;
        auto &indices = scene->geometry.meshes.indices;

        auto &meshes = scene->geometry.meshes.meshes;
        mesh_count = meshes.size();
        for (auto &mesh : meshes) {
            RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

            size_t pos_byte_offset = mesh.pos_index * sizeof(point3);
            size_t indices_byte_offset = mesh.indices_index * sizeof(u32);

            rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                       pos.data(), pos_byte_offset, sizeof(point3),
                                       mesh.num_vertices);

            rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                       indices.data(), indices_byte_offset,
                                       3 * sizeof(u32), mesh.num_indices / 3);

            rtcCommitGeometry(geom);

            /* From Embree 4 docs:
             * The geometry IDs are assigned sequentially, starting from 0, as long as no
             * ge- ometry got detached.
             * */
            rtcAttachGeometry(rtc_scene, geom);
            rtcReleaseGeometry(geom);
        }
    }

    void
    initialize_spheres() {
        auto &spheres = scene->geometry.spheres;
        auto &vertices = spheres.vertices;

        sphere_count = spheres.num_spheres;

        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);

        rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4,
                                   vertices.data(), 0, sizeof(SphereVertex),
                                   sphere_count);

        rtcCommitGeometry(geom);

        rtcAttachGeometry(rtc_scene, geom);
        rtcReleaseGeometry(geom);
    }

    ~EmbreeDevice() {
        rtcReleaseDevice(device);
        rtcReleaseScene(rtc_scene);
    }

private:
    Scene *scene;

    /// goem_ids are assigned sequentially by Embree
    /// we can know which type of object was intersected by looking at the counts for
    /// the different geometries if they were created sequentially
    u32 mesh_count{0};
    u32 sphere_count{0};

    RTCDevice device;
    RTCScene rtc_scene;
};

#endif // PT_EMBREE_DEVICE_H
