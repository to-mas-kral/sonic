#ifndef PT_EMBREE_DEVICE_H
#define PT_EMBREE_DEVICE_H

#include "integrator/intersection.h"
#include "math/aabb.h"
#include "scene/scene.h"

#include <embree4/rtcore.h>
#include <fmt/core.h>
#include <iostream>
#include <limits>

inline void
errorFunction(void *userPtr, const RTCError error, const char *str) {
    spdlog::error(fmt::format("Embree error {}: {}", (i32)error, str));
}

inline void
filter_intersect_mesh(const RTCFilterFunctionNArguments *args) {
    assert(args->context);

    const auto *hit = reinterpret_cast<RTCHit *const>(args->hit);

    // Not sure why this is needed, but it's in the Embree tutorials...
    if (args->valid[0] != -1) {
        return;
    }

    const auto *mesh = static_cast<Mesh *>(args->geometryUserPtr);

    const auto bary = vec2(hit->u, hit->v);
    const auto bar = vec3(1.f - bary.x - bary.y, bary.x, bary.y);
    const auto uv = mesh->calc_uvs(hit->primID, bar);

    const auto alpha = mesh->alpha->fetch(uv);
    constexpr auto alpha_rand = 0.5f;

    if (alpha_rand > alpha) {
        args->valid[0] = 0;
    }
}

inline void
filter_intersect_sphere(const RTCFilterFunctionNArguments *args) {
    assert(args->context);

    const auto *hit = reinterpret_cast<RTCHit *const>(args->hit);
    const auto *ray = reinterpret_cast<RTCRay *const>(args->ray);

    // Not sure why this is needed, but it's in the Embree tutorials...
    if (args->valid[0] != -1) {
        return;
    }

    const auto *spheres = static_cast<Spheres *>(args->geometryUserPtr);

    const auto orig = point3(ray->org_x, ray->org_y, ray->org_z);
    const auto dir = vec3(ray->dir_x, ray->dir_y, ray->dir_z);

    const auto raypos = orig + ray->tfar * dir;
    const auto normal = Spheres::calc_normal(raypos, spheres->vertices[hit->primID].pos);
    const auto uv = Spheres::calc_uvs(normal);

    const auto &attribs = &spheres->attribs[hit->primID];

    const auto alpha = attribs->alpha->fetch(uv);
    constexpr auto alpha_rand = 0.3f;

    if (alpha_rand < alpha) {
        args->valid[0] = 0;
    }
}

class EmbreeDevice {
public:
    explicit
    EmbreeDevice(Scene &scene)
        : scene(&scene) {
        device = initialize_device();
        initialize_scene();
    }

    Intersection
    get_triangle_its(const Mesh *meshes, const u32 mesh_index, const u32 triangle_index,
                     const vec2 &bary) const {
        const auto &mesh = meshes[mesh_index];

        const auto bar = vec3(1.f - bary.x - bary.y, bary.x, bary.y);

        const auto indices = mesh.get_tri_indices(triangle_index);
        const auto pos_arr = mesh.get_tri_pos(indices);
        const auto pos = barycentric_interp(bar, pos_arr[0], pos_arr[1], pos_arr[2]);

        const auto normal = mesh.calc_normal(triangle_index, bar, false);
        const auto geometric_normal = mesh.calc_normal(triangle_index, bar, true);
        const auto uv = mesh.calc_uvs(triangle_index, bar);

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
    get_sphere_its(const Spheres &spheres, const u32 sphere_id, const point3 &pos) const {
        const auto &center = spheres.vertices[sphere_id].pos;
        const auto normal = Spheres::calc_normal(pos, center);

        const auto &attribs = spheres.attribs[sphere_id];

        return Intersection{
            .material_id = attribs.material_id,
            .light_id = attribs.light_id,
            .has_light = attribs.has_light,
            .normal = normal,
            .geometric_normal = Spheres::calc_normal(pos, center, true),
            .pos = pos,
            .uv = Spheres::calc_uvs(normal),
        };
    }

    std::optional<Intersection>
    intersect_instance(const point3 &orig, const vec3 &dir,
                       const RTCRayHit &rayhit) const {
        const auto instance_indice =
            scene->geometry.instances.indices[rayhit.hit.instPrimID[0]];

        const auto &instanced_obj =
            scene->geometry.instances.instanced_objs[instance_indice];

        auto its = Intersection::make_empty();

        if (rayhit.hit.geomID < mesh_geom_counts[instance_indice]) {
            its = get_triangle_its(instanced_obj.meshes.meshes.data(), rayhit.hit.geomID,
                                   rayhit.hit.primID, vec2(rayhit.hit.u, rayhit.hit.v));
        } else {
            const point3 pos = orig + rayhit.ray.tfar * dir;
            its = get_sphere_its(instanced_obj.spheres, rayhit.hit.primID, pos);
        }

        const auto &transform =
            scene->geometry.instances.world_from_instances[rayhit.hit.instPrimID[0]];
        const auto &transform_inv_trans =
            scene->geometry.instances.wfi_inv_trans[rayhit.hit.instPrimID[0]];

        its.pos = transform.transform_point(its.pos);
        its.normal = transform_inv_trans.transform_vec(its.normal).normalized();
        its.geometric_normal =
            transform_inv_trans.transform_vec(its.geometric_normal).normalized();

        return its;
    }

    std::optional<Intersection>
    intersect_non_instance(const point3 &orig, const vec3 &dir,
                           const RTCRayHit &rayhit) const {
        if (rayhit.hit.geomID < mesh_geom_count) {
            return get_triangle_its(scene->geometry.meshes.meshes.data(),
                                    rayhit.hit.geomID, rayhit.hit.primID,
                                    vec2(rayhit.hit.u, rayhit.hit.v));
        } else {
            const point3 pos = orig + rayhit.ray.tfar * dir;
            return get_sphere_its(scene->geometry.spheres, rayhit.hit.primID, pos);
        }
    }

    std::optional<Intersection>
    cast_ray(const point3 &orig, const vec3 &dir) const {
        RTCRayHit rayhit{};
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

        rtcIntersect1(main_scene, &rayhit);

        if (rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) {
            return intersect_instance(orig, dir, rayhit);
        }

        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            return intersect_non_instance(orig, dir, rayhit);
        }

        return {};
    }

    std::optional<Intersection>
    cast_ray(const Ray &ray) const {
        return cast_ray(ray.o, ray.dir);
    }

    bool
    is_visible(const point3 a, const point3 b) const {
        const vec3 dir = b - a;
        const point3 orig = a;

        // tfar is relative to the ray length
        constexpr f32 tfar = 0.999f;

        RTCRay rtc_ray{};
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

        rtcOccluded1(main_scene, &rtc_ray);

        if (rtc_ray.tfar == -INFINITY) {
            return false;
        } else {
            return true;
        }
    }

    static RTCDevice
    initialize_device() {
        const RTCDevice device = rtcNewDevice(nullptr);

        if (!device) {
            spdlog::error(fmt::format("Cannot create Embree device, error: {}\n",
                                      static_cast<i32>(rtcGetDeviceError(nullptr))));
        }

        rtcSetDeviceErrorFunction(device, errorFunction, nullptr);
        return device;
    }

    RTCScene
    initialize_scene() {
        main_scene = rtcNewScene(device);

        initialize_meshes();
        initialize_spheres();
        initialize_instances();

        rtcCommitScene(main_scene);

        RTCBounds bounds{};
        rtcGetSceneBounds(main_scene, &bounds);

        const auto low = vec3(bounds.lower_x, bounds.lower_y, bounds.lower_z);
        const auto high = vec3(bounds.upper_x, bounds.upper_y, bounds.upper_z);

        scene->set_scene_bounds(AABB(low, high));

        return main_scene;
    }

    void
    initialize_meshes() {
        auto &meshes = scene->geometry.meshes.meshes;
        mesh_geom_count = meshes.size();
        for (const auto &mesh : meshes) {
            const auto geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

            rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                       mesh.pos, 0, sizeof(point3), mesh.num_verts);

            rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                       mesh.indices, 0, 3 * sizeof(u32),
                                       mesh.num_indices / 3);

            if (mesh.alpha) {
                rtcSetGeometryIntersectFilterFunction(geom, filter_intersect_mesh);
                rtcSetGeometryOccludedFilterFunction(geom, filter_intersect_mesh);
                rtcSetGeometryUserData(geom, (void *)(&mesh));
            }

            rtcCommitGeometry(geom);

            /* From Embree 4 docs:
             * The geometry IDs are assigned sequentially, starting from 0, as long as no
             * geometry got detached.
             * */
            rtcAttachGeometry(main_scene, geom);
            rtcReleaseGeometry(geom);
        }
    }

    void
    initialize_instances() {
        const auto &instances = scene->geometry.instances;

        if (instances.indices.empty()) {
            return;
        }

        instance_count = instances.instanced_objs.size();

        instance_scenes.reserve(instance_count);
        for (const auto &instance : instances.instanced_objs) {
            auto instance_scene = rtcNewScene(device);

            {
                mesh_geom_counts.push_back(instance.meshes.meshes.size());

                for (const auto &mesh : instance.meshes.meshes) {
                    const auto geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

                    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                               RTC_FORMAT_FLOAT3, mesh.pos, 0,
                                               sizeof(point3), mesh.num_verts);

                    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                               RTC_FORMAT_UINT3, mesh.indices, 0,
                                               3 * sizeof(u32), mesh.num_indices / 3);

                    if (mesh.alpha) {
                        rtcSetGeometryIntersectFilterFunction(geom,
                                                              filter_intersect_mesh);
                        rtcSetGeometryOccludedFilterFunction(geom, filter_intersect_mesh);
                        rtcSetGeometryUserData(geom, (void *)(&mesh));
                    }

                    rtcCommitGeometry(geom);

                    /* From Embree 4 docs:
                     * The geometry IDs are assigned sequentially, starting from 0, as
                     * long as no geometry got detached.
                     * */
                    rtcAttachGeometry(instance_scene, geom);
                    rtcReleaseGeometry(geom);
                }

                const auto &spheres = instance.spheres;
                const auto &vertices = spheres.vertices;

                for (int i = 0; i < spheres.num_spheres(); ++i) {
                    const auto geom =
                        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);

                    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                               RTC_FORMAT_FLOAT4, &vertices[i], 0,
                                               sizeof(SphereVertex), 1);

                    if (spheres.attribs[i].alpha) {
                        rtcSetGeometryIntersectFilterFunction(geom,
                                                              filter_intersect_sphere);
                        rtcSetGeometryOccludedFilterFunction(geom,
                                                             filter_intersect_sphere);
                        rtcSetGeometryUserData(geom, (void *)(&spheres));
                    }

                    rtcCommitGeometry(geom);

                    rtcAttachGeometry(instance_scene, geom);
                    rtcReleaseGeometry(geom);
                }
            }

            rtcCommitScene(instance_scene);

            instance_scenes.push_back(instance_scene);
        }

        const auto instance_array =
            rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE_ARRAY);

        rtcSetGeometryInstancedScenes(instance_array, instance_scenes.data(),
                                      instance_scenes.size());

        rtcSetSharedGeometryBuffer(instance_array, RTC_BUFFER_TYPE_INDEX, 0,
                                   RTC_FORMAT_UINT, instances.indices.data(), 0,
                                   sizeof(u32), instances.indices.size());

        rtcSetSharedGeometryBuffer(
            instance_array, RTC_BUFFER_TYPE_TRANSFORM, 0,
            RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, instances.world_from_instances.data(), 0,
            sizeof(SquareMatrix4), instances.world_from_instances.size());

        rtcCommitGeometry(instance_array);
        rtcAttachGeometry(main_scene, instance_array);
        rtcReleaseGeometry(instance_array);
    }

    void
    initialize_spheres() {
        const auto &spheres = scene->geometry.spheres;
        const auto &vertices = spheres.vertices;

        sphere_geom_count = spheres.num_spheres();

        for (int i = 0; i < sphere_geom_count; ++i) {
            const RTCGeometry geom =
                rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);

            rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4,
                                       &vertices[i], 0, sizeof(SphereVertex), 1);

            if (spheres.attribs[i].alpha) {
                rtcSetGeometryIntersectFilterFunction(geom, filter_intersect_sphere);
                rtcSetGeometryOccludedFilterFunction(geom, filter_intersect_sphere);
                rtcSetGeometryUserData(geom, (void *)(&spheres));
            }

            rtcCommitGeometry(geom);

            rtcAttachGeometry(main_scene, geom);
            rtcReleaseGeometry(geom);
        }
    }

    ~
    EmbreeDevice() {
        rtcReleaseDevice(device);
        rtcReleaseScene(main_scene);

        for (const auto &scene : instance_scenes) {
            rtcReleaseScene(scene);
        }
    }

private:
    Scene *scene;

    /// goem_ids are assigned sequentially by Embree
    /// we can know which type of object was intersected by looking at the counts for
    /// the different geometries if they were created sequentially
    u32 mesh_geom_count{0};
    u32 sphere_geom_count{0};

    u32 instance_count{0};
    std::vector<RTCScene> instance_scenes{};
    // TODO: refactor later when multilevel instancing is added
    std::vector<u32> mesh_geom_counts{};

    RTCDevice device;
    RTCScene main_scene;
};

#endif // PT_EMBREE_DEVICE_H
