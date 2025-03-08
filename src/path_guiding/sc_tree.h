#ifndef SS_TREE_H
#define SS_TREE_H

#include "../materials/material_id.h"
#include "../math/axis.h"
#include "../math/vecmath.h"
#include "../spectrum/sampled_lambdas.h"
#include "../utils/make_array.h"
#include "binary_tree.h"

#include <algorithm>
#include <memory>
#include <spdlog/spdlog.h>

class Reservoir {
public:
    Reservoir()
        : sampling_binary_tree(std::make_unique<BinaryTree>()),
          recording_binary_tree(std::make_unique<BinaryTree>()) {}

    Reservoir(const Reservoir &other)
        : sampling_binary_tree(std::make_unique<BinaryTree>(*other.sampling_binary_tree)),
          recording_binary_tree(
              std::make_unique<BinaryTree>(*other.recording_binary_tree)) {}

    Reservoir(Reservoir &&other) noexcept
        : sampling_binary_tree(std::move(other.sampling_binary_tree)),
          recording_binary_tree(std::move(other.recording_binary_tree)) {}

    Reservoir &
    operator=(const Reservoir &other) {
        if (this == &other) {
            return *this;
        }
        sampling_binary_tree = std::make_unique<BinaryTree>(*other.sampling_binary_tree);
        recording_binary_tree =
            std::make_unique<BinaryTree>(*other.recording_binary_tree);
        return *this;
    }

    Reservoir &
    operator=(Reservoir &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        sampling_binary_tree = std::move(other.sampling_binary_tree);
        recording_binary_tree = std::move(other.recording_binary_tree);
        return *this;
    }

    ~Reservoir() = default;

    SampledLambdas
    sample(const f32 xi) const {
        return sampling_binary_tree->sample(xi);
    }

    f32
    pdf(const f32 lambda) const {
        return sampling_binary_tree->pdf(lambda);
    }

    void
    record(const SampledLambdas &lambdas, const spectral &radiance) const {
        recording_binary_tree->record(lambdas, radiance);
    }

    void
    refine() {
        recording_binary_tree->refine(0.03F);
        sampling_binary_tree = std::make_unique<BinaryTree>(*recording_binary_tree);
        recording_binary_tree->reset_flux();
    }

    std::unique_ptr<BinaryTree> sampling_binary_tree{nullptr};
    std::unique_ptr<BinaryTree> recording_binary_tree{nullptr};
};

constexpr u32 MAX_RESERVOIRS_PER_MAT = 64;

class Reservoirs {
public:
    Reservoirs() = default;

    Reservoirs(const Reservoirs &other)
        : x_coords(other.x_coords), y_coords(other.y_coords),
          reservoirs(other.reservoirs), num_reservoirs(other.num_reservoirs) {}

    Reservoirs(Reservoirs &&other) noexcept
        : x_coords(other.x_coords), y_coords(other.y_coords),
          reservoirs(std::move(other.reservoirs)), num_reservoirs(other.num_reservoirs) {}

    Reservoirs &
    operator=(const Reservoirs &other) {
        if (this == &other) {
            return *this;
        }
        x_coords = other.x_coords;
        y_coords = other.y_coords;
        reservoirs = other.reservoirs;
        num_reservoirs = other.num_reservoirs;
        return *this;
    }

    Reservoirs &
    operator=(Reservoirs &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        x_coords = other.x_coords;
        y_coords = other.y_coords;
        reservoirs = std::move(other.reservoirs);
        num_reservoirs = other.num_reservoirs;
        return *this;
    }

    ~Reservoirs() = default;

    const Reservoir *
    find_reservoir_inner(uvec2 pixel, bool is_training_phase);

    const Reservoir *
    find_reservoir(uvec2 pixel, bool is_training_phase);

    void
    refine() {
        for (int i = 0; i < num_reservoirs; ++i) {
            reservoirs[i].refine();
        }
    }

    // Reservoirs in SOA layout
    std::array<u16, MAX_RESERVOIRS_PER_MAT> x_coords{};
    std::array<u16, MAX_RESERVOIRS_PER_MAT> y_coords{};
    std::array<Reservoir, MAX_RESERVOIRS_PER_MAT> reservoirs{};
    std::mutex reservoirs_mutex;
    u32 num_reservoirs{0};
};

struct ReservoirsBucket {
    ReservoirsBucket() = default;

    ~ReservoirsBucket() = default;

    ReservoirsBucket(const ReservoirsBucket &other)
        : mark_for_init(other.mark_for_init.load()) {
        if (other.reservoirs) {
            reservoirs = std::move(std::make_unique<Reservoirs>(*other.reservoirs));
        }
    }

    ReservoirsBucket(ReservoirsBucket &&other) noexcept
        : mark_for_init(other.mark_for_init.load()),
          reservoirs(std::move(other.reservoirs)) {}

    ReservoirsBucket &
    operator=(const ReservoirsBucket &other) {
        if (this == &other) {
            return *this;
        }
        mark_for_init = other.mark_for_init.load();
        if (other.reservoirs) {
            reservoirs = std::make_unique<Reservoirs>(*other.reservoirs);
        }
        return *this;
    }

    ReservoirsBucket &
    operator=(ReservoirsBucket &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        mark_for_init = other.mark_for_init.load();
        reservoirs = std::move(other.reservoirs);
        return *this;
    }

    std::atomic<bool> mark_for_init{false};
    std::unique_ptr<Reservoirs> reservoirs{nullptr};
};

class LgTree {
public:
    explicit LgTree(const u32 num_materials) {
        m_reservoirs = std::vector(num_materials, ReservoirsBucket());
    }

    LgTree(const LgTree &other) : m_reservoirs(other.m_reservoirs) {}

    LgTree(LgTree &&other) noexcept : m_reservoirs(std::move(other.m_reservoirs)) {}

    LgTree &
    operator=(const LgTree &other) {
        if (this == &other) {
            return *this;
        }
        m_reservoirs = other.m_reservoirs;
        return *this;
    }

    LgTree &
    operator=(LgTree &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        m_reservoirs = std::move(other.m_reservoirs);
        return *this;
    }

    ~LgTree() = default;

    const Reservoir *
    find_reservoir(const MaterialId mat_id, const uvec2 pixel,
                   const bool is_training_phase) {
        auto &bucket = m_reservoirs[mat_id.inner];
        if (bucket.reservoirs == nullptr) {
            bucket.mark_for_init.store(true);
            return nullptr;
        }

        return m_reservoirs[mat_id.inner].reservoirs->find_reservoir(pixel,
                                                                     is_training_phase);
    }

    void
    refine() {
        int i = 0;
        for (auto &bucket : m_reservoirs) {
            if (bucket.mark_for_init && bucket.reservoirs == nullptr) {
                bucket.reservoirs = std::make_unique<Reservoirs>();
            }
            i++;

            if (bucket.reservoirs != nullptr) {
                bucket.reservoirs->refine();
            }
        }
    }

    const std::vector<ReservoirsBucket> &
    reservoirs() const {
        return m_reservoirs;
    }

private:
    std::vector<ReservoirsBucket> m_reservoirs;
    std::mutex buckets_mutex;
};

#endif // SS_TREE_H
