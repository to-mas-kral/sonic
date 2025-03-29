#ifndef SS_TREE_H
#define SS_TREE_H

#include "../materials/material_id.h"
#include "../math/axis.h"
#include "../math/vecmath.h"
#include "../spectrum/sampled_lambdas.h"
#include "../spectrum/spectrum_consts.h"
#include "binary_tree.h"

#include <algorithm>
#include <memory>
#include <spdlog/spdlog.h>

constexpr u32 STARTING_BIN_WIDTH = 50;
constexpr u32 MIN_BIN_WIDTH = 5;
constexpr u32 MAX_HISTOGRAM_BINS = (LAMBDA_RANGE - 1 + MIN_BIN_WIDTH - 1) / MIN_BIN_WIDTH;

class Reservoir {
public:
    Reservoir()
        : sampling_binary_tree(std::make_unique<BinaryTree>()),
          recording_binary_tree(std::make_unique<BinaryTree>()) {}

    Reservoir(const Reservoir &other)
        : sampling_binary_tree(
              std::make_unique<BinaryTree>(*other.sampling_binary_tree)) {
        if (other.recording_binary_tree != nullptr) {
            recording_binary_tree =
                std::make_unique<BinaryTree>(*other.recording_binary_tree);
        }
    }

    Reservoir(Reservoir &&other) noexcept
        : sampling_binary_tree(std::move(other.sampling_binary_tree)) {
        if (other.recording_binary_tree != nullptr) {
            recording_binary_tree = std::move(other.recording_binary_tree);
        }
    }

    Reservoir &
    operator=(const Reservoir &other) {
        if (this == &other) {
            return *this;
        }
        sampling_binary_tree = std::make_unique<BinaryTree>(*other.sampling_binary_tree);
        if (other.recording_binary_tree != nullptr) {
            recording_binary_tree =
                std::make_unique<BinaryTree>(*other.recording_binary_tree);
        }
        return *this;
    }

    Reservoir &
    operator=(Reservoir &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        sampling_binary_tree = std::move(other.sampling_binary_tree);
        if (other.recording_binary_tree != nullptr) {
            recording_binary_tree = std::move(other.recording_binary_tree);
        }
        return *this;
    }

    ~Reservoir() = default;

    SampledLambdas
    sample(const f32 xi_base) const {
        return sampling_binary_tree->sample(xi_base);
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
    refine(const bool is_last_iter = false) {
        recording_binary_tree->refine(0.03F);
        sampling_binary_tree = std::make_unique<BinaryTree>(*recording_binary_tree);

        if (is_last_iter) {
            recording_binary_tree = nullptr;
        } else {
            recording_binary_tree->reset_radiance();
        }
    }

    std::unique_ptr<BinaryTree> sampling_binary_tree{nullptr};
    std::unique_ptr<BinaryTree> recording_binary_tree{nullptr};
};

class Reservoirs {
public:
    Reservoirs() {
        x_coords.reserve(8);
        y_coords.reserve(8);
        reservoirs.reserve(8);
    }

    Reservoirs(const Reservoirs &other)
        : x_coords(other.x_coords), y_coords(other.y_coords),
          reservoirs(other.reservoirs) {}

    Reservoirs(Reservoirs &&other) noexcept
        : x_coords(std::move(other.x_coords)), y_coords(std::move(other.y_coords)),
          reservoirs(std::move(other.reservoirs)) {}

    Reservoirs &
    operator=(const Reservoirs &other) {
        if (this == &other) {
            return *this;
        }
        x_coords = other.x_coords;
        y_coords = other.y_coords;
        reservoirs = other.reservoirs;
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
        return *this;
    }

    ~Reservoirs() = default;

    Reservoir *
    find_reservoir_inner(uvec2 pixel, bool is_training_phase);

    Reservoir *
    find_reservoir(uvec2 pixel, bool is_training_phase);

    void
    refine(const bool is_last_iter = false) {
        if (reservoirs.capacity() == 0) {
            x_coords.reserve(x_coords.size() * 2);
            y_coords.reserve(y_coords.size() * 2);
            reservoirs.reserve(reservoirs.size() * 2);
        }

        for (auto &res : reservoirs) {
            res.refine(is_last_iter);
        }
    }

    // Reservoirs in SOA layout
    std::vector<u16> x_coords;
    std::vector<u16> y_coords;
    std::vector<Reservoir> reservoirs;
    std::mutex reservoirs_mutex;
};

class ReservoirsContainer {
public:
    explicit ReservoirsContainer(const u32 num_materials) {
        m_reservoirs.reserve(num_materials);
        for (int i = 0; i < num_materials; ++i) {
            m_reservoirs.push_back(nullptr);
        }
    }

    ReservoirsContainer(const ReservoirsContainer &other) {
        m_reservoirs.reserve(other.m_reservoirs.size());
        for (int i = 0; i < other.m_reservoirs.size(); ++i) {
            if (other.m_reservoirs[i] == nullptr) {
                m_reservoirs.push_back(nullptr);
            } else {
                m_reservoirs.push_back(
                    std::make_unique<Reservoirs>(*other.m_reservoirs[i]));
            }
        }
    }

    ReservoirsContainer(ReservoirsContainer &&other) noexcept {
        m_reservoirs.reserve(other.m_reservoirs.size());
        for (int i = 0; i < other.m_reservoirs.size(); ++i) {
            if (other.m_reservoirs[i] == nullptr) {
                m_reservoirs.push_back(nullptr);
            } else {
                m_reservoirs.push_back(
                    std::make_unique<Reservoirs>(*other.m_reservoirs[i]));
            }
        }
    }

    ReservoirsContainer &
    operator=(const ReservoirsContainer &other) {
        if (this == &other) {
            return *this;
        }
        m_reservoirs.clear();
        m_reservoirs.reserve(other.m_reservoirs.size());
        for (int i = 0; i < other.m_reservoirs.size(); ++i) {
            m_reservoirs.push_back(std::make_unique<Reservoirs>(*other.m_reservoirs[i]));
        }
        return *this;
    }

    ReservoirsContainer &
    operator=(ReservoirsContainer &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        m_reservoirs.clear();
        m_reservoirs.reserve(other.m_reservoirs.size());
        for (int i = 0; i < other.m_reservoirs.size(); ++i) {
            m_reservoirs.push_back(std::make_unique<Reservoirs>(*other.m_reservoirs[i]));
        }
        return *this;
    }

    ~ReservoirsContainer() = default;

    Reservoir *
    find_reservoir(const MaterialId mat_id, const uvec2 pixel,
                   const bool is_training_phase) {
        const auto find_reservoir_inner = [this, mat_id, pixel,
                                           is_training_phase]() -> Reservoir * {
            const auto &reservoirs = m_reservoirs[mat_id.inner];

            if (reservoirs == nullptr && !is_training_phase) {
                return nullptr;
            }

            if (reservoirs == nullptr) {
                m_reservoirs[mat_id.inner] = std::make_unique<Reservoirs>(Reservoirs());
            }

            return m_reservoirs[mat_id.inner]->find_reservoir(pixel, is_training_phase);
        };

        if (is_training_phase) {
            const std::scoped_lock lock(reservoirs_mutex);
            return find_reservoir_inner();
        } else {
            return find_reservoir_inner();
        }
    }

    void
    refine(const bool is_last_iter = false) {
        for (const auto &reservoirs : m_reservoirs) {
            if (reservoirs) {
                reservoirs->refine(is_last_iter);
            }
        }
    }

    const std::vector<std::unique_ptr<Reservoirs>> &
    reservoirs() const {
        return m_reservoirs;
    }

private:
    std::vector<std::unique_ptr<Reservoirs>> m_reservoirs;
    std::mutex reservoirs_mutex;
};

#endif // SS_TREE_H
