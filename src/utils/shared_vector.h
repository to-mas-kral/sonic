#ifndef PT_SHARED_VECTOR_H
#define PT_SHARED_VECTOR_H

#include <cstring>
#include <cuda/std/cassert>

#include "cuda_err.h"
#include "numtypes.h"

/// A "vector" data structure using Unified Memory.
/// It is designed in a way that the amount of elements can only be modified in host code.
/// Device code can access / modify individual elements but not modify the "vector"
/// itself.
// Storing objects that have pointers to non-unified memory is UB if used in kernels !!!
// TODO: implement iterator for SharedVector ?
template <class T> class SharedVector {
public:
    __host__ SharedVector() : m_len(0), cap(0), mem(nullptr) {}

    // Do I need synchronizations here ? I don't think so execept for destructor...
    // but could be problematic if multi-threading is used and kernels would be launched
    // from multiple threads (is that even allowed tho ?)

    __host__ explicit SharedVector(u64 capacity) : m_len(0), cap(capacity), mem(nullptr) {
        CUDA_CHECK(cudaMallocManaged((void **)&mem, cap * sizeof(T)))

        cudaDeviceSynchronize();
    }

    __host__ explicit SharedVector(T elem, u64 count)
        : m_len(count), cap(count), mem(nullptr) {
        if (count != 0) {
            CUDA_CHECK(cudaMallocManaged((void **)&mem, cap * sizeof(T)))

            cudaDeviceSynchronize();

            // TODO: is there some sort of memset for this ?
            for (int i = 0; i < count; i++) {
                mem[i] = elem;
            }
        }
    }

    __host__ SharedVector(std::initializer_list<T> l) {
        assert(l.size() > 0);

        cap = l.size();
        CUDA_CHECK(cudaMallocManaged((void **)&mem, cap * sizeof(T)))

        cudaDeviceSynchronize();

        memcpy(mem, l.begin(), l.size() * sizeof(T));

        m_len = l.size();
    }

    __host__ ~SharedVector() {
        if (mem != nullptr) {
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaFree(mem))
        }
    }

    SharedVector(SharedVector const &) = delete;

    SharedVector &operator=(SharedVector const &) = delete;

    SharedVector(SharedVector &&other) noexcept {
        cap = other.cap;
        m_len = other.m_len;
        mem = other.mem;

        other.cap = 0;
        other.m_len = 0;
        other.mem = nullptr;
    };

    SharedVector &operator=(SharedVector &&other) noexcept {
        mem = other.mem;
        cap = other.cap;
        m_len = other.m_len;

        other.cap = 0;
        other.m_len = 0;
        other.mem = nullptr;

        return *this;
    };

    void swap(SharedVector *other) {
        auto other_mem = other->mem;
        auto other_cap = other->cap;
        auto other_m_len = other->m_len;

        other->mem = mem;
        other->cap = cap;
        other->m_len = m_len;

        mem = other_mem;
        cap = other_cap;
        m_len = other_m_len;
    }

    __host__ __device__ T &last() const {
        assert(m_len > 0);
        return mem[m_len - 1];
    }

    __host__ __device__ T &get_unchecked(u64 idx) const { return mem[idx]; }

    __host__ void assume_all_init() { m_len = cap; }

    __host__ __device__ T &operator[](u64 idx) {
        assert(idx < m_len);
        return mem[idx];
    }

    __host__ __device__ const T &operator[](u64 idx) const {
        assert(idx < m_len);
        return mem[idx];
    }

    // TODO: refactor this std::move nonsense...
    __host__ void push(T &&elem) {
        if (cap == 0 || mem == nullptr) {
            // TODO: could select better default size based on T's size...
            cap = 8;
            CUDA_CHECK(cudaMallocManaged((void **)&mem, cap * sizeof(T)))

            cudaDeviceSynchronize();
        }

        if (m_len >= cap) {
            resize();
        }
        mem[m_len] = std::move(elem);
        m_len++;
    }

    __host__ __device__ u64 size() const { return m_len; }

    __host__ __device__ T *get_ptr() const { return mem; }

private:
    __host__ void resize() {
        u64 new_cap = cap * 2;

        T *new_mem = nullptr;
        CUDA_CHECK(cudaMallocManaged((void **)&new_mem, new_cap * sizeof(T)))

        cudaDeviceSynchronize();

        std::memcpy(new_mem, mem, m_len * sizeof(T));
        CUDA_CHECK(cudaFree(mem))
        mem = new_mem;
        cap = new_cap;
    }

    // Pointer to the allocated memory
    T *mem = nullptr;
    // The number of entries
    u64 m_len = 0;
    // The capacity
    u64 cap = 0;
};

#endif // PT_SHARED_VECTOR_H
