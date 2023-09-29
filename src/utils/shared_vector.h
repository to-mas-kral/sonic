#ifndef PT_SHARED_VECTOR_H
#define PT_SHARED_VECTOR_H

#include <cstring>
#include <cuda/std/cassert>

#include "../numtypes.h"

/// A "vector" data structure using Unified Memory.
/// It is designed in a way that the amount of elements can only be modified in host code.
/// Device code can access / modify individual elements but not modify the "vector" itself.
/// FIXME: storing objects that have pointers to non-unified memory is UB if used in kernels
template<class T>
class SharedVector {
public:
    __host__ SharedVector() : m_len(0), cap(0), mem(nullptr) {}

    __host__ explicit SharedVector(u64 capacity) : m_len(0), cap(capacity), mem(nullptr) {
        cudaMallocManaged((void **) &mem, cap);
        cudaDeviceSynchronize();
    }

    __host__ SharedVector(std::initializer_list<T> l) {
        assert(l.size() > 0);

        cap = l.size();
        cudaMallocManaged((void **) &mem, cap);
        cudaDeviceSynchronize();

        memcpy(mem, l.begin(), l.size() * sizeof(T));

        m_len = l.size();
    }

    __host__ ~SharedVector() {
        if (mem != nullptr) {
            cudaDeviceSynchronize();
            cudaFree(mem);
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

    __host__ __device__ T &operator[](u64 idx) {
        assert(idx < m_len);
        return mem[idx];
    }

    __host__ __device__ const T &operator[](u64 idx) const {
        assert(idx < m_len);
        return mem[idx];
    }

    __host__ void resize() {
        u64 new_cap = cap * 2;

        T *new_mem = nullptr;
        cudaMallocManaged((void **) &new_mem, new_cap);
        cudaDeviceSynchronize();

        std::memcpy(new_mem, mem, m_len * sizeof(T));

        cudaFree(mem);
        mem = new_mem;
        cap = new_cap;
    }

    /*__host__ void push(T elem) {
        if (m_len >= cap) { resize(); }
        mem[m_len] = elem;
        m_len++;
    }*/

    __host__ void push(T &&elem) {
        if (cap == 0 || mem == nullptr) {
            // TODO: could select better default size based on T's size...
            cap = 8;
            cudaMallocManaged((void **) &mem, cap);
            cudaDeviceSynchronize();
        }

        if (m_len >= cap) { resize(); }
        mem[m_len] = std::move(elem);
        m_len++;
    }

    __host__ __device__ u64 len() {
        return m_len;
    }

private:
    // Pointer to the allocated memory
    T *mem{nullptr};
    // The number of entries
    u64 m_len{0};
    // The capacity
    u64 cap{0};
};


#endif //PT_SHARED_VECTOR_H
