#ifndef PT_CHUNK_ALLOCATOR_H
#define PT_CHUNK_ALLOCATOR_H

#include <memory_resource>
#include <vector>

struct ChunkRecord {
    static ChunkRecord
    make(void *ptr) {
        return ChunkRecord{
            .start_ptr = ptr,
            .current_ptr = ptr,
        };
    }

    void *start_ptr;
    void *current_ptr;
};

template <size_t CHUNK_SIZE = 8 * 8192> class ChunkAllocator {
public:
    explicit ChunkAllocator() {
        void *next_chunk = std::aligned_alloc(8, CHUNK_SIZE);
        m_chunks.push_back(ChunkRecord::make(next_chunk));
    }

    template <typename T>
    T *
    allocate() {
        static constexpr size_t T_SIZE = sizeof(T);
        static constexpr size_t T_ALIGN = alignof(T);
        static_assert(T_SIZE <= CHUNK_SIZE);

        if (m_bytes_remaining < T_SIZE) {
            void *next_chunk = std::aligned_alloc(8, CHUNK_SIZE);
            m_chunks.push_back(ChunkRecord::make(next_chunk));
            m_bytes_remaining = CHUNK_SIZE;
        }

        ChunkRecord &chunk = m_chunks[m_chunks.size() - 1];

        auto return_ptr =
            std::align(T_ALIGN, T_SIZE, chunk.current_ptr, m_bytes_remaining);
        // return_ptr can be nullptr if space is too small, but that was already checked
        assert(return_ptr != nullptr);

        chunk.current_ptr = (std::byte *)return_ptr + T_SIZE;
        return reinterpret_cast<T *>(return_ptr);
    }

    ~ChunkAllocator() {
        for (auto &chunk : m_chunks) {
            std::free(chunk.start_ptr);
        }
    }

private:
    std::vector<ChunkRecord> m_chunks{};
    size_t m_bytes_remaining{CHUNK_SIZE};
};

#endif // PT_CHUNK_ALLOCATOR_H
