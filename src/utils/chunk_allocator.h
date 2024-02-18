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

template <typename T> class ChunkAllocator {
    static constexpr size_t T_SIZE = sizeof(T);
    static constexpr size_t T_ALIGN = alignof(T);
    static constexpr size_t DEFAULT_CHUNK_SIZE = 8 * 8192;

    static_assert(DEFAULT_CHUNK_SIZE % T_ALIGN == 0);

public:
    explicit ChunkAllocator(size_t chunk_size = DEFAULT_CHUNK_SIZE)
        : m_chunk_size{chunk_size}, m_bytes_remaining{chunk_size} {
        if (chunk_size < T_SIZE) {
            throw std::runtime_error("Chunk size is too small for T");
        }

        void *next_chunk = nullptr;
        next_chunk = std::aligned_alloc(T_ALIGN, DEFAULT_CHUNK_SIZE);
        m_chunks.push_back(ChunkRecord::make(next_chunk));
    }

    T *
    allocate() {
        if (m_bytes_remaining < T_SIZE) {
            void *next_chunk = nullptr;
            next_chunk = std::aligned_alloc(T_ALIGN, DEFAULT_CHUNK_SIZE);
            m_chunks.push_back(ChunkRecord::make(next_chunk));
            m_bytes_remaining = DEFAULT_CHUNK_SIZE;
        }

        ChunkRecord &chunk = m_chunks[m_chunks.size() - 1];
        T *return_ptr = reinterpret_cast<T *>(chunk.current_ptr);
        chunk.current_ptr = (unsigned char *)chunk.current_ptr + T_SIZE;
        m_bytes_remaining -= T_SIZE;
        return return_ptr;
    }

    ~ChunkAllocator() {
        for (auto &chunk : m_chunks) {
            std::free(chunk.start_ptr);
        }
    }

private:
    std::vector<ChunkRecord> m_chunks{};
    size_t m_bytes_remaining{DEFAULT_CHUNK_SIZE};
    size_t m_chunk_size{DEFAULT_CHUNK_SIZE};
};

#endif // PT_CHUNK_ALLOCATOR_H
