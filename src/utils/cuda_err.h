#ifndef PT_CUDA_ERR_H
#define PT_CUDA_ERR_H

#define CUDA_CHECK(ans)                                                               \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define CUDA_CHECK_LAST_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#endif // PT_CUDA_ERR_H
