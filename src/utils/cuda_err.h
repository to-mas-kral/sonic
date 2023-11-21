#ifndef PT_CUDA_ERR_H
#define PT_CUDA_ERR_H

#include <iostream>

#include <optix.h>
#include <optix_function_table.h>

#define CUDA_CHECK(ans)                                                                  \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define CUDA_CHECK_LAST_ERROR() checkLast(__FILE__, __LINE__)
inline void
checkLast(const char *const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)

inline void
optixCheck(OptixResult res, const char *call, const char *file, unsigned int line) {
    if (res != OPTIX_SUCCESS) {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line
                  << ")\n";
        exit(1);
    }
}

#define OPTIX_CHECK_LOG(call)                                                            \
    do {                                                                                 \
        char LOG[2048];                                                                  \
        size_t LOG_SIZE = sizeof(LOG);                                                   \
        optixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, __FILE__, __LINE__);      \
    } while (false)

inline void
optixCheckLog(OptixResult res, const char *log, size_t sizeof_log,
              size_t sizeof_log_returned, const char *call, const char *file,
              unsigned int line) {
    if (res != OPTIX_SUCCESS) {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line
                  << ")\nLog:\n"
                  << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "")
                  << '\n';
        exit(1);
    }
}

#endif // PT_CUDA_ERR_H
