#ifndef PT_CUDA_BOX_H
#define PT_CUDA_BOX_H

template <class T> class CudaBox {
public:
    CudaBox() { CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(T))) }

    explicit CudaBox(T *item) : CudaBox() { set(item); }

    void
    set(T *item) {
        assert(ptr != 0);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(ptr), item, sizeof(T),
                              cudaMemcpyHostToDevice))
    }

    ~CudaBox(){CUDA_CHECK(cudaFree(reinterpret_cast<void *>(ptr)))}

    CUdeviceptr get_ptr() const {
        return ptr;
    }

private:
    CUdeviceptr ptr{};
};

#endif // PT_CUDA_BOX_H
