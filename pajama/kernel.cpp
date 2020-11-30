#include <hip/hip_runtime.h>

constexpr size_t N = ${N};
constexpr size_t no_threads = ${no_threads};
constexpr float a = ${a};

extern "C" {
  __global__ void saxpy(float* x, float* y){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += no_threads){
      y[i] += a * x[i];
    }
  }
}

