#include <hip/hip_runtime.h>

const int CHUNK_SIZE = 64;

extern "C" {
  __device__ void advPosHalf_jit(
				 const int p,
				 const double2* __restrict__ px0,
				 const double2* __restrict__ pu0,
				 const double dt,
				 double2* __restrict__ pxp) {

    pxp[p] = px0[p] + pu0[p] * dt;

  }

  __global__ void gpuMain1_jit(double dt)
  {
    constexpr int nump = ${nump};
    const double2* const px = ${px};
    double2* const px0 = ${px0};
    const double2* const pu = ${pu};
    double2* const pu0 = ${pu0};
    double2* const pxp = ${pxp};

    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;
  
    double dth = 0.5 * dt;
  
    // save off point variable values from previous cycle
    px0[p] = px[p];
    pu0[p] = pu[p];
  
    // ===== Predictor step =====
    // 1. advance mesh to center of time step
    advPosHalf_jit(p, px0, pu0, dth, pxp);
  }
} // extern "C"
