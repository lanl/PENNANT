#include <hip/hip_runtime.h>
#include "../src.hip/Vec2.hh"

constexpr int CHUNK_SIZE = ${CHUNK_SIZE};

extern "C" {

  //-- gpuMain1 ----------------------------------------------------------
  __launch_bounds__(256)
  __global__ void gpuMain1_jit(double dt)
  {
    constexpr int nump = ${nump};
    const double2* const px = ${px};
    const double2* const pu = ${pu};
    double2* const pu0 = ${pu0};
    double2* const pxp = ${pxp};

    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    double dth = 0.5 * dt;

    // save off point variable values from previous cycle
    pu0[p] = pu[p];
    pxp[p] = px[p] + pu0[p] * dth;

    // ===== Predictor step =====
    // 1. advance mesh to center of time step
  }

} // extern "C"
