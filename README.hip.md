# PENNANT HIP version

The HIP version of PENNANT, intended to run in AMD GPUs, is based off the 
cuda branch of the PENNANT repository at https://github.com/lanl/PENNANT
git@github.com. The current HIP version is a straightforward HIP port, without
any optimizations. 

## Prerequisites

The HIP version of PENNANT depends assumes a ROCm 3.0 install (older versions
may work too), and on the rocThrust and rocPrim libearies. 
See https://github.com/ROCmSoftwarePlatform/rocThrust and
https://github.com/ROCmSoftwarePlatform/rocPrim, or install these libraries on
Ubuntu systems with
```
sudo apt install rocthrust rocprim
```

## Building

To build PENNANT:

```
make -f Makefile.hip -j `nproc`
```

This will build the PENNANT binary in directory build_hip.

## Changes w.r.t. the CUDA version

The CUDA version contains only two source files with CUDA code: `HydroGPU.cu`
and `Vec2.hh`, both in the `src` directory. We hipified these into a new
`src.hip` directory, and added symlinks to the remaining files that needed
no hipification. Some manual changes were required to get the hipified
files to compile, and we list the most significant ones here:

* The clang-based HIP compiler does not allow `static __constant__` and
  `static __device__` declarations at the global scope. We replaced
  these with `__constant__` declarations.
* The HIP compiler does not allow the declaration of `__shared__` arrays
  at global scope. Instead, we declared those arrays in `__global__` kernels
  or `__device__` functions, and passed pointers to the `__shared__` arrays
  to `__device__` functions called from those kernels or device functions.

Ideally, instead of modifying the hipified version of PENNANT, we would modify
the CUDA version of PENNANT such that (a) there is no difference in
correctness of performance between the original and modified CUDA versions, and
(b) the hipified version of the modified CUDA version requires no further
changes in order to compile and run. We will look into this for future
versions of the HIP version of PENNANT.
