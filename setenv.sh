#!/bin/bash

module load craype-accel-amd-gfx90a
module load rocm

########## process args ##########

use_gpu=1  # gpu-direct MPI
env="cray" # amd is also a valid choice
while [ $# -ge 1 ]; do
  case $1 in
    --env)
      env=$2
      shift;
      ;;
    --indirect)
      use_gpu=0
      ;;
    *)
      echo "Unknown Option: $1"
      exit 1
      ;;
  esac
  shift;
done

if [ $use_gpu -eq 1 ]; then
    ## This must be set before running with gpu-aware MPI.
    # It is picked up by cray-mpich
    export MPICH_GPU_SUPPORT_ENABLED=1
    # This is to tell the source code about it:
    export CXXFLAGS="-DUSE_GPU_AWARE_MPI"
else
    export CXXFLAGS=""
fi

## This fixes a bug in cray-mpich that shows up at scale
#export MPICH_SMP_SINGLE_COPY_MODE=CMA
## This fixes a cray-mpich compilation setup issue on crusher
#LDFLAGS="-Wl,--allow-shlib-undefined"

if [[ x"$env" == x"amd" ]]; then
  module load PrgEnv-amd
  module load cray-mpich
  MPI_HOME="$MPICH_DIR"
  export CXX=`which hipcc`
  export CXXFLAGS="$CXXFLAGS -DUSE_MPI -I$MPI_HOME/include -amdgpu-target=gfx90a -D__HIP_ARCH_GFX90A__=1"
  export LDFLAGS="$LDFLAGS -amdgpu-target=gfx90a -L$MPI_HOME/lib -lmpi $PE_MPICH_GTL_DIR_amd_gfx90a -lmpi_gtl_hsa -L$ROCM_PATH/lib -lhsa-runtime64"
else
  module load PrgEnv-cray
  module load cray-mpich
  MPI_HOME="$MPICH_DIR"
  export CXX=`which CC`
  export CXXFLAGS="$CXXFLAGS -DUSE_MPI"
  export CXXFLAGS="$CXXFLAGS -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip"
  export LDFLAGS="$LDFLAGS --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64"
fi

# Checking for integer overflows and other nonsense:
# http://embed.cs.utah.edu/ioc/
# https://www.cs.utah.edu/~regehr/papers/tosem15.pdf
# https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
# alternately, -fsanitize=undefined

# Uncomment these 3 lines to run int-arithmetic checker:
#export CXXFLAGS="$CXXFLAGS -fsanitize=integer -g -fno-omit-frame-pointer"
#export LDFLAGS="$LDFLAGS -fsanitize=integer -g -fno-omit-frame-pointer"
#export UBSAN_OPTIONS="print_stacktrace=1"
# TODO: use a wrapper script to set this per-node:
# export UBSAN_OPTIONS=log_path=...

# cray mpi lives in /opt/cray/pe/lib64, but is found from PrgEnv-cray?
#export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
