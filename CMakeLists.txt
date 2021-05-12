cmake_minimum_required(VERSION 3.10)

project(pennant LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The following three lines are not necessary on some systems, while
# leaving them out causes issues with finding HIP and rocBLAS on other
# systems. Beats me.
set(ROCM_PATH /opt/rocm CACHE STRING "Path to ROCm install")
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# Preferred way of working: don't change options in this file. Instead, call cmake with
# e.g. "-DPENNANT_MPI=OFF" if you want to change that setting
option(PENNANT_USE_MPI "Build with MPI support" ON)
option(PENNANT_USE_GPU_AWARE_MPI "Build with GPU-aware MPI. Requires supporting MPI stack" OFF)
option(PENNANT_TIMERS "Put timers around important code sections and print stats" OFF)
option(PENNANT_FAST_MATH "Use math optimizations that may not be IEEE 764 compliant" OFF)
option(PENNANT_ARCH_GFX900 "Build with support for gfx900 (vega10)" OFF)
option(PENNANT_ARCH_GFX906 "Build with support for gfx906 (vega20, MI50/60)" OFF)
option(PENNANT_ARCH_GFX908 "Build with support for gfx908 (vega7nm, MI100)" OFF)
option(PENNANT_ARCH_GFX90A "Build with support for gfx90a (MI200)" OFF)

# Pick your favorite GPU vendor
set(PENNANT_GPU_VENDOR "AMD" CACHE STRING "Are we using AMD or NVIDIA GPUs?")
set_property(CACHE PENNANT_GPU_VENDOR PROPERTY STRINGS "AMD" "NVIDIA")

# Load CMake settings for building HIP code. Mark some CMake variables defined
# by the HIP package as advanced, so that they don't show up by default when
# using ccmake.
find_package(HIP REQUIRED)
mark_as_advanced(FORCE
  GPU_TARGETS
  AMDGPU_TARGETS
  AMDDeviceLibs_DIR
  HIP_CLANG_INCLUDE_PATH
  HIP_DIR
  HIP_VERBOSE_BUILD
  HSA_HEADER
  ROCclr_DIR
  amd_comgr_DIR
  hsa-runtime64_DIR)

# Use hipcc for compiling and linking
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
#set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Basic set of PENNANT source files
set(pennant_sources
  src.hip/Driver.cc
  src.hip/ExportGold.cc
  src.hip/GenMesh.cc
  src.hip/HydroBC.cc
  src.hip/Hydro.cc
  src.hip/HydroGPU.cc
  src.hip/ImportGMV.cc
  src.hip/InputFile.cc
  src.hip/main.cc
  src.hip/Mesh.cc
  src.hip/Parallel.cc
  src.hip/PolyGas.cc
  src.hip/QCS.cc
  src.hip/TTS.cc
  src.hip/WriteXY.cc)

if(PENNANT_USE_GPU_AWARE_MPI)
  SET(PENNANT_USE_MPI ON CACHE BOOL "Build with MPI support" FORCE)
endif()

# PENNANT source file that handles MPI comms, if MPI is supported
if(PENNANT_USE_MPI)
  list(APPEND pennant_sources src.hip/HydroMPI.cc)
endif()

# PENNANT source file with functionality for timing various sections of the code
if(PENNANT_TIMERS)
  list(APPEND pennant_sources timers/scoped_timers.cpp)
endif()

# At this point, the list of sources is complete, and we can add them to the executable
add_executable(pennant ${pennant_sources})

# One timers include file is always included, even if we're not using
# timers; in that case, the macros in the include file evaluate to the empty string
target_include_directories(pennant PRIVATE timers)

# compilation options
# todo: differentiate between Debug, Release, RelWithDebug builds
if(PENNANT_GPU_VENDOR STREQUAL "AMD")
  list(APPEND PENNANT_COMPILATION_OPTIONS "-g" "-O3" "-Wall" "-Wno-unused-result" "-Werror")
endif()
if(PENNANT_GPU_VENDOR STREQUAL "NVIDIA")
  list(APPEND PENNANT_COMPILATION_OPTIONS "-O3" "-Xcompiler" "-Wall -Wno-unknown-pragmas")
endif()

if(PENNANT_FAST_MATH)
  list(APPEND PENNANT_COMPILATION_OPTIONS "-ffp-contract=fast" "-ffast-math" "-funsafe-math-optimizations")
endif()

if(PENNANT_GPU_VENDOR STREQUAL "AMD")
  if(PENNANT_ARCH_GFX900)
    list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx900")
    set(PENNANT_HAVE_ARCH true)
  endif()
  if(PENNANT_ARCH_GFX906)
    list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx906")
    set(PENNANT_HAVE_ARCH true)
  endif()
  if(PENNANT_ARCH_GFX908)
    list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx908")
    set(PENNANT_HAVE_ARCH true)
  endif()
  if(PENNANT_ARCH_GFX90A)
    list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx90a")
    set(PENNANT_HAVE_ARCH true)
  endif()
  if(NOT PENNANT_HAVE_ARCH)
    message(SEND_ERROR "You must specify at least one GPU architecture, e.g. -DPENNANT_ARCH_GFX90A=On")
  endif()
endif()

if(PENNANT_GPU_VENDOR STREQUAL "NVIDIA")
  list(APPEND PENNANT_COMPILATION_OPTIONS "-arch=sm_70")
endif()

target_compile_options(pennant PRIVATE ${PENNANT_COMPILATION_OPTIONS})

# Optionally configure MPI 
if(PENNANT_USE_MPI)
  # To select a specific MPI version, users should define an environment
  # variable MPI_HOME when running cmake or ccmake (only needed once)
  find_package(MPI REQUIRED)
  # show some of the MPI variable if we use ccmake, to help verifying
  # that we use the expected MPI version
  mark_as_advanced(CLEAR
    MPI_CXX_HEADER_DIR
    MPI_CXX_LINK_FLAGS
    MPI_mpi_LIBRARY)

  target_compile_definitions(pennant PRIVATE USE_MPI)
  target_link_libraries(pennant PRIVATE MPI::MPI_CXX)
endif()

if(PENNANT_USE_GPU_AWARE_MPI)
  target_compile_definitions(pennant PRIVATE USE_GPU_AWARE_MPI)
endif()

# Optionally configure the use of timing code
if(PENNANT_TIMERS)
  target_compile_definitions(pennant PRIVATE USE_TIMERS)
endif()
