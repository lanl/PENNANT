cmake_minimum_required(VERSION 3.10)

project(pennant LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Preferred way of working: don't change options in this file. Instead, call cmake with
# e.g. "-DPENNANT_MPI=OFF" if you want to change that setting
option(PENNANT_MPI "Build with MPI support" ON)
option(PENNANT_TIMERS "Put timers around important code sections and print stats" OFF)
option(PENNANT_FAST_MATH "Use math optimizations that may not be IEEE 764 compliant" OFF)
option(PENNANT_ARCH_GFX900 "Build with support for gfx900 (vega10)" OFF)
option(PENNANT_ARCH_GFX906 "Build with support for gfx906 (vega20, MI50/60)" ON)
option(PENNANT_ARCH_GFX908 "Build with support for gfx908 (vega7nm, MI100)" ON)
option(PENNANT_ARCH_GFX90A "Build with support for gfx90a (MI200)" OFF)

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
  HSA_HEADER
  ROCclr_DIR
  amd_comgr_DIR
  hsa-runtime64_DIR)

# Use hipcc for compiling and linking
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

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

# PENNANT source file that handles MPI comms, if MPI is supported
if(PENNANT_MPI)
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
list(APPEND PENNANT_COMPILATION_OPTIONS "-O3" "-Wall" "-Werror")

if(PENNANT_FAST_MATH)
  list(APPEND PENNANT_COMPILATION_OPTIONS "-ffp-contract=fast" "-ffast-math" "-funsafe-math-optimizations")
endif()

if(PENNANT_ARCH_GFX900)
  list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx900")
endif()
if(PENNANT_ARCH_GFX906)
  list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx906")
endif()
if(PENNANT_ARCH_GFX908)
  list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx908")
endif()
if(PENNANT_ARCH_GFX90A)
  list(APPEND PENNANT_COMPILATION_OPTIONS "--amdgpu-target=gfx90a")
endif()

target_compile_options(pennant PRIVATE ${PENNANT_COMPILATION_OPTIONS})

# Optionally configure MPI 
if(PENNANT_MPI)
  find_package(MPI REQUIRED)

  target_compile_definitions(pennant PRIVATE USE_MPI)
  target_link_libraries(pennant PRIVATE MPI::MPI_CXX)
endif()

# Optionally configure the use of timing code
if(PENNANT_TIMERS)
  target_compile_definitions(pennant PRIVATE USE_TIMERS)
endif()
