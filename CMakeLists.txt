cmake_minimum_required(VERSION 3.10)

project(pennant LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WITH_MPI "Build with MPI support" ON)

find_package(HIP REQUIRED)

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

add_executable(pennant
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
 src.hip/WriteXY.cc
 timers/scoped_timers.cpp)

target_include_directories(pennant PRIVATE timers)

list(APPEND HIP_HIPCC_FLAGS "-O3" "-Wall" "-Werror"
#"-Wno-unused-command-line-argument"
#"-ffp-contract=fast"
#"-ffast-math"
#"-funsafe-math-optimizations"
)
foreach(target ${AMDGPU_TARGETS})
  list(APPEND HIP_HIPCC_FLAGS "--amdgpu-target=${target}")
endforeach()
target_compile_options(pennant PRIVATE ${HIP_HIPCC_FLAGS})

if(WITH_MPI)
  find_package(MPI REQUIRED)

  add_library(hydroMPI src.hip/HydroMPI.cc)
  target_compile_definitions(hydroMPI PRIVATE USE_MPI)
  target_compile_options(hydroMPI PRIVATE ${HIP_HIPCC_FLAGS})
  target_link_libraries(hydroMPI PRIVATE MPI::MPI_CXX)
  
  target_compile_definitions(pennant PRIVATE USE_MPI)
  target_link_libraries(pennant PRIVATE MPI::MPI_CXX)
  target_link_libraries(pennant PRIVATE hydroMPI)
endif()
