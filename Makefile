# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

LG_RT_DIR := ${HOME}/github/legion/runtime
GASNET_ROOT= ${HOME}/opt/gasnet/1.24.0-hack
GASNET=$(GASNET_ROOT)
CONDUIT=ibv
USE_GASNET=1

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_INFO	# Compile time logging level
SHARED_LOWLEVEL ?= 0		# Use shared-memory runtime (not recommended)
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= pennant
# List all the application source files here
GEN_SRC		?= $(wildcard src/*.cc)		# .cu files
GEN_GPU_SRC	?= 				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?= -std=c++11 -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-variable -D__STDC_FORMAT_MACROS -DDISABLE_BARRIER_MIGRATION
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

src/Driver.o: src/Driver.hh src/Parallel.hh src/Mesh.hh src/Hydro.hh
src/ExportGold.o: src/ExportGold.hh src/Parallel.hh src/Vec2.hh src/Mesh.hh
src/GenerateMesh.o: src/GenerateMesh.hh src/Parallel.hh src/Vec2.hh \
	src/InputParameters.hh src/Parallel.hh
src/GlobalMesh.o: src/GlobalMesh.hh src/GenerateMesh.hh
src/Hydro.o: src/Hydro.hh src/Parallel.hh src/Memory.hh src/Mesh.hh \
	src/PolyGas.hh src/TTS.hh src/QCS.hh src/HydroBC.hh \
	src/InputParameters.hh
src/HydroBC.o: src/HydroBC.hh src/Memory.hh src/Mesh.hh src/Vec2.hh \
	src/Parallel.hh
src/InputFile.o: src/InputFile.hh
src/Memory.o: src/Memory.hh
src/Mesh.o: src/Mesh.hh src/Vec2.hh src/Memory.hh src/Parallel.hh \
	src/WriteXY.hh src/ExportGold.hh src/GenerateMesh.hh src/Vec2.hh \
	src/InputParameters.hh
src/Parallel.o: src/Parallel.hh src/Vec2.hh src/AddReductionOp.hh \
	src/Driver.hh src/MinReductionOp.hh src/GlobalMesh.hh
src/PolyGas.o: src/PolyGas.hh src/Memory.hh src/Hydro.hh src/Mesh.hh \
	src/Vec2.hh src/InputParameters.hh src/Parallel.hh
src/QCS.o: src/QCS.hh src/Memory.hh src/Vec2.hh src/Mesh.hh src/Hydro.hh \
	src/InputParameters.hh
src/TTS.o: src/TTS.hh src/Vec2.hh src/Mesh.hh src/Hydro.hh \
	src/InputParameters.hh src/Parallel.hh
src/WriteXY.o: src/WriteXY.hh src/Parallel.hh src/Mesh.hh
src/WriteTask.o: src/WriteTask.hh src/Parallel.hh
src/main.o: src/Parallel.hh src/InputParameters.hh src/InputFile.hh \
	src/AddReductionOp.hh src/MinReductionOp.hh src/Driver.hh \
	src/WriteTask.hh

