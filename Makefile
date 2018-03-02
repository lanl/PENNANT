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

LG_RT_DIR ?= ${HOME}/Projects/src/legion/runtime
GASNET_ROOT ?= ${HOME}/public/install/GASNet-1.28.0-fPIC
GASNET ?= $(GASNET_ROOT)
CONDUIT ?= mpi
USE_GASNET ?= 1
NPX ?= 1

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
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
CC_FLAGS	?= -std=c++11 -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-variable -D__STDC_FORMAT_MACROS -DDISABLE_BARRIER_MIGRATION -I$(LG_RT_DIR)/realm -I$(LG_RT_DIR)/legion
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?= -L/usr/lib64 -lpmi -g #-pg
LEGION_LD_FLAGS	?= -L/usr/lib64 -lpmi -g #-pg

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

