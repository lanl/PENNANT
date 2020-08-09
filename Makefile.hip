BUILDDIR := build_hip
PRODUCT := pennant
# uncomment the line below, or set MPI as an environment variable, to compile with MPI support
# MPI := OpenMPI
# uncomment the line below, or set an environment variable JIT to any value, to compile with JIT support
# JIT = 1
# uncomment the line below, or set an environment variable TIMERS to any value, to compile with Timers support
# TIMERS = 1

ifndef ROCM_PATH
    ROCM_PATH := /opt/rocm
endif

SRCDIR := src.hip
THRUSTDIR := ${ROCM_PATH}/rocthrust
PRIMDIR := ${ROCM_PATH}/rocprim

HDRS := $(wildcard $(SRCDIR)/*.hh)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(SRCS:$(SRCDIR)/%.cc=$(BUILDDIR)/%.o)
DEPS := $(SRCS:$(SRCDIR)/%.cc=$(BUILDDIR)/%.d)

BINARY := $(BUILDDIR)/$(PRODUCT)

CPPFLAGS := -I. -I$(THRUSTDIR)/include -I$(PRIMDIR)/include
CPPFLAGS += -Wall -Werror
CPPFLAGS += -Wno-unused-command-line-argument

# hipcc flags:
CXX := ${ROCM_PATH}/bin/hipcc
CXXFLAGS_DEBUG := -g
CXXFLAGS_OPT := -O3 -g

# Timer support
TIMERSDIR := timers
CPPFLAGS += -I$(TIMERSDIR)
ifdef TIMERS
    TM_HDRS := $(wildcard $(TIMERSDIR)/*.h)
    TM_SRCS := $(wildcard $(TIMERSDIR)/*.cpp)
    TM_OBJS := $(TM_SRCS:$(TIMERSDIR)/%.cpp=$(BUILDDIR)/%.o)
    TM_DEPS := $(TM_SRCS:$(TIMERSDIR)/%.cpp=$(BUILDDIR)/%.d)

    HDRS += $(TM_HDRS)
    SRCS += $(TM_SRCS)
    OBJS += $(TM_OBJS)
    DEPS += $(TM_DEPS)

    CPPFLAGS += -DUSE_TIMERS
endif

# JIT support for GPU kernels
ifdef JIT
    PAJAMADIR := pajama

    PJ_HDRS := $(PAJAMADIR)/pajama.h
    PJ_SRCS := $(PAJAMADIR)/pajama.cpp
    PJ_OBJS := $(PJ_SRCS:$(PAJAMADIR)/%.cpp=$(BUILDDIR)/%.o)
    PJ_DEPS := $(PJ_SRCS:$(PAJAMADIR)/%.cpp=$(BUILDDIR)/%.d)

    HDRS += $(PJ_HDRS)
    SRCS += $(PJ_SRCS)
    OBJS += $(PJ_OBJS)
    DEPS += $(PJ_DEPS)

    CPPFLAGS += -DUSE_JIT -I$(PAJAMADIR) 
    CPPFLAGS += -Wno-c++17-extensions
endif

LD := $(CXX)
LDFLAGS := -Wno-unused-command-line-argument

# select optimized or debug
CXXFLAGS := $(CXXFLAGS_OPT) $(CPPFLAGS) -g

# MPI compile and link flags
ifeq ($(MPI),OpenMPI)
	CXXFLAGS += $(shell mpicc.openmpi --showme:compile) -DUSE_MPI
	LDFLAGS += $(shell mpicc.openmpi --showme:link)
endif

# ------------------------------------------------------------
# Are we compiling with/without MPI/JIT? Compiler flags can be hard
# to spot, so we shot the build configuration before and after building

BUILDCONF_MSG := "| Build configuration:"

ifeq ($(findstring -DUSE_MPI,${CXXFLAGS}),-DUSE_MPI)
BUILDCONF_MSG += "| with MPI"
else
BUILDCONF_MSG += "| without MPI"
endif

ifeq ($(findstring -DUSE_JIT,${CXXFLAGS}),-DUSE_JIT)
BUILDCONF_MSG += "| with JIT"
else
BUILDCONF_MSG += "| without JIT"
endif

ifeq ($(findstring -DUSE_TIMERS,${CXXFLAGS}),-DUSE_TIMERS)
BUILDCONF_MSG += "| with Timers"
else
BUILDCONF_MSG += "| without Timers"
endif

BUILDCONF_MSG += "|"

define showconfig
	@echo ${BUILDCONF_MSG}
endef
# ------------------------------------------------------------
all : showconf $(BINARY)
	$(showconfig)

showconf:
	$(showconfig)

-include $(DEPS)

$(BINARY) : $(OBJS)
	@echo linking $@
	$(maketargetdir)
	$(LD) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	@echo compiling $<
	$(maketargetdir)
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

$(BUILDDIR)/%.o : $(PAJAMADIR)/%.cpp
	@echo compiling $<
	$(maketargetdir)
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

$(BUILDDIR)/%.o : $(TIMERSDIR)/%.cpp
	@echo compiling $<
	$(maketargetdir)
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

$(BUILDDIR)/%.d : $(SRCDIR)/%.cc
	@echo making depends for $<
	$(maketargetdir)
	@$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -M $< | sed "1s![^ \t]\+\.o!$(@:.d=.o) $@!" >$@

$(BUILDDIR)/%.d : $(PAJAMADIR)/%.cpp
	@echo making depends for $<
	$(maketargetdir)
	@$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -M $< | sed "1s![^ \t]\+\.o!$(@:.d=.o) $@!" >$@

$(BUILDDIR)/%.d : $(TIMERSDIR)/%.cpp
	@echo making depends for $<
	$(maketargetdir)
	@$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -M $< | sed "1s![^ \t]\+\.o!$(@:.d=.o) $@!" >$@

define maketargetdir
	-@mkdir -p $(dir $@) > /dev/null 2>&1
endef

# keep *.d dependency files, otherwise, "make clean" will remake them, and then remove them.
clean :
	rm -f $(BINARY) $(OBJS)

cleanall :
	rm -rf $(BUILDDIR)
