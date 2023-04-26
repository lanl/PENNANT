# if ROCM_PATH is undefined, issue a 'module load rocm'
# if MPICH_DIR is undefined, issue a 'module load cray-mpich'

BUILDDIR := build
PRODUCT := pennant
# uncomment the line below, or set an environment variable TIMERS to any value, to compile with Timers support
# TIMERS = 1
# uncomment the line below, or set an environment variable ROCTX to any value, to compile with roctracer support
# ROCTX = 1
# THERM = 1

SRCDIR := src
THRUSTDIR := ${ROCM_PATH}/rocthrust
PRIMDIR := ${ROCM_PATH}/rocprim

HDRS := $(wildcard $(SRCDIR)/*.hh)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(SRCS:$(SRCDIR)/%.cc=$(BUILDDIR)/%.o)
DEPS := $(SRCS:$(SRCDIR)/%.cc=$(BUILDDIR)/%.d)

BINARY := $(BUILDDIR)/$(PRODUCT)

CPPFLAGS := -I. -I$(THRUSTDIR)/include -I$(PRIMDIR)/include  -I$(MPICH_DIR)/include -I$(ROCM_PATH)/include 
CPPFLAGS += -std=c++14 -Wall -Werror -DUSE_MPI #-D__HIP_ARCH_GFX908__
CPPFLAGS += -Wno-unused-command-line-argument -amdgpu-target=gfx90a -ffp-contract=fast -ffast-math -funsafe-math-optimizations

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

LD := $(CXX) -O3
LDFLAGS := -Wno-unused-command-line-argument -L${MPICH_DIR}/../../../gtl/lib -lmpi_gtl_hsa -L$(MPICH_DIR)/lib -lmpi

ifdef ROCTX
    CPPFLAGS += -DUSE_ROCTX -I${ROCM_PATH}/roctracer/include
    LDFLAGS += -L${ROCM_PATH}/lib -lroctx64
endif

ifdef THERM
    CPPFLAGS += -DUSE_THERM  -I${ROCM_PATH}/rocm_smi/include
    LDFLAGS += -L${ROCM_PATH}/rocm_smi/lib -lrocm_smi64
endif

# select optimized or debug
CXXFLAGS := $(CXXFLAGS_OPT) $(CPPFLAGS) -g

# ------------------------------------------------------------
# Are we compiling with/without timers and rocTracer support? Compiler flags can be hard
# to spot, so we shot the build configuration before and after building

BUILDCONF_MSG := "| Build configuration:"

ifeq ($(findstring -DUSE_TIMERS,${CXXFLAGS}),-DUSE_TIMERS)
BUILDCONF_MSG += "| with Timers"
else
BUILDCONF_MSG += "| without Timers"
endif

ifeq ($(findstring -DUSE_ROCTX,${CXXFLAGS}),-DUSE_ROCTX)
BUILDCONF_MSG += "| with rocTracer support"
else
BUILDCONF_MSG += "| without rocTracer support"
endif


ifeq ($(findstring -DUSE_THERM,${CXXFLAGS}),-DUSE_THERM)
BUILDCONF_MSG += "| with Thermal output"
else
BUILDCONF_MSG += "| without Thermal output"
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
	rm -f $(BINARY) $(OBJS) build/*.d

cleanall :
	rm -rf $(BUILDDIR)
