BUILDDIR := build_hip
PRODUCT := pennant
# uncomment the line below, or set MPI as an environment variable, to compile with MPI support
# MPI := OpenMPI

SRCDIR := src.hip
THRUSTDIR := /opt/rocm/rocthrust
PRIMDIR := /opt/rocm/rocprim

HDRS := $(wildcard $(SRCDIR)/*.hh)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(SRCS:$(SRCDIR)/%.cc=$(BUILDDIR)/%.o)
DEPS := $(SRCS:$(SRCDIR)/%.cc=$(BUILDDIR)/%.d)

BINARY := $(BUILDDIR)/$(PRODUCT)

CPPFLAGS := -I. -I$(THRUSTDIR)/include -I$(PRIMDIR)/include 
CPPFLAGS += -Wall -Werror
CPPFLAGS += -Wno-unused-command-line-argument

# hipcc flags:
CXX := /opt/rocm/bin/hipcc
CXXFLAGS_DEBUG := -g
CXXFLAGS_OPT := -O3

LD := $(CXX)
LDFLAGS := -Wno-unused-command-line-argument

# select optimized or debug
CXXFLAGS := $(CXXFLAGS_OPT) $(CPPFLAGS) -g

# MPI compile and link flags
ifeq ($(MPI),OpenMPI)
	CXXFLAGS += $(shell mpicc.openmpi --showme:compile) -DUSE_MPI
	LDFLAGS += $(shell mpicc.openmpi --showme:link)
endif

ifeq ($(findstring -DUSE_MPI,${CXXFLAGS}),-DUSE_MPI)
MPI_MSG="*** using MPI ***"
else
MPI_MSG="*** NOT using MPI ***"
endif

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

$(BUILDDIR)/%.d : $(SRCDIR)/%.cc
	@echo making depends for $<
	$(maketargetdir)
	@$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -M $< | sed "1s![^ \t]\+\.o!$(@:.d=.o) $@!" >$@

define maketargetdir
	-@mkdir -p $(dir $@) > /dev/null 2>&1
endef

define showconfig
	@echo ${MPI_MSG}
endef

clean :
	rm -f $(BINARY) $(OBJS) $(DEPS)
