/*
 * Parallel.hh
 *
 *  Created on: May 31, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef PARALLEL_HH_
#define PARALLEL_HH_

#include <limits>
#include <stdint.h>

#include "InputParameters.hh"
#include "Vec2.hh"

#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

// TODO remove circular dependencies by breaking this class up

// Parallel provides helper functions and variables for
// running in distributed parallel mode using Legion.

enum ZoneFields {
	FID_ZR,
	FID_ZE,
	FID_ZP,
};

enum PointFields {
	FID_PF,
	FID_PMASWT,
	FID_GHOST_PF,
	FID_GHOST_PMASWT,
};

enum ZonePtsCRSFields {
	FID_ZONE_PTS_PTR,
};

struct RunStat {
	int cycle;
	double time;
};

struct TimeStep {
	double dt_;
	char message_[80];
	TimeStep() {
		dt_ = std::numeric_limits<double>::max();
		snprintf(message_, 80, "Error: uninitialized");
	}
	TimeStep(const TimeStep &copy) {
		dt_ = copy.dt_;
		snprintf(message_, 80, "%s", copy.message_);
	}
	inline friend bool operator<(const TimeStep &l, const TimeStep &r) {
		return l.dt_ < r.dt_;
	}
	inline friend bool operator>(const TimeStep &l, const TimeStep &r) {
		return r < l;
	}
	inline friend bool operator<=(const TimeStep &l, const TimeStep &r) {
		return !(l > r);
	}
	inline friend bool operator>=(const TimeStep &l, const TimeStep &r) {
		return !(l < r);
	}
	inline friend bool operator==(const TimeStep &l, const TimeStep &r) {
		return l.dt_ == r.dt_;
	}
	inline friend bool operator!=(const TimeStep &l, const TimeStep &r) {
		return !(l == r);
	}
};

enum TaskIDs {
	TOP_LEVEL_TASK_ID,
	DRIVER_TASK_ID,
	WRITE_TASK_ID,
	GLOBAL_SUM_TASK_ID,
	GLOBAL_MIN_TASK_ID,
	ADD_REDOP_ID,
	MIN_REDOP_ID,
};

typedef RegionAccessor<AccessorType::Generic, double> DoubleAccessor;
typedef RegionAccessor<AccessorType::Generic, double2> Double2Accessor;
typedef RegionAccessor<AccessorType::Generic, int> IntAccessor;


class Parallel {
public:
	// TODO fix these
    //static int num_subregions() {return 1;}           // number of MPI PEs in use
                                // (1 if not using MPI)
    //static int mype() { return 0; }            // PE number for my rank
                                // (0 if not using MPI)

    static void run(InputParameters input_params,
    		Context ctx, HighLevelRuntime *runtime);

    // TODO use Legion
    static void globalSum(int& x);     // find sum over all PEs - overloaded
    static void globalSum(int64_t& x);
    //EXport GOld stuff to be converted to Legion
    static void globalSum(double& x);
    static void gather(const int x, int* y);
                                // gather list of ints from all PEs
    static void scatter(const int* x, int& y);
                                // gather list of ints from all PEs

    template<typename T>
    static void gatherv(               // gather variable-length list
            const T *x, const int numx,
            T* y, const int* numy);
    template<typename T>
    static void gathervImpl(           // helper function for gatherv
            const T *x, const int numx,
            T* y, const int* numy);
    // Legion stuff
    static Future globalSum(double local_value,
			  DynamicCollective& dc_reduction,
			  Runtime *runtime, Context ctx,
			  Predicate pred = Predicate::TRUE_PRED);
	static const TaskID sumTaskID = GLOBAL_SUM_TASK_ID;
	static double globalSumTask(const Task *task,
					const std::vector<PhysicalRegion> &regions,
					Context ctx, HighLevelRuntime *runtime);

	static Future globalMin(TimeStep local_value,
			  DynamicCollective& dc_reduction,
			  Runtime *runtime, Context ctx,
			  Predicate pred = Predicate::TRUE_PRED);
	static const TaskID minTaskID = GLOBAL_MIN_TASK_ID;
	static TimeStep globalMinTask(const Task *task,
					const std::vector<PhysicalRegion> &regions,
					Context ctx, HighLevelRuntime *runtime);
};  // class Parallel

struct SPMDArgs {
	DynamicCollective add_reduction;
	DynamicCollective min_reduction;
	DirectInputParameters direct_input_params;
    std::string meshtype;
    std::string probname;
    std::vector<double> bcx;
    std::vector<double> bcy;
	PhaseBarrier pbarrier_as_master;
	std::vector<PhaseBarrier> masters_pbarriers;
};

class SPMDArgsSerializer {
public:
    SPMDArgsSerializer() {spmd_args = nullptr; bit_stream = nullptr; bit_stream_size = 0; free_bit_stream = false;}
    ~SPMDArgsSerializer() {if (free_bit_stream) free(bit_stream);}
    void archive(SPMDArgs* spmd_args);
    void* getBitStream();
    size_t getBitStreamSize();

    void restore(SPMDArgs* spmd_args);
    void setBitStream(void* bit_stream);
private:
    SPMDArgs* spmd_args;
    void* bit_stream;
    size_t bit_stream_size;
    bool free_bit_stream;
};

enum Variants {
  CPU_VARIANT,
  GPU_VARIANT,
};

namespace TaskHelper {
  template<typename T>
  void base_cpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
    T::cpu_run(task, regions, ctx, runtime);
  }

#ifdef USE_CUDA
  template<typename T>
  void base_gpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
	const int *p = (int*)task->local_args;
    T::gpu_run(*p, regions);
  }
#endif

  template<typename T>
  void register_cpu_variants(void)
  {
    HighLevelRuntime::register_legion_task<base_cpu_wrapper<T> >(T::TASK_ID, Processor::LOC_PROC,
                                                                 false/*single*/, true/*index*/,
                                                                 CPU_VARIANT,
                                                                 TaskConfigOptions(T::CPU_BASE_LEAF),
                                                                 T::TASK_NAME);
  }

};

#endif /* PARALLEL_HH_ */
