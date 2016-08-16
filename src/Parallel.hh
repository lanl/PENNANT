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

#include "GlobalMesh.hh"
#include "InputParameters.hh"
#include "Vec2.hh"

#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

// Parallel provides helper functions and variables for
// running in distributed parallel mode using Legion.

enum ZoneFields {
	FID_ZR,
	FID_ZRP,
	FID_ZE,
	FID_ZP,
	FID_PX,
	FID_PF,
	FID_PMASWT,
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

class Parallel {
public:
	// TODO fix these
    static int num_subregions() {return 1;}           // number of MPI PEs in use
                                // (1 if not using MPI)
    static int mype() { return 0; }            // PE number for my rank
                                // (0 if not using MPI)

    Parallel(InputParameters input_params,
    		Context ctx, HighLevelRuntime *runtime);
    ~Parallel();
    void run(InputParameters input_params);

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

private:
	GlobalMesh global_mesh_;
	std::vector<void*> serializer;
	MustEpochLauncher must_epoch_launcher;
	Context ctx_;
	HighLevelRuntime *runtime_;
};  // class Parallel

struct SPMDArgs {
	DynamicCollective add_reduction_;
	DynamicCollective min_reduction_;
	int shard_id_;
	DirectInputParameters direct_input_params_;
    // Legion cannot handle data structures with indirections in them
    int n_meshtype_;
    int n_probname_;
    int n_bcx_;
    int n_bcy_;
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
