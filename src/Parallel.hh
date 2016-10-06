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
	FID_ZX,
	FID_ZXP,
	FID_ZVOL,
    FID_ZAREA,
    FID_ZAREAP,
	FID_ZVOLP,
	FID_ZVOL0,
	FID_ZDL,
	FID_ZRP,
	FID_ZM,
	FID_ZETOT,
	FID_ZW,
	FID_ZWR,
	FID_ZSS,
	FID_ZDU,
};

enum SidesAndCornersFields {
    FID_SAREA,
    FID_SVOL,
    FID_SAREAP,
    FID_SVOLP,
    FID_SSURFP,
    FID_SMF,
    FID_CMASWT,
    FID_SFP,
    FID_SFQ,
    FID_SFT,
    FID_CFTOT,
    FID_SMAP_SIDE_TO_PT1,
    FID_SMAP_SIDE_TO_ZONE,
    FID_SMAP_SIDE_TO_EDGE,
};

enum EdgeFields {
    FID_EX,
    FID_EXP,
    FID_ELEN,
};

enum PointFields {
	FID_PF,
	FID_PMASWT,
	FID_GHOST_PF,
	FID_GHOST_PMASWT,
	FID_PX0,
	FID_PX,
    FID_PXP,
    FID_PU,
    FID_PU0,
    FID_PAP,
};

enum ZonePtsCRSFields {
	FID_ZONE_PTS_PTR,
};

struct RunStat {
	int cycle;
	double time;
};

struct TimeStep {
	double dt;
	char message[80];
	TimeStep() {
		dt = std::numeric_limits<double>::max();
		snprintf(message, 80, "Error: uninitialized");
	}
	TimeStep(const TimeStep &copy) {
		dt = copy.dt;
		snprintf(message, 80, "%s", copy.message);
	}
	inline friend bool operator<(const TimeStep &l, const TimeStep &r) {
		return l.dt < r.dt;
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
		return l.dt == r.dt;
	}
	inline friend bool operator!=(const TimeStep &l, const TimeStep &r) {
		return !(l == r);
	}
};

enum TaskIDs {
	TOP_LEVEL_TASK_ID,
	DRIVER_TASK_ID,
    CORRECTOR_TASK_ID,
    WRITE_TASK_ID,
	GLOBAL_SUM_TASK_ID,
	GLOBAL_MIN_TASK_ID,
    ADD_REDOP_ID,
    ADD2_REDOP_ID,
	MIN_REDOP_ID,
};

typedef RegionAccessor<AccessorType::Generic, double> DoubleAccessor;
typedef RegionAccessor<AccessorType::Generic, double2> Double2Accessor;
typedef RegionAccessor<AccessorType::Generic, int> IntAccessor;


class Parallel {
public:

    static void run(InputParameters input_params,
    		Context ctx, HighLevelRuntime *runtime);

    // TODO use Legion
    static void globalSum(int& x);     // find sum over all PEs - overloaded
    static void globalSum(int64_t& x);
    // TODO Export Gold stuff to be converted to Legion
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

class ArgsSerializer {
public:
    ArgsSerializer() {bit_stream = nullptr; bit_stream_size = 0; free_bit_stream = false;}
    ~ArgsSerializer() {if (free_bit_stream) free(bit_stream);}
    void* getBitStream();
    size_t getBitStreamSize();
    void setBitStream(void* bit_stream);
protected:
    void* bit_stream;
    size_t bit_stream_size;
    bool free_bit_stream;
};

class SPMDArgsSerializer : public ArgsSerializer {
public:
    void archive(SPMDArgs* spmd_args);
    void restore(SPMDArgs* spmd_args);
};

struct CorrectorTaskArgs {
    double dt;
    double cfl;
    double cflv;
    int num_points;
    int num_sides;
    int num_zones;
    int my_color;
    int num_subregions;
    int nzones_x, nzones_y;
    std::vector<int> zone_chunk_CRS;
    std::vector<int> side_chunk_CRS;
    std::vector<int> point_chunk_CRS;
    std::string meshtype;
    std::vector<double> bcx, bcy;
};

class CorrectorTaskArgsSerializer : public ArgsSerializer {
public:
    void archive(CorrectorTaskArgs* hydro_task2_args);
    void restore(CorrectorTaskArgs* hydro_task2_args);
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
