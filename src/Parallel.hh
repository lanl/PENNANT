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

// Parallel provides helper functions and variables for
// running in distributed parallel mode using Legion.

enum ZoneFields {
	FID_ZR,
	FID_ZE,
	FID_ZP,
	FID_ZX,
	FID_ZVOL,
    FID_ZAREA,
	FID_ZVOL0,
	FID_ZDL,
	FID_ZM,
	FID_ZETOT,
	FID_ZWR,
	FID_ZSS,
	FID_ZDU,
    FID_Z_DBL2_TEMP,
    FID_Z_DBL_TEMP1,
    FID_Z_DBL_TEMP2,
};

enum SidesAndCornersFields {
    FID_SAREA,
    FID_SVOL,
    FID_SMF,
    FID_CMASWT,
    FID_SFP,
    FID_SFQ,
    FID_SFT,
    FID_CFTOT,
    FID_SMAP_SIDE_TO_PT1,
    FID_SMAP_SIDE_TO_PT2,
    FID_SMAP_SIDE_TO_ZONE,
    FID_SMAP_SIDE_TO_EDGE,
    FID_MAP_CRN2CRN_NEXT,
    FID_S_DBL_TEMP,
};

enum EdgeFields {
    FID_EX,
    FID_E_DBL2_TEMP,
    FID_E_DBL_TEMP,
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
    FID_MAP_PT2CRN_FIRST,
    FID_PT_LOCAL2GLOBAL,
};

enum ZonePtsCRSFields {
	FID_ZONE_PTS_PTR,
    FID_ZONE_CHUNKS_CRS,
    FID_SIDE_CHUNKS_CRS,
    FID_POINT_CHUNKS_CRS,
    FID_BCX_CHUNKS_CRS,
    FID_BCY_CHUNKS_CRS,
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
	inline TimeStep& operator=(const TimeStep &rhs) {
	    if (this != &rhs) {
	        this->dt = rhs.dt;
	        snprintf(this->message, 80, "%s", rhs.message);
	    }
	    return *this;
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
    CALCDT_TASK_ID,
    CORRECTOR_TASK_ID,
    DRIVER_TASK_ID,
    HALO_TASK_ID,
    PREDICTOR_POINT_TASK_ID,
    PREDICTOR_TASK_ID,
    WRITE_TASK_ID,
    GLOBAL_SUM_TASK_ID,
    GLOBAL_SUM_INT64_TASK_ID,
    ADD_REDOP_ID,
    ADD_INT64_REDOP_ID,
    ADD2_REDOP_ID,
	MIN_REDOP_ID,
};

typedef RegionAccessor<AccessorType::SOA<sizeof(double)>, double> DoubleSOAAccessor;
typedef RegionAccessor<AccessorType::SOA<sizeof(double2)>, double2> Double2SOAAccessor;
typedef RegionAccessor<AccessorType::SOA<sizeof(int)>, int> IntSOAAccessor;

typedef RegionAccessor<AccessorType::Generic, double> DoubleAccessor;
typedef RegionAccessor<AccessorType::Generic, double2> Double2Accessor;
typedef RegionAccessor<AccessorType::Generic, int> IntAccessor;
typedef RegionAccessor<AccessorType::Generic, ptr_t> PtrTAccessor;


class Parallel {
public:

    static void run(InputParameters input_params,
            Context ctx, HighLevelRuntime *runtime);

    static Future globalSum(double local_value,
            DynamicCollective& dc_reduction,
            Runtime *runtime, Context ctx,
            Predicate pred = Predicate::TRUE_PRED);
	static const TaskID sumTaskID = GLOBAL_SUM_TASK_ID;
	static double globalSumTask(const Task *task,
	        const std::vector<PhysicalRegion> &regions,
	        Context ctx, HighLevelRuntime *runtime);

    static Future globalSumInt64(int64_t local_value,
            DynamicCollective& dc_reduction,
            Runtime *runtime, Context ctx,
            Predicate pred = Predicate::TRUE_PRED);
    static const TaskID sumInt64TaskID = GLOBAL_SUM_INT64_TASK_ID;
    static int64_t globalSumInt64Task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime);

	static Future globalMin(Future local_value,
	        DynamicCollective& dc_reduction,
	        Runtime *runtime, Context ctx,
	        Predicate pred = Predicate::TRUE_PRED);
};  // class Parallel

struct SPMDArgs {
    DynamicCollective add_reduction;
    DynamicCollective add_int64_reduction;
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


struct DoCycleTasksArgs {
    DynamicCollective min_reduction;
    double cfl;
    double cflv;
    int num_points;
    int num_sides;
    int num_zones;
    int num_edges;
    int my_color;
    int num_subregions;
    int nzones_x, nzones_y;
    double qgamma, q1, q2;
    double ssmin, alpha, gamma;
    int num_zone_chunks;
    int num_side_chunks;
    int num_point_chunks;
    std::string meshtype;
    std::vector<std::vector<int>> boundary_conditions_x;
    std::vector<std::vector<int>> boundary_conditions_y;
    // offsets into concatantated start/stop indeces for boundary pt chunks, compressed row storage
    std::vector<int> bcx_point_chunk_CRS_offsets;
    std::vector<int> bcy_point_chunk_CRS_offsets;
};


class DoCycleTasksArgsSerializer : public ArgsSerializer {
public:
    void archive(DoCycleTasksArgs* hydro_task2_args);
    void restore(DoCycleTasksArgs* hydro_task2_args);
};


struct CalcDtTaskArgs {
    Future dt_hydro;
    TimeStep last;
    double dtmax;
    double dtinit;
    double dtfac;
    double tstop;
    RunStat run_stat;
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
