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

using namespace Legion;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

// Parallel provides helper functions and variables for
// running in distributed parallel mode using Legion.

// FID = field id
// _Z = zone, _S = side, _C = corner, _P = point

// Zone: a quadrangular or hexagonal 2D subvolume of the 2D grid
enum ZoneFields {
  FID_ZR,           // zone rho (mass density per volume)
  FID_ZE,           // zone energy density (per mass)
  FID_ZP,           // zone pressure
  FID_ZX,           // zone 2D location
  FID_ZVOL,         // zone volume (actually an area, since Pennant is 2D)
  FID_ZAREA,      // zone surface area (actually perimeter, since Pennant is 2D)
  FID_ZVOL0,        // zone volume initial (at the start of a timestep)
  FID_ZDL,          // zone characteristic length (?)
  FID_ZM,           // zone total mass
  FID_ZETOT,        // zone total energy
  FID_ZWR,          // zone work rate (dE + P*dV)/dt
  FID_ZSS,          // zone speed of sound
  FID_ZDU,          // zone change in velocity
  // Temporaries
  FID_Z_DBL2_TEMP,  // double2
  FID_Z_DBL_TEMP1,  // double
  FID_Z_DBL_TEMP2,  // double
};

// Side: a triangular 2D subvolume of a zone
// Corner: a quadrangular 2D subvolume of a zone
// Each zone has 4 sides and 4 corners
// Each corner overlaps 2 sides and vice versa
enum SidesAndCornersFields {
  FID_SAREA,   // side area (actually a length, since Pennant is 2D)
  FID_SVOL,    // side volume (fraction of zone volume assigned to this side)
  FID_SMF,     // side mass fraction (side area / zone area)
  FID_CMASWT,  // corner weighted mass (some combination of corner's two sides)
  FID_SFP,     // side 2D force due to pressure
  FID_SFQ,     // side 2D force due to artificial viscosity
  FID_SFT,     // side 2D force due to TTS algorithm
  FID_CFTOT,   // corner total force
  // Maps
  FID_SMAP_SIDE_TO_PT1,  // side start point (farthest clockwise)
  FID_SMAP_SIDE_TO_PT2,  // side end point (farthest counterclockwise)
  FID_SMAP_SIDE_TO_ZONE,  // side's zone
  FID_SMAP_SIDE_TO_EDGE,  // side's edge
  // There is no side-to-corner map; each side has the same index as the corner
  // associated with its start point
  FID_MAP_CRN2CRN_NEXT,  // the next corner moving around the corner's point
  // Temporaries
  FID_S_DBL_TEMP,  // double
};

// Edge: a 1D segment along the edge of a zone
// Each side has 1 edge; each corner has half of 2 edges
enum EdgeFields {
  FID_EX,  // the midpoint between the edge's 2 points
  // Temporaries
  FID_E_DBL2_TEMP,  // double2
  FID_E_DBL_TEMP,   // double
};

// Point: the points defining the edges of the zones
// Each edge or side has 2 points; each corner has 1; each zone has 4 or 6
// Ghost point: a point belonging to a local zone but not owned locally
enum PointFields {
  FID_PF,            // point force
  FID_PMASWT,        // point weighted mass (sum over point's corners)
  FID_GHOST_PF,      // ghost-point force
  FID_GHOST_PMASWT,  // ghost-point weighted mass
  FID_PX0,           // point initial location (at start of timestep)
  FID_PX,            // point 2D location (at end of timestep)
  FID_PXP,           // point partial-step location (at midpoint of timestep)
  FID_PU,            // point 2D velocity (at end of timestep)
  FID_PU0,           // point initial velocity (at start of timestep)
  // Maps
  FID_MAP_PT2CRN_FIRST,  // first corner (iterate using corner-to-corner map)
  FID_PT_LOCAL2GLOBAL,   // point global index
};

enum ZonePtsCRSFields {
  FID_ZONE_PTS_PTR,      //
  FID_ZONE_CHUNKS_CRS,   //
  FID_SIDE_CHUNKS_CRS,   //
  FID_POINT_CHUNKS_CRS,  //
  FID_BCX_CHUNKS_CRS,    //
  FID_BCY_CHUNKS_CRS,    //
};

struct RunStat {
  int cycle;    // number of iterations completed
  double time;  // amount of simulation time elapsed
};

struct TimeStep {
  double dt;  // amount of simulation time elapsed during this iteration
  char message[80];
  TimeStep() {
    dt = std::numeric_limits<double>::max();
    snprintf(message, 80, "Error: uninitialized");
  }
  TimeStep(const TimeStep& copy) {
    dt = copy.dt;
    snprintf(message, 80, "%s", copy.message);
  }
  inline TimeStep& operator=(const TimeStep& rhs) {
    if (this != &rhs) {
      this->dt = rhs.dt;
      snprintf(this->message, 80, "%s", rhs.message);
    }
    return *this;
  }
  inline friend bool operator<(const TimeStep& l, const TimeStep& r) {
    return l.dt < r.dt;
  }
  inline friend bool operator>(const TimeStep& l, const TimeStep& r) {
    return r < l;
  }
  inline friend bool operator<=(const TimeStep& l, const TimeStep& r) {
    return !(l > r);
  }
  inline friend bool operator>=(const TimeStep& l, const TimeStep& r) {
    return !(l < r);
  }
  inline friend bool operator==(const TimeStep& l, const TimeStep& r) {
    return l.dt == r.dt;
  }
  inline friend bool operator!=(const TimeStep& l, const TimeStep& r) {
    return !(l == r);
  }
};

enum TaskIDs {
  // Top level: reads params, sets up Legion, launches DriverTask
  TOP_LEVEL_TASK_ID,  // top_level_task

  // Calculate dt: try to slightly increase the timestep each iteration, but
  // take the minimum of all suggestions returned from various CorrectorTask
  // if that's smaller or use the time to the end of the simulation
  CALCDT_TASK_ID,  // CalcDtTask

  // Corrector: applies boundary conditions, computes the state of the hydro
  // system and updates forces across the grid for the second half of the
  // timestep. Updates energies and suggests the next timestep size.
  CORRECTOR_TASK_ID,  // CorrectorTask

  // Driver: simulation top level, contains parameters, runs iterations
  // Each iteration goes: PredictorPoint, Predictor, Halo, Corrector, CalcDt
  DRIVER_TASK_ID,  // DriverTask

  // Halo summation: does a partial sum of local corner elements to points in
  // the halo of the local chunk
  HALO_TASK_ID,  // HaloTask

  // Point predictor: moves grid points to midpoint of timestep
  PREDICTOR_POINT_TASK_ID,  // PredictorPointTask

  // Predictor: computes the new grid after point movement, then computes the
  // state of the hydrodynamic system and updates forces across the grid up
  // to the midpoint of the timestep
  PREDICTOR_TASK_ID,  // PredictorTask

  // Write: save the simulation results to disk
  WRITE_TASK_ID,  // WriteTask

  // Global sum for double
  GLOBAL_SUM_TASK_ID,  // Parallel::globalSumTask

  // Global sum for integer
  GLOBAL_SUM_INT64_TASK_ID,  // Parallel::globalSumInt64Task

  // Additive reduction operator for double
  ADD_REDOP_ID,  // AddReductionOp

  // Additive reduction operator for integer
  ADD_INT64_REDOP_ID,  // AddInt64ReductionOp

  // Additive reduction operator for double2 (2D vectors)
  ADD2_REDOP_ID,  // Add2ReductionOp

  // Minimum timestep reduction
  MIN_REDOP_ID,  // MinReductionOp
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

  static void run(InputParameters input_params, Context ctx, Runtime* runtime);

  static Future globalSum(double local_value, DynamicCollective& dc_reduction,
      Runtime* runtime, Context ctx, Predicate pred = Predicate::TRUE_PRED);
  static const TaskID sumTaskID = GLOBAL_SUM_TASK_ID;
  static double globalSumTask(const Task* task,
      const std::vector<PhysicalRegion> &regions, Context ctx,
      Runtime* runtime);

  static Future globalSumInt64(int64_t local_value,
      DynamicCollective& dc_reduction, Runtime* runtime, Context ctx,
      Predicate pred = Predicate::TRUE_PRED);
  static const TaskID sumInt64TaskID = GLOBAL_SUM_INT64_TASK_ID;
  static int64_t globalSumInt64Task(const Task* task,
      const std::vector<PhysicalRegion> &regions, Context ctx,
      Runtime* runtime);

  static Future globalMin(Future local_value, DynamicCollective& dc_reduction,
      Runtime* runtime, Context ctx, Predicate pred = Predicate::TRUE_PRED);
};
// class Parallel

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
  ArgsSerializer() {
    bit_stream = nullptr;
    bit_stream_size = 0;
    free_bit_stream = false;
  }
  ~ArgsSerializer() {
    if (free_bit_stream) free(bit_stream);
  }
  void* getBitStream();
  size_t getBitStreamSize();
  void setBitStream(void* bit_stream);
protected:
  void* bit_stream;
  size_t bit_stream_size;
  bool free_bit_stream;
};

class SPMDArgsSerializer: public ArgsSerializer {
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

  // offsets into concatenated start/stop indices for boundary pt chunks, compressed row storage
  std::vector<int> bcx_point_chunk_CRS_offsets;
  std::vector<int> bcy_point_chunk_CRS_offsets;
};

class DoCycleTasksArgsSerializer: public ArgsSerializer {
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
void base_cpu_wrapper(const Task* task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime* runtime) {
  T::cpu_run(task, regions, ctx, runtime);
}

#ifdef USE_CUDA
template<typename T>
void base_gpu_wrapper(const Task* task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  const int* p = (int*)task->local_args;
  T::gpu_run(*p, regions);
}
#endif

template<typename T>
void register_cpu_variants(void) {
  Runtime::register_legion_task<base_cpu_wrapper<T>>(T::TASK_ID,
    Processor::LOC_PROC, false /* single */, true /* index */, CPU_VARIANT,
    TaskConfigOptions(T::CPU_BASE_LEAF), T::TASK_NAME);
}
}

#endif /* PARALLEL_HH_ */
