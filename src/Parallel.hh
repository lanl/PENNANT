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

#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

// Namespace Parallel provides helper functions and variables for
// running in distributed parallel mode using MPI, or for stubbing
// these out if not using MPI.

class Parallel {
public:
    static int num_subregions() {return 1;}           // number of MPI PEs in use
                                // (1 if not using MPI)
    static int mype() { return 0; }            // PE number for my rank
                                // (0 if not using MPI)
	MustEpochLauncher must_epoch_launcher;
	Context ctx_;
	HighLevelRuntime *runtime_;

    void init(InputParameters input_params,
    		Context ctx, HighLevelRuntime *runtime);
    void run();
    void finalize();
    ~Parallel();

    static void globalMinLoc(double& x, int& xpe);
                                // find minimum over all PEs, and
                                // report which PE had the minimum
    static void globalSum(int& x);     // find sum over all PEs - overloaded
    static void globalSum(int64_t& x);
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
private:
	std::vector<void*> indirects;
};  // class Parallel

struct SPMDArgs {
	DynamicCollective add_reduction_;
	DynamicCollective min_reduction_;
	int shard_id_;
	int ntasks_;
	int task_id_;
    double tstop_;                  // simulation stop time
    int cstop_;                     // simulation stop cycle
    double dtmax_;                  // maximum timestep size
    double dtinit_;                 // initial timestep size
    double dtfac_;                  // factor limiting timestep growth
    int dtreport_;                  // frequency for timestep reports
    int chunk_size_;                // max size for processing chunks
    bool write_xy_file_;            // flag:  write .xy file?
    bool write_gold_file_;          // flag:  write Ensight file?
    int nzones_x_, nzones_y_;       // global number of zones, in x and y
                                    // directions
    double len_x_, len_y_;          // length of mesh sides, in x and y
                                    // directions
    double cfl_;                    // Courant number, limits timestep
    double cflv_;                   // volume change limit for timestep
    double rho_init_;               // initial density for main mesh
    double energy_init_;            // initial energy for main mesh
    double rho_init_sub_;           // initial density in subregion
    double energy_init_sub_;        // initial energy in subregion
    double vel_init_radial_;        // initial velocity in radial direction
    double gamma_;                  // coeff. for ideal gas equation
    double ssmin_;                  // minimum sound speed for gas
    double alfa_;                   // alpha coefficient for TTS model
    double qgamma_;                 // gamma coefficient for Q model
    double q1_, q2_;                // linear and quadratic coefficients
                                    // for Q model
    double subregion_xmin_; 		   // bounding box for a subregion
    double subregion_xmax_; 		   // if xmin != std::numeric_limits<double>::max(),
    double subregion_ymin_;         // should have 4 entries:
    double subregion_ymax_; 		   // xmin, xmax, ymin, ymax

    // Legion cannot handle data structures with indirections in them

    int n_meshtype_;
    int n_probname_;
    int n_bcx_;
    int n_bcy_;
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
	GLOBAL_SUM_TASK_ID,
	GLOBAL_MIN_TASK_ID,
	ADD_REDOP_ID,
	MIN_REDOP_ID,
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
