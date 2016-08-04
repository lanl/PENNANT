/*
 * Parallel.cc
 *
 *  Created on: May 31, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Parallel.hh"

#include <vector>
#include <algorithm>
#include <numeric>

#include "AddReductionOp.hh"
#include "Driver.hh"
#include "MinReductionOp.hh"
#include "Vec2.hh"


namespace Parallel {

// We're in serial mode, so only 1 PE.
int num_subregions = 1;
int mype = 0;
MustEpochLauncher must_epoch_launcher;
Context ctx_;
HighLevelRuntime *runtime_;

void init(InputParameters input_params,
		Context ctx, HighLevelRuntime *runtime) {
	  num_subregions = input_params.ntasks_;

	  ctx_ = ctx;
	  runtime_ = runtime;

	  // we're going to use a must epoch launcher, so we need at least as many
	  //  processors in our system as we have subregions - check that now
	  std::set<Processor> all_procs;
	  Realm::Machine::get_machine().get_all_processors(all_procs);
	  int num_loc_procs = 0;
	  for(std::set<Processor>::const_iterator it = all_procs.begin();
	      it != all_procs.end();
	      it++)
	    if((*it).kind() == Processor::LOC_PROC)
	      num_loc_procs++;

	  if(num_loc_procs < num_subregions) {
	    printf("FATAL ERROR: This test uses a must epoch launcher, which requires\n");
	    printf("  a separate Realm processor for each subregion.  %d of the necessary\n",
		   num_loc_procs);
	    printf("  %d are available.  Please rerun with '-ll:cpu %d'.\n",
		   num_subregions, num_subregions);
	    exit(1);
	  }

	  Rect<1> launch_bounds(Point<1>(0),Point<1>(num_subregions-1));

	  double zero = 0.0;
	  DynamicCollective add_reduction =
		runtime_->create_dynamic_collective(ctx_, num_subregions, AddReductionOp::redop_id,
						   &zero, sizeof(zero));

	  TimeStep max;
	  DynamicCollective min_reduction =
		runtime_->create_dynamic_collective(ctx_, num_subregions, MinReductionOp::redop_id,
						   &max, sizeof(max));

	  std::vector<SPMDArgs> args(num_subregions);

	  for (int color = 0; color < num_subregions; color++) {
		  args[color].add_reduction_ = add_reduction;
		  args[color].min_reduction_ = min_reduction;
		  args[color].shard_id_ = color;
		  args[color].input_params_ = input_params;

		  DriverTask driver_launcher(&(args[color]));
		  DomainPoint point(color);
		  must_epoch_launcher.add_single_task(point, driver_launcher);
	  }

}  // init


void run() {
	  FutureMap fm = runtime_->execute_must_epoch(ctx_, must_epoch_launcher);
	  fm.wait_all_results();
}

void finalize() {
}  // final


void globalMinLoc(double& x, int& xpe) {
    if (num_subregions == 1) {
        xpe = 0;
        return;
    }
#ifdef USE_MPI
    struct doubleInt {
        double d;
        int i;
    } xdi, ydi;
    xdi.d = x;
    xdi.i = mype;
    MPI_Allreduce(&xdi, &ydi, 1, MPI_DOUBLE_INT, MPI_MINLOC,
            MPI_COMM_WORLD);
    x = ydi.d;
    xpe = ydi.i;
#endif
}


void globalSum(int& x) {
    if (num_subregions == 1) return;
#ifdef USE_MPI
    int y;
    MPI_Allreduce(&x, &y, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    x = y;
#endif
}


void globalSum(int64_t& x) {
    if (num_subregions == 1) return;
#ifdef USE_MPI
    int64_t y;
    MPI_Allreduce(&x, &y, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    x = y;
#endif
}


void globalSum(double& x) {
    if (num_subregions == 1) return;
#ifdef USE_MPI
    double y;
    MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    x = y;
#endif
}


void gather(int x, int* y) {
    if (num_subregions == 1) {
        y[0] = x;
        return;
    }
#ifdef USE_MPI
    MPI_Gather(&x, 1, MPI_INT, y, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
}


void scatter(const int* x, int& y) {
    if (num_subregions == 1) {
        y = x[0];
        return;
    }
#ifdef USE_MPI
    MPI_Scatter((void*) x, 1, MPI_INT, &y, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
}


template<typename T>
void gathervImpl(
        const T *x, const int numx,
        T* y, const int* numy) {

    if (num_subregions == 1) {
        std::copy(x, x + numx, y);
        return;
    }
#ifdef USE_MPI
    const int type_size = sizeof(T);
    int sendcount = type_size * numx;
    std::vector<int> recvcount, disp;
    if (mype == 0) {
        recvcount.resize(num_subregions);
        for (int pe = 0; pe < num_subregions; ++pe) {
            recvcount[pe] = type_size * numy[pe];
        }
        // exclusive scan isn't available in the standard library,
        // so we use an inclusive scan and displace it by one place
        disp.resize(num_subregions + 1);
        std::partial_sum(recvcount.begin(), recvcount.end(), &disp[1]);
    } // if mype

    MPI_Gatherv((void*) x, sendcount, MPI_BYTE,
            y, &recvcount[0], &disp[0], MPI_BYTE,
            0, MPI_COMM_WORLD);
#endif

}


template<>
void gatherv(
        const double2 *x, const int numx,
        double2* y, const int* numy) {
    gathervImpl(x, numx, y, numy);
}


template<>
void gatherv(
        const double *x, const int numx,
        double* y, const int* numy) {
    gathervImpl(x, numx, y, numy);
}


template<>
void gatherv(
        const int *x, const int numx,
        int* y, const int* numy) {
    gathervImpl(x, numx, y, numy);
}


}  // namespace Parallel

