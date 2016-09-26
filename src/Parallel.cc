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
#include <iostream>
#include <string>

#include "AddReductionOp.hh"
#include "Driver.hh"
#include "GlobalMesh.hh"
#include "MinReductionOp.hh"
#include "Vec2.hh"
#include "WriteTask.hh"


void Parallel::run(InputParameters input_params,
		Context ctx, HighLevelRuntime *runtime)
{
    GlobalMesh global_mesh(input_params, ctx, runtime);
    MustEpochLauncher must_epoch_launcher;
    const int num_subregions_ = input_params.directs_.ntasks_;

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

	  if(num_loc_procs < num_subregions_) {
	    printf("FATAL ERROR: This test uses a must epoch launcher, which requires\n");
	    printf("  a separate Realm processor for each subregion.  %d of the necessary\n",
		   num_loc_procs);
	    printf("  %d are available.  Please rerun with '-ll:cpu %d'.\n",
		   num_subregions_, num_subregions_);
	    exit(1);
	  }

	  Rect<1> launch_bounds(Point<1>(0),Point<1>(num_subregions_-1));

	  double zero = 0.0;
	  DynamicCollective add_reduction =
		runtime->create_dynamic_collective(ctx, num_subregions_, AddReductionOp::redop_id,
						   &zero, sizeof(zero));

	  TimeStep max;
	  DynamicCollective min_reduction =
		runtime->create_dynamic_collective(ctx, num_subregions_, MinReductionOp::redop_id,
						   &max, sizeof(max));

	  std::vector<SPMDArgs> args(num_subregions_);
	  std::vector<SPMDArgsSerializer> args_seriliazed(num_subregions_);

	  for (int color = 0; color < num_subregions_; color++) {
		  args[color].pbarrier_as_master = global_mesh.phase_barriers[color];
		  args[color].add_reduction = add_reduction;
		  args[color].min_reduction = min_reduction;
		  args[color].direct_input_params = input_params.directs_;
		  args[color].direct_input_params.task_id_ = color;
          args[color].meshtype = input_params.meshtype_;
          args[color].probname = input_params.probname_;
          args[color].bcx = input_params.bcx_;
          args[color].bcy = input_params.bcy_;

		  std::vector<LogicalRegion> lregions_halos;
          lregions_halos.push_back(global_mesh.halos_points[color].getLRegion());
          for (int i=0; i < global_mesh.masters[color].size(); i++) {
              lregions_halos.push_back(global_mesh.halos_points[(global_mesh.masters[color])[i]].getLRegion());
              args[color].masters_pbarriers.push_back(global_mesh.phase_barriers[(global_mesh.masters[color])[i]]);
          }

		  args_seriliazed[color].archive(&(args[color]));

		  DomainPoint point(color);
		  LogicalRegion my_zones = runtime->get_logical_subregion_by_color(ctx,
				  global_mesh.zones.getLPart(), color);
		  LogicalRegion my_pts = runtime->get_logical_subregion_by_color(ctx,
				  global_mesh.points.getLPart(), color);

		  DriverTask driver_launcher(color, my_zones, global_mesh.zones.getLRegion(),
				  my_pts, global_mesh.points.getLRegion(),
				  lregions_halos,
				  args_seriliazed[color].getBitStream(), args_seriliazed[color].getBitStreamSize());
		  must_epoch_launcher.add_single_task(point, driver_launcher);
	  }

	  FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
	  fm.wait_all_results();

	  //RunStat run_stat = fm.get_result<RunStat>(0);

      SPMDArgs arg;
      SPMDArgsSerializer serial;
      arg.probname = input_params.probname_;
      arg.direct_input_params = input_params.directs_;
      serial.archive(&arg);

	  WriteTask write_launcher(global_mesh.zones.getLRegion(),
              serial.getBitStream(), serial.getBitStreamSize());
      runtime->execute_task(ctx, write_launcher);
}


void Parallel::globalSum(int& x) {
    //if (num_subregions_ == 1)
	return;
#ifdef USE_MPI
    int y;
    MPI_Allreduce(&x, &y, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    x = y;
#endif
}


void Parallel::globalSum(int64_t& x) {
    //if (num_subregions_ == 1)
	return;
#ifdef USE_MPI
    int64_t y;
    MPI_Allreduce(&x, &y, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    x = y;
#endif
}


void Parallel::globalSum(double& x) {
    //if (num_subregions_ == 1)
	return;
#ifdef USE_MPI
    double y;
    MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    x = y;
#endif
}


void Parallel::gather(int x, int* y) {
    //if (num_subregions_ == 1)
	{
        y[0] = x;
        return;
    }
#ifdef USE_MPI
    MPI_Gather(&x, 1, MPI_INT, y, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
}


void Parallel::scatter(const int* x, int& y) {
    //if (num_subregions_ == 1)
	{
        y = x[0];
        return;
    }
#ifdef USE_MPI
    MPI_Scatter((void*) x, 1, MPI_INT, &y, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
}


template<typename T>
void Parallel::gathervImpl(
        const T *x, const int numx,
        T* y, const int* numy) {

    //if (num_subregions_ == 1)
	{
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
void Parallel::gatherv(
        const double2 *x, const int numx,
        double2* y, const int* numy) {
    gathervImpl(x, numx, y, numy);
}


template<>
void Parallel::gatherv(
        const double *x, const int numx,
        double* y, const int* numy) {
    gathervImpl(x, numx, y, numy);
}


template<>
void Parallel::gatherv(
        const int *x, const int numx,
        int* y, const int* numy) {
    gathervImpl(x, numx, y, numy);
}

// Legion Stuff

Future Parallel::globalSum(double local_value,
		DynamicCollective& dc_reduction,
		Runtime *runtime, Context ctx,
		Predicate pred)
{
  TaskLauncher launcher(sumTaskID, TaskArgument(&local_value, sizeof(local_value)), pred, 0 /*default mapper*/);
  double zero = 0.0;
  launcher.set_predicate_false_result(TaskArgument(&zero, sizeof(zero)));
  Future f = runtime->execute_task(ctx, launcher);
  runtime->defer_dynamic_collective_arrival(ctx, dc_reduction, f);
  f.get_result<double>();
  dc_reduction = runtime->advance_dynamic_collective(ctx, dc_reduction);
  Future ff2 = runtime->get_dynamic_collective_result(ctx, dc_reduction);
  return ff2;
}

double Parallel::globalSumTask (const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime)
{
	double value = *(const double *)(task->args);
	return value;
}

Future Parallel::globalMin(TimeStep local_value,
		DynamicCollective& dc_reduction,
		Runtime *runtime, Context ctx,
		Predicate pred)
{
  TaskLauncher launcher(minTaskID, TaskArgument(&local_value, sizeof(local_value)), pred, 0 /*default mapper*/);
  TimeStep max;
  launcher.set_predicate_false_result(TaskArgument(&max, sizeof(max)));
  Future f = runtime->execute_task(ctx, launcher);
  runtime->defer_dynamic_collective_arrival(ctx, dc_reduction, f);
  f.get_result<TimeStep>();
  dc_reduction = runtime->advance_dynamic_collective(ctx, dc_reduction);
  Future ff2 = runtime->get_dynamic_collective_result(ctx, dc_reduction);
  return ff2;
}


TimeStep Parallel::globalMinTask (const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime)
{
	TimeStep value = *(const TimeStep *)(task->args);
	return value;
}


template<class Type>
static size_t archiveScalar(Type scalar, void* bit_stream)
{
    memcpy(bit_stream, (void*)(&scalar), sizeof(Type));
    return sizeof(Type);
}


static size_t archiveString(std::string name, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t size_size = archiveScalar(name.length()+1, (void*)serialized);
    serialized += size_size;

    size_t string_size = name.length() * sizeof(char);
    memcpy((void*)serialized, (void*)name.c_str(), string_size);
    serialized += string_size;

    size_t terminator_size = archiveScalar('\0', (void*)serialized);

    return size_size + string_size + terminator_size;
}

template<class Type>
static size_t archiveVector(std::vector<Type> vec, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t size_size = archiveScalar(vec.size(), (void*)serialized);
    serialized += size_size;

    size_t vec_size = vec.size() * sizeof(Type);
    memcpy((void*)serialized, (void*)vec.data(), vec_size);

    return size_size + vec_size;
}

void SPMDArgsSerializer::archive(SPMDArgs* args)
{
    assert(args != nullptr);
    spmd_args = args;

    bit_stream_size = 2 * sizeof(DynamicCollective) + sizeof(DirectInputParameters)
            + sizeof(PhaseBarrier) + 5 * sizeof(size_t)
            + (args->meshtype.length()  + 1 + args->probname.length() + 1) * sizeof(char)
            + (args->bcx.size() + args->bcy.size()) * sizeof(double)
            + args->masters_pbarriers.size() * sizeof(PhaseBarrier);
    bit_stream = malloc(bit_stream_size);
    free_bit_stream = true;

    unsigned char *serialized = (unsigned char*)(bit_stream);

    size_t stream_size = 0;
    stream_size += archiveScalar(args->add_reduction, (void*)(serialized+stream_size));
    stream_size += archiveScalar(args->min_reduction, (void*)(serialized+stream_size));
    stream_size += archiveScalar(args->direct_input_params, (void*)(serialized+stream_size));
    stream_size += archiveScalar(args->pbarrier_as_master, (void*)(serialized+stream_size));
    stream_size += archiveString(args->meshtype, (void*)(serialized+stream_size));
    stream_size += archiveString(args->probname, (void*)(serialized+stream_size));
    stream_size += archiveVector(args->bcx, (void*)(serialized+stream_size));
    stream_size += archiveVector(args->bcy, (void*)(serialized+stream_size));
    stream_size += archiveVector(args->masters_pbarriers, (void*)(serialized+stream_size));

    assert(stream_size == bit_stream_size);
}


template<class Type>
static size_t restoreScalar(Type* scalar, void* bit_stream)
{
    memcpy((void*)scalar, bit_stream, sizeof(Type));
    return sizeof(Type);
}


static size_t restoreString(std::string* name, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t n_chars;
    size_t size_size = restoreScalar(&n_chars, (void*)serialized);
    serialized += size_size;

    size_t string_size = n_chars * sizeof(char);
    char *buffer = (char *)malloc(string_size);
    memcpy((void *)buffer, (void *)serialized, string_size);
    *name = std::string(buffer);

    return size_size + string_size;
}


template<class Type>
static size_t restoreVector(std::vector<Type>* vec, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t n_entries;
    size_t size_size = restoreScalar(&n_entries, (void*)serialized);
    serialized += size_size;

    vec->resize(n_entries);
    size_t vec_size = n_entries * sizeof(Type);
    memcpy((void*)vec->data(), (void*)serialized, vec_size);

    return size_size + vec_size;
}


void SPMDArgsSerializer::restore(SPMDArgs* args)
{
    assert(args != nullptr);
    assert(bit_stream != nullptr);
    spmd_args = args;

    unsigned char *serialized_args = (unsigned char *) bit_stream;

    bit_stream_size = 0;
    bit_stream_size += restoreScalar(&(args->add_reduction), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(args->min_reduction), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(args->direct_input_params), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(args->pbarrier_as_master), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreString(&(args->meshtype), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreString(&(args->probname), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(args->bcx), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(args->bcy), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(args->masters_pbarriers), (void*)(serialized_args + bit_stream_size));
}


void* SPMDArgsSerializer::getBitStream()
{
    return bit_stream;
}


size_t SPMDArgsSerializer::getBitStreamSize()
{
    return bit_stream_size;
}


void SPMDArgsSerializer::setBitStream(void* stream)
{
    bit_stream = stream;
};
