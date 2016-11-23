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
#include "AddInt64ReductionOp.hh"
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
    const int num_subregions_ = input_params.directs.ntasks;

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

      int64_t int_zero = 0;
      DynamicCollective add_int64_reduction =
        runtime->create_dynamic_collective(ctx, num_subregions_, AddInt64ReductionOp::redop_id,
                           &int_zero, sizeof(int_zero));

	  TimeStep max;
	  DynamicCollective min_reduction =
		runtime->create_dynamic_collective(ctx, num_subregions_, MinReductionOp::redop_id,
						   &max, sizeof(max));

	  std::vector<SPMDArgs> args(num_subregions_);
	  std::vector<SPMDArgsSerializer> args_seriliazed(num_subregions_);

	  for (int color = 0; color < num_subregions_; color++) {
		  args[color].pbarrier_as_master = global_mesh.phase_barriers[color];
          args[color].add_reduction = add_reduction;
          args[color].add_int64_reduction = add_int64_reduction;
		  args[color].min_reduction = min_reduction;
		  args[color].direct_input_params = input_params.directs;
		  args[color].direct_input_params.task_id = color;
          args[color].meshtype = input_params.meshtype;
          args[color].probname = input_params.probname;
          args[color].bcx = input_params.bcx;
          args[color].bcy = input_params.bcy;

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

		  DriverTask driver_launcher(color, my_zones, global_mesh.zones.getLRegion(),
				  lregions_halos,
				  args_seriliazed[color].getBitStream(), args_seriliazed[color].getBitStreamSize());
		  must_epoch_launcher.add_single_task(point, driver_launcher);
	  }

	  FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
	  fm.wait_all_results();

	  //RunStat run_stat = fm.get_result<RunStat>(0);

      SPMDArgs arg;
      SPMDArgsSerializer serial;
      arg.probname = input_params.probname;
      arg.direct_input_params = input_params.directs;
      serial.archive(&arg);

	  WriteTask write_launcher(global_mesh.zones.getLRegion(),
              serial.getBitStream(), serial.getBitStreamSize());
      runtime->execute_task(ctx, write_launcher);
}


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
  dc_reduction = runtime->advance_dynamic_collective(ctx, dc_reduction);
  Future ff2 = runtime->get_dynamic_collective_result(ctx, dc_reduction);
  return ff2;
}


Future Parallel::globalSumInt64(int64_t local_value,
        DynamicCollective& dc_reduction,
        Runtime *runtime, Context ctx,
        Predicate pred)
{
  TaskLauncher launcher(sumInt64TaskID, TaskArgument(&local_value, sizeof(local_value)), pred, 0 /*default mapper*/);
  int64_t zero = 0;
  launcher.set_predicate_false_result(TaskArgument(&zero, sizeof(zero)));
  Future f = runtime->execute_task(ctx, launcher);
  runtime->defer_dynamic_collective_arrival(ctx, dc_reduction, f);
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


int64_t Parallel::globalSumInt64Task (const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime)
{
    int64_t value = *(const int64_t *)(task->args);
    return value;
}


Future Parallel::globalMin(Future local_value,
		DynamicCollective& dc_reduction,
		Runtime *runtime, Context ctx,
		Predicate pred)
{
  runtime->defer_dynamic_collective_arrival(ctx, dc_reduction, local_value);
  dc_reduction = runtime->advance_dynamic_collective(ctx, dc_reduction);
  Future ff2 = runtime->get_dynamic_collective_result(ctx, dc_reduction);
  return ff2;
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

template<class Type>
static size_t archiveTensor(std::vector<std::vector<Type>> tensor, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t size_size = archiveScalar(tensor.size(), (void*)serialized);
    serialized += size_size;

    size_t tensor_size = 0;
    for (int i = 0; i < tensor.size(); i++) {
        size_t vec_size = archiveVector(tensor[i], (void*)serialized);
        serialized += vec_size;
        tensor_size += vec_size;
    }

    return size_size + tensor_size;
}


void SPMDArgsSerializer::archive(SPMDArgs* spmd_args)
{
    assert(spmd_args != nullptr);

    bit_stream_size = 3 * sizeof(DynamicCollective) + sizeof(DirectInputParameters)
            + sizeof(PhaseBarrier) + 5 * sizeof(size_t)
            + (spmd_args->meshtype.length() + 1 + spmd_args->probname.length() + 1) * sizeof(char)
            + (spmd_args->bcx.size() + spmd_args->bcy.size()) * sizeof(double)
            + spmd_args->masters_pbarriers.size() * sizeof(PhaseBarrier);
    bit_stream = malloc(bit_stream_size);
    free_bit_stream = true;

    unsigned char *serialized = (unsigned char*)(bit_stream);

    size_t stream_size = 0;
    stream_size += archiveScalar(spmd_args->add_reduction, (void*)(serialized+stream_size));
    stream_size += archiveScalar(spmd_args->add_int64_reduction, (void*)(serialized+stream_size));
    stream_size += archiveScalar(spmd_args->min_reduction, (void*)(serialized+stream_size));
    stream_size += archiveScalar(spmd_args->direct_input_params, (void*)(serialized+stream_size));
    stream_size += archiveScalar(spmd_args->pbarrier_as_master, (void*)(serialized+stream_size));
    stream_size += archiveString(spmd_args->meshtype, (void*)(serialized+stream_size));
    stream_size += archiveString(spmd_args->probname, (void*)(serialized+stream_size));
    stream_size += archiveVector(spmd_args->bcx, (void*)(serialized+stream_size));
    stream_size += archiveVector(spmd_args->bcy, (void*)(serialized+stream_size));
    stream_size += archiveVector(spmd_args->masters_pbarriers, (void*)(serialized+stream_size));

    assert(stream_size == bit_stream_size);
}


void DoCycleTasksArgsSerializer::archive(DoCycleTasksArgs* docycle_args)
{
    assert(docycle_args != nullptr);

    bit_stream_size = sizeof(DynamicCollective) + 8 * sizeof(double) + 11 * sizeof(int)
        + sizeof(size_t) + (docycle_args->meshtype.length() + 1) * sizeof(char)
        + sizeof(size_t) + docycle_args->boundary_conditions_x.size() * sizeof(size_t)
        + sizeof(size_t) + docycle_args->boundary_conditions_y.size() * sizeof(size_t)
        + sizeof(size_t) + docycle_args->bcx_point_chunk_CRS_offsets.size() * sizeof(int)
        + sizeof(size_t) + docycle_args->bcy_point_chunk_CRS_offsets.size() * sizeof(int);
    for (size_t i = 0; i < docycle_args->boundary_conditions_x.size(); i++)
        bit_stream_size += docycle_args->boundary_conditions_x[i].size() * sizeof(int);
    for (size_t i = 0; i < docycle_args->boundary_conditions_y.size(); i++)
        bit_stream_size += docycle_args->boundary_conditions_y[i].size() * sizeof(int);
    bit_stream = malloc(bit_stream_size);
    free_bit_stream = true;

    unsigned char *serialized = (unsigned char*)(bit_stream);

    size_t stream_size = 0;
    stream_size += archiveScalar(docycle_args->min_reduction, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->cfl, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->cflv, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_points, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_sides, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_zones, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_edges, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->nzones_x, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->nzones_y, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_subregions, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->my_color, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->qgamma, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->q1, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->q2, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->ssmin, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->alpha, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->gamma, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_zone_chunks, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_side_chunks, (void*)(serialized+stream_size));
    stream_size += archiveScalar(docycle_args->num_point_chunks, (void*)(serialized+stream_size));
    stream_size += archiveString(docycle_args->meshtype, (void*)(serialized+stream_size));
    stream_size += archiveTensor(docycle_args->boundary_conditions_x, (void*)(serialized+stream_size));
    stream_size += archiveTensor(docycle_args->boundary_conditions_y, (void*)(serialized+stream_size));
    stream_size += archiveVector(docycle_args->bcx_point_chunk_CRS_offsets, (void*)(serialized+stream_size));
    stream_size += archiveVector(docycle_args->bcy_point_chunk_CRS_offsets, (void*)(serialized+stream_size));

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


template<class Type>
static size_t restoreTensor(std::vector<std::vector<Type>>* tensor, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t n_entries;
    size_t size_size = restoreScalar(&n_entries, (void*)serialized);
    serialized += size_size;

    size_t tensor_size = 0;
    for (int i = 0; i < n_entries; i++) {
        std::vector<Type> vec;
        size_t vec_size = restoreVector(&vec, (void*)serialized);
        tensor->push_back(vec);
        serialized += vec_size;
        tensor_size += vec_size;
    }

    return size_size + tensor_size;
}


void SPMDArgsSerializer::restore(SPMDArgs* spmd_args)
{
    assert(spmd_args != nullptr);
    assert(bit_stream != nullptr);

    unsigned char *serialized_args = (unsigned char *) bit_stream;

    bit_stream_size = 0;
    bit_stream_size += restoreScalar(&(spmd_args->add_reduction), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(spmd_args->add_int64_reduction), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(spmd_args->min_reduction), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(spmd_args->direct_input_params), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(spmd_args->pbarrier_as_master), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreString(&(spmd_args->meshtype), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreString(&(spmd_args->probname), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(spmd_args->bcx), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(spmd_args->bcy), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(spmd_args->masters_pbarriers), (void*)(serialized_args + bit_stream_size));
}


void DoCycleTasksArgsSerializer::restore(DoCycleTasksArgs* docycle_args)
{
    assert(docycle_args != nullptr);
    assert(bit_stream != nullptr);

    unsigned char *serialized_args = (unsigned char *) bit_stream;

    bit_stream_size = 0;
    bit_stream_size += restoreScalar(&(docycle_args->min_reduction), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->cfl), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->cflv), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_points), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_sides), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_zones), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_edges), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->nzones_x), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->nzones_y), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_subregions), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->my_color), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->qgamma), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->q1), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->q2), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->ssmin), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->alpha), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->gamma), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_zone_chunks), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_side_chunks), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreScalar(&(docycle_args->num_point_chunks), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreString(&(docycle_args->meshtype), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreTensor(&(docycle_args->boundary_conditions_x), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreTensor(&(docycle_args->boundary_conditions_y), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(docycle_args->bcx_point_chunk_CRS_offsets), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(docycle_args->bcy_point_chunk_CRS_offsets), (void*)(serialized_args + bit_stream_size));
}


void* ArgsSerializer::getBitStream()
{
    return bit_stream;
}


size_t ArgsSerializer::getBitStreamSize()
{
    return bit_stream_size;
}


void ArgsSerializer::setBitStream(void* stream)
{
    bit_stream = stream;
};
