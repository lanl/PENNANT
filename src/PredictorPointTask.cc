/*
 * PredictorPointTask.cc
 *
 *  Created on: Oct 17, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#include "PredictorPointTask.hh"

#include <algorithm>

#include "GenerateMesh.hh"
#include "Hydro.hh"
#include "InputParameters.hh"
#include "LogicalStructured.hh"
#include "Memory.hh"
#include "Vec2.hh"

enum idx {
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
};

PredictorPointTask::PredictorPointTask(
        LogicalRegion mesh_points,
        LogicalRegion point_chunks,
        LogicalRegion hydro_points,
        void *args, const size_t &size)
	 : TaskLauncher(PredictorPointTask::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(hydro_points, READ_ONLY, EXCLUSIVE, hydro_points));
    add_field(ONE, FID_PU);
    add_region_requirement(RegionRequirement(hydro_points, READ_WRITE, EXCLUSIVE, hydro_points));
    add_field(TWO, FID_PU0);
    add_region_requirement(RegionRequirement(mesh_points, READ_WRITE, EXCLUSIVE, mesh_points));
    add_field(THREE, FID_PXP);
    add_field(THREE, FID_PX0);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(FOUR, FID_PX);
    add_region_requirement(RegionRequirement(point_chunks, READ_ONLY, EXCLUSIVE, point_chunks));
    add_field(FIVE, FID_POINT_CHUNKS_CRS);
}

/*static*/ const char * const PredictorPointTask::TASK_NAME = "PredictorPointTask";


/*static*/
void PredictorPointTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == SIX);
	assert(task->regions.size() == SIX);

    DoCycleTasksArgs args;
    DoCycleTasksArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    assert(task->regions[ONE].privilege_fields.size() == 1);
    LogicalStructured hydro_points(ctx, runtime, regions[ONE]);
    const double2* pt_vel = hydro_points.getRawPtr<double2>(FID_PU);

    assert(task->regions[FOUR].privilege_fields.size() == 1);
    LogicalStructured mesh_points(ctx, runtime, regions[FOUR]);
    const double2* pt_x = mesh_points.getRawPtr<double2>(FID_PX);

    assert(task->regions[FIVE].privilege_fields.size() == 1);
    LogicalStructured point_chunks(ctx, runtime, regions[FIVE]);
    const int* point_chunks_CRS = point_chunks.getRawPtr<int>(FID_POINT_CHUNKS_CRS);

    assert(task->regions[TWO].privilege_fields.size() == 1);
    LogicalStructured hydro_write_points(ctx, runtime, regions[TWO]);
    double2* pt_vel0 = hydro_write_points.getRawPtr<double2>(FID_PU0);

    assert(task->regions[THREE].privilege_fields.size() == 2);
    LogicalStructured mesh_write_points(ctx, runtime, regions[THREE]);
    double2* pt_x_pred = mesh_write_points.getRawPtr<double2>(FID_PXP);
    double2* pt_x0 = mesh_write_points.getRawPtr<double2>(FID_PX0);

    assert(task->futures.size() == 1);
    Future f1 = task->futures[0];
    TimeStep time_step = f1.get_result<TimeStep>();

    for (int pch = 0; pch < args.num_point_chunks; ++pch) {
        int pfirst = point_chunks_CRS[pch];
        int plast = point_chunks_CRS[pch+1];

        // save off point variable values from previous cycle
        std::copy(&pt_x[pfirst], &pt_x[plast], &pt_x0[pfirst]);
        std::copy(&pt_vel[pfirst], &pt_vel[plast], &pt_vel0[pfirst]);

        // ===== Predictor step =====
        // 1. advance mesh to center of time step
        Hydro::advPosHalf(time_step.dt, pfirst, plast,
                pt_x0, pt_vel0,
                pt_x_pred);
    }
}

