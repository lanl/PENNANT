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
    FOUR,
    SIX,
    NINE,
    TEN,
    ELEVEN,
};

PredictorPointTask::PredictorPointTask(
        LogicalRegion mesh_points,
        LogicalRegion hydro_points,
        void *args, const size_t &size)
	 : TaskLauncher(PredictorPointTask::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(hydro_points, READ_ONLY, EXCLUSIVE, hydro_points));
    add_field(FOUR, FID_PU);
    add_region_requirement(RegionRequirement(hydro_points, READ_WRITE, EXCLUSIVE, hydro_points));
    add_field(SIX, FID_PU0);
    add_region_requirement(RegionRequirement(mesh_points, READ_WRITE, EXCLUSIVE, mesh_points));
    add_field(NINE, FID_PXP);
    add_field(NINE, FID_PX0);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(TEN, FID_PX);
}

/*static*/ const char * const PredictorPointTask::TASK_NAME = "PredictorPointTask";


/*static*/
void PredictorPointTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == ELEVEN);
	assert(task->regions.size() == ELEVEN);

    DoCycleTasksArgs args;
    DoCycleTasksArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    assert(task->regions[FOUR].privilege_fields.size() == 1);
    LogicalStructured hydro_points(ctx, runtime, regions[FOUR]);
    const double2* pt_vel = hydro_points.getRawPtr<double2>(FID_PU);

    assert(task->regions[TEN].privilege_fields.size() == 1);
    LogicalStructured mesh_points(ctx, runtime, regions[TEN]);
    const double2* pt_x = mesh_points.getRawPtr<double2>(FID_PX);

    assert(task->regions[SIX].privilege_fields.size() == 1);
    LogicalStructured hydro_write_points(ctx, runtime, regions[SIX]);
    double2* pt_vel0 = hydro_write_points.getRawPtr<double2>(FID_PU0);

    assert(task->regions[NINE].privilege_fields.size() == 2);
    LogicalStructured mesh_write_points(ctx, runtime, regions[NINE]);
    double2* pt_x_pred = mesh_write_points.getRawPtr<double2>(FID_PXP);
    double2* pt_x0 = mesh_write_points.getRawPtr<double2>(FID_PX0);

    assert(task->futures.size() == 1);
    Future f1 = task->futures[0];
    TimeStep time_step = f1.get_result<TimeStep>();

    for (int pch = 0; pch < (args.point_chunk_CRS.size()-1); ++pch) {
        int pfirst = args.point_chunk_CRS[pch];
        int plast = args.point_chunk_CRS[pch+1];

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

