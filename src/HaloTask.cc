/*
 * HaloTask.cc
 *
 *  Created on: Oct 3, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#include "HaloTask.hh"

#include "GenerateMesh.hh"
#include "InputParameters.hh"
#include "LocalMesh.hh"
#include "LogicalStructured.hh"
#include "Vec2.hh"


HaloTask::HaloTask(LogicalRegion mesh_sides,
        LogicalRegion mesh_points,
        LogicalRegion mesh_local_points,
        LogicalRegion point_chunks,
        LogicalRegion hydro_sides_and_corners,
        void *args, const size_t &size)
	 : TaskLauncher(HaloTask::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(mesh_sides, READ_ONLY, EXCLUSIVE, mesh_sides));
    add_field(0/*idx*/, FID_MAP_CRN2CRN_NEXT);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(1/*idx*/, FID_MAP_PT2CRN_FIRST);
    add_field(1/*idx*/, FID_PT_LOCAL2GLOBAL);
    add_region_requirement(RegionRequirement(hydro_sides_and_corners, READ_ONLY, EXCLUSIVE, hydro_sides_and_corners));
    add_field(2/*idx*/, FID_CMASWT);
    add_field(2/*idx*/, FID_CFTOT);
    add_region_requirement(RegionRequirement(mesh_local_points, READ_WRITE, EXCLUSIVE, mesh_local_points));
    add_field(3/*idx*/, FID_PF);
    add_field(3/*idx*/, FID_PMASWT);
    add_region_requirement(RegionRequirement(point_chunks, READ_ONLY, EXCLUSIVE, point_chunks));
    add_field(4, FID_POINT_CHUNKS_CRS);
}

/*static*/ const char * const HaloTask::TASK_NAME = "HaloTask";


/*static*/
void HaloTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == 5);
	assert(task->regions.size() == 5);

    assert(task->regions[0].privilege_fields.size() == 1);
    LogicalStructured sides(ctx, runtime, regions[0]);
    const int* map_crn2crn_next = sides.getRawPtr<int>(FID_MAP_CRN2CRN_NEXT);

	assert(task->regions[1].privilege_fields.size() == 2);
    LogicalStructured points(ctx, runtime, regions[1]);
    const int* map_pt2crn_first = points.getRawPtr<int>(FID_MAP_PT2CRN_FIRST);
    const ptr_t* pt_local2globalID = points.getRawPtr<ptr_t>(FID_PT_LOCAL2GLOBAL);

	assert(task->regions[2].privilege_fields.size() == 2);
	LogicalStructured sides_and_corners(ctx, runtime, regions[2]);
	const double* corner_mass = sides_and_corners.getRawPtr<double>(FID_CMASWT);
    const double2* corner_force = sides_and_corners.getRawPtr<double2>(FID_CFTOT);

    assert(task->regions[4].privilege_fields.size() == 1);
    LogicalStructured point_chunks(ctx, runtime, regions[4]);
    const int* point_chunks_CRS = point_chunks.getRawPtr<int>(FID_POINT_CHUNKS_CRS);

    assert(task->regions[3].privilege_fields.size() == 2);
    LogicalUnstructured local_points_by_gid(ctx, runtime, regions[3]);
    DoubleSOAAccessor pt_weighted_mass = local_points_by_gid.getRegionSOAAccessor<double>(FID_PMASWT);
    Double2SOAAccessor pt_force = local_points_by_gid.getRegionSOAAccessor<double2>(FID_PF);

    DoCycleTasksArgs args;
    DoCycleTasksArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    LocalMesh::sumOnProc(corner_mass, corner_force,
            point_chunks_CRS,
            args.num_point_chunks,
            map_pt2crn_first,
            map_crn2crn_next,
            pt_local2globalID,
            pt_weighted_mass, pt_force);

}

