/*
 * CorrectorTask.cc
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

#include "CorrectorTask.hh"

#include<iostream>

#include "Hydro.hh"
#include "HydroBC.hh"
#include "InputParameters.hh"
#include "LocalMesh.hh"
#include "LogicalStructured.hh"
#include "Memory.hh"
#include "Vec2.hh"


CorrectorTask::CorrectorTask(LogicalRegion mesh_zones,
        LogicalRegion mesh_sides,
        LogicalRegion mesh_zone_pts,
        LogicalRegion mesh_points,
        LogicalRegion mesh_edges,
        LogicalRegion mesh_local_points,
        LogicalRegion hydro_zones,
        LogicalRegion hydro_sides_and_corners,
        LogicalRegion hydro_points,
		void *args, const size_t &size)
	 : TaskLauncher(CorrectorTask::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(mesh_zones, READ_ONLY, EXCLUSIVE, mesh_zones));
    add_field(0/*idx*/, FID_ZDL);
    add_field(0/*idx*/, FID_ZVOL0);
    add_region_requirement(RegionRequirement(hydro_zones, READ_ONLY, EXCLUSIVE, hydro_zones));
    add_field(1/*idx*/, FID_ZDU);
    add_field(1/*idx*/, FID_ZSS);
    add_field(1/*idx*/, FID_ZM);
    add_field(1/*idx*/, FID_ZP);
    add_region_requirement(RegionRequirement(hydro_zones, READ_WRITE, EXCLUSIVE, hydro_zones));
    add_field(2/*idx*/, FID_ZR);
    add_field(2/*idx*/, FID_ZE);
    add_field(2/*idx*/, FID_ZWR);
    add_field(2/*idx*/, FID_ZETOT);
    add_region_requirement(RegionRequirement(mesh_sides, READ_ONLY, EXCLUSIVE, mesh_sides));
    add_field(3/*idx*/, FID_SMAP_SIDE_TO_ZONE);
    add_field(3/*idx*/, FID_SMAP_SIDE_TO_PT1);
    add_field(3/*idx*/, FID_SMAP_SIDE_TO_PT2);
    add_field(3/*idx*/, FID_SMAP_SIDE_TO_EDGE);
    add_region_requirement(RegionRequirement(mesh_zone_pts, READ_ONLY, EXCLUSIVE, mesh_zone_pts));
    add_field(4/*idx*/, FID_ZONE_PTS_PTR);
    add_region_requirement(RegionRequirement(hydro_sides_and_corners, READ_ONLY, EXCLUSIVE, hydro_sides_and_corners));
    add_field(5/*idx*/, FID_SFP);
    add_field(5/*idx*/, FID_SFQ);
    add_region_requirement(RegionRequirement(mesh_local_points, READ_WRITE, EXCLUSIVE, mesh_local_points));
    add_field(6/*idx*/, FID_PF);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(7/*idx*/, FID_PXP);
    add_field(7/*idx*/, FID_PX0);
    add_region_requirement(RegionRequirement(mesh_edges, READ_WRITE, EXCLUSIVE, mesh_edges));
    add_field(8/*idx*/, FID_EX);
    add_region_requirement(RegionRequirement(mesh_zones, READ_WRITE, EXCLUSIVE, mesh_zones));
    add_field(9/*idx*/, FID_ZX);
    add_field(9/*idx*/, FID_ZVOL);
    add_field(9/*idx*/, FID_ZAREA);
    add_region_requirement(RegionRequirement(mesh_sides, READ_WRITE, EXCLUSIVE, mesh_sides));
    add_field(10/*idx*/, FID_SAREA);
    add_field(10/*idx*/, FID_SVOL);
    add_region_requirement(RegionRequirement(hydro_points, READ_WRITE, EXCLUSIVE, hydro_points));
    add_field(11/*idx*/, FID_PU);
    add_field(11/*idx*/, FID_PU0);
    add_region_requirement(RegionRequirement(mesh_points, READ_WRITE, EXCLUSIVE, mesh_points));
    add_field(12/*idx*/, FID_PX);
    add_region_requirement(RegionRequirement(mesh_local_points, READ_ONLY, EXCLUSIVE, mesh_local_points));
    add_field(13/*idx*/, FID_PMASWT);
}

/*static*/ const char * const CorrectorTask::TASK_NAME = "CorrectorTask";


/*static*/
TimeStep CorrectorTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == 14);
	assert(task->regions.size() == 14);

	assert(task->regions[0].privilege_fields.size() == 2);
    LogicalStructured mesh_zones(ctx, runtime, regions[0]);
    const double* zone_dl = mesh_zones.getRawPtr<double>(FID_ZDL);
    const double* zone_vol0 = mesh_zones.getRawPtr<double>(FID_ZVOL0);

    assert(task->regions[1].privilege_fields.size() == 4);
    LogicalStructured hydro_read_zones(ctx, runtime, regions[1]);
    const double* zone_dvel = hydro_read_zones.getRawPtr<double>(FID_ZDU);
    const double* zone_sound_speed = hydro_read_zones.getRawPtr<double>(FID_ZSS);
    const double* zone_mass = hydro_read_zones.getRawPtr<double>(FID_ZM);
    const double* zone_pressure = hydro_read_zones.getRawPtr<double>(FID_ZP);

    assert(task->regions[3].privilege_fields.size() == 4);
    LogicalStructured mesh_sides(ctx, runtime, regions[3]);
    const int* map_side2zone = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_ZONE);
    const int* map_side2pt1 = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT1);
    const int* map_side2pt2 = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT2);
    const int* map_side2edge = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_EDGE);

    assert(task->regions[4].privilege_fields.size() == 1);
    LogicalStructured mesh_zone_pts(ctx, runtime, regions[4]);
    const int* zone_pts_ptr = mesh_zone_pts.getRawPtr<int>(FID_ZONE_PTS_PTR);

    assert(task->regions[5].privilege_fields.size() == 2);
    LogicalStructured hydro_sides_and_corners(ctx, runtime, regions[5]);
    const double2* side_force_pres = hydro_sides_and_corners.getRawPtr<double2>(FID_SFP);
    const double2* side_force_visc = hydro_sides_and_corners.getRawPtr<double2>(FID_SFQ);

    assert(task->regions[7].privilege_fields.size() == 2);
    LogicalStructured mesh_points(ctx, runtime, regions[7]);
    const double2* pt_x_pred = mesh_points.getRawPtr<double2>(FID_PXP);
    const double2* pt_x0 = mesh_points.getRawPtr<double2>(FID_PX0);

    assert(task->regions[13].privilege_fields.size() == 1);
    LogicalUnstructured local_points_by_gid(ctx, runtime, regions[13]);
    const DoubleAccessor point_mass = local_points_by_gid.getRegionAccessor<double>(FID_PMASWT);

    assert(task->regions[6].privilege_fields.size() == 1);
    LogicalUnstructured local_write_points_by_gid(ctx, runtime, regions[6]);
    Double2Accessor point_force = local_write_points_by_gid.getRegionAccessor<double2>(FID_PF);

    assert(task->regions[2].privilege_fields.size() == 4);
    LogicalStructured hydro_write_zones(ctx, runtime, regions[2]);
    double* zone_rho = hydro_write_zones.getRawPtr<double>(FID_ZR);
    double* zone_energy_density = hydro_write_zones.getRawPtr<double>(FID_ZE);
    double* zone_work_rate = hydro_write_zones.getRawPtr<double>(FID_ZWR);
    double* zone_energy_tot = hydro_write_zones.getRawPtr<double>(FID_ZETOT);
    double* zone_work = AbstractedMemory::alloc<double>(hydro_write_zones.size());

    assert(task->regions[8].privilege_fields.size() == 1);
    LogicalStructured mesh_edges(ctx, runtime, regions[8]);
    double2* edge_x = mesh_edges.getRawPtr<double2>(FID_EX);

    assert(task->regions[9].privilege_fields.size() == 3);
    LogicalStructured mesh_write_zones(ctx, runtime, regions[9]);
    double2* zone_x = mesh_write_zones.getRawPtr<double2>(FID_ZX);
    double* zone_vol = mesh_write_zones.getRawPtr<double>(FID_ZVOL);
    double* zone_area = mesh_write_zones.getRawPtr<double>(FID_ZAREA);

    assert(task->regions[10].privilege_fields.size() == 2);
    LogicalStructured mesh_write_sides(ctx, runtime, regions[10]);
    double* side_vol = mesh_write_sides.getRawPtr<double>(FID_SVOL);
    double* side_area = mesh_write_sides.getRawPtr<double>(FID_SAREA);

    assert(task->regions[11].privilege_fields.size() == 2);
    LogicalStructured hydro_write_points(ctx, runtime, regions[11]);
    double2* pt_vel = hydro_write_points.getRawPtr<double2>(FID_PU);
    double2* pt_vel0 = hydro_write_points.getRawPtr<double2>(FID_PU0);

    assert(task->regions[12].privilege_fields.size() == 1);
    LogicalStructured mesh_write_points(ctx, runtime, regions[12]);
    double2* pt_x = mesh_write_points.getRawPtr<double2>(FID_PX);

    DoCycleTasksArgs args;
    DoCycleTasksArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    InputParameters input_params;
    input_params.meshtype = args.meshtype;
    input_params.directs.nzones_x = args.nzones_x;
    input_params.directs.nzones_y = args.nzones_y;
    input_params.directs.ntasks = args.num_subregions;
    input_params.directs.task_id = args.my_color;
    GenerateMesh* generate_mesh = new GenerateMesh(input_params);

    assert(task->futures.size() == 1);
    Future f1 = task->futures[0];
    TimeStep time_step = f1.get_result<TimeStep>();

    for (int pt_chunk = 0; pt_chunk < (args.point_chunk_CRS.size()-1); ++pt_chunk) {
        int pfirst = args.point_chunk_CRS[pt_chunk];
        int plast = args.point_chunk_CRS[pt_chunk+1];

        // 4a. apply boundary conditions
        const double2 vfixx = double2(1., 0.);
        const double2 vfixy = double2(0., 1.);
        for (int x = 0; x < args.boundary_conditions_x.size(); ++x) {
            int bfirst = args.bcx_point_chunk_CRS[x][pt_chunk];
            int blast = args.bcx_point_chunk_CRS[x][pt_chunk+1];
            HydroBC::applyFixedBC(generate_mesh, vfixx, args.boundary_conditions_x[x],
                    pt_vel0, point_force, bfirst, blast);
        }
        for (int y = 0; y < args.boundary_conditions_y.size(); ++y) {
            int bfirst = args.bcy_point_chunk_CRS[y][pt_chunk];
            int blast = args.bcy_point_chunk_CRS[y][pt_chunk+1];
            HydroBC::applyFixedBC(generate_mesh, vfixy, args.boundary_conditions_y[y],
                    pt_vel0, point_force, bfirst, blast);
        }

        // 5. compute accelerations
        // ===== Corrector step =====
        // 6. advance mesh to end of time step
        Hydro::calcAccelAndAdvPosFull(generate_mesh, point_force, point_mass,
                time_step.dt, pt_vel0, pt_x0, pt_vel, pt_x,
                pfirst, plast);
    }

    for (int side_chunk = 0; side_chunk < (args.side_chunk_CRS.size()-1); ++side_chunk) {
        int sfirst = args.side_chunk_CRS[side_chunk];
        int slast = args.side_chunk_CRS[side_chunk+1];
        int zfirst = LocalMesh::side_zone_chunks_first(side_chunk, map_side2zone, args.side_chunk_CRS);
        int zlast = LocalMesh::side_zone_chunks_last(side_chunk, map_side2zone, args.side_chunk_CRS);

        // 6a. compute new mesh geometry
        LocalMesh::calcCtrs(sfirst, slast, pt_x,
                map_side2zone, args.num_sides, args.num_zones, map_side2pt1, map_side2pt2, map_side2edge, zone_pts_ptr,
                edge_x, zone_x);
        LocalMesh::calcVols(sfirst, slast, pt_x, zone_x,
                map_side2zone, args.num_sides, args.num_zones, map_side2pt1, map_side2pt2, zone_pts_ptr,
                side_area, side_vol, zone_area, zone_vol);

        // 7. compute work
        std::fill(&zone_work[zfirst], &zone_work[zlast], 0.);
        Hydro::calcWork(time_step.dt, map_side2pt1, map_side2pt2, map_side2zone, zone_pts_ptr, side_force_pres,
                side_force_visc, pt_vel, pt_vel0, pt_x_pred, zone_energy_tot, zone_work,
                sfirst, slast);
    } // side chunk

    TimeStep recommend;
    recommend.dt = 1.e99;
    strcpy(recommend.message, "Hydro default");

    for (int zone_chunk = 0; zone_chunk < (args.zone_chunk_CRS.size()-1); ++zone_chunk) {
        int zone_first = args.zone_chunk_CRS[zone_chunk];
        int zone_last = args.zone_chunk_CRS[zone_chunk+1];

        // 7a. compute work rate
        Hydro::calcWorkRate(time_step.dt, zone_vol, zone_vol0, zone_work, zone_pressure,
                zone_work_rate, zone_first, zone_last);

        // 8. update state variables
        Hydro::calcEnergy(zone_energy_tot, zone_mass, zone_energy_density, zone_first, zone_last);
        Hydro::calcRho(zone_vol, zone_mass, zone_rho, zone_first, zone_last);

        // 9.  compute timestep for next cycle
        Hydro::calcDtHydro(time_step.dt, zone_first, zone_last, zone_dl,
                zone_dvel, zone_sound_speed, args.cfl,
                zone_vol, zone_vol0, args.cflv, recommend);
    }

    AbstractedMemory::free(zone_work);

    return recommend;
}

