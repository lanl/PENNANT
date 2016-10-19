/*
 * PredictorTask.cc
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

#include "PredictorTask.hh"

#include <algorithm>

#include "GenerateMesh.hh"
#include "Hydro.hh"
#include "InputParameters.hh"
#include "LocalMesh.hh"
#include "LogicalStructured.hh"
#include "PolyGas.hh"
#include "QCS.hh"
#include "TTS.hh"
#include "Memory.hh"
#include "Vec2.hh"

enum idx {
    ZERO,
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    ELEVEN,
};

PredictorTask::PredictorTask(LogicalRegion mesh_zones,
        LogicalRegion mesh_sides,
        LogicalRegion mesh_zone_pts,
        LogicalRegion mesh_points,
        LogicalRegion hydro_zones,
        LogicalRegion hydro_sides_and_corners,
        LogicalRegion hydro_points,
        void *args, const size_t &size)
	 : TaskLauncher(PredictorTask::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(mesh_sides, READ_ONLY, EXCLUSIVE, mesh_sides));
    add_field(ZERO, FID_MAP_CRN2CRN_NEXT);
    add_field(ZERO, FID_SMAP_SIDE_TO_ZONE);
    add_field(ZERO, FID_SMAP_SIDE_TO_PT1);
    add_field(ZERO, FID_SMAP_SIDE_TO_EDGE);
    add_field(ZERO, FID_SMF);
    add_region_requirement(RegionRequirement(mesh_zones, READ_WRITE, EXCLUSIVE, mesh_zones));
    add_field(ONE, FID_ZDL);
    add_field(ONE, FID_ZVOL0);
    add_region_requirement(RegionRequirement(hydro_sides_and_corners, READ_WRITE, EXCLUSIVE, hydro_sides_and_corners));
    add_field(TWO, FID_CFTOT);
    add_field(TWO, FID_SFQ);
    add_field(TWO, FID_SFT);
    add_field(TWO, FID_SFP);
    add_field(TWO, FID_CMASWT);
    add_region_requirement(RegionRequirement(mesh_zone_pts, READ_ONLY, EXCLUSIVE, mesh_zone_pts));
    add_field(THREE, FID_ZONE_PTS_PTR);
    add_region_requirement(RegionRequirement(hydro_points, READ_ONLY, EXCLUSIVE, hydro_points));
    add_field(FOUR, FID_PU);
    add_region_requirement(RegionRequirement(mesh_zones, READ_ONLY, EXCLUSIVE, mesh_zones));
    add_field(FIVE, FID_ZVOL);
    add_region_requirement(RegionRequirement(hydro_points, READ_WRITE, EXCLUSIVE, hydro_points));
    add_field(SIX, FID_PU0);
    add_region_requirement(RegionRequirement(hydro_zones, READ_ONLY, EXCLUSIVE, hydro_zones));
    add_field(SEVEN, FID_ZR);
    add_field(SEVEN, FID_ZE);
    add_field(SEVEN, FID_ZM);
    add_field(SEVEN, FID_ZWR);
    add_region_requirement(RegionRequirement(hydro_zones, READ_WRITE, EXCLUSIVE, hydro_zones));
    add_field(EIGHT, FID_ZDU);
    add_field(EIGHT, FID_ZSS);
    add_field(EIGHT, FID_ZP);
    add_region_requirement(RegionRequirement(mesh_points, READ_WRITE, EXCLUSIVE, mesh_points));
    add_field(NINE, FID_PXP);
    add_field(NINE, FID_PX0);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(TEN, FID_PX);
}

/*static*/ const char * const PredictorTask::TASK_NAME = "PredictorTask";


/*static*/
void PredictorTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == ELEVEN);
	assert(task->regions.size() == ELEVEN);

    DoCycleTasksArgs args;
    DoCycleTasksArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    assert(task->regions[ZERO].privilege_fields.size() == 5);
    LogicalStructured mesh_sides(ctx, runtime, regions[ZERO]);
    const int* map_crn2crn_next = mesh_sides.getRawPtr<int>(FID_MAP_CRN2CRN_NEXT);
    const int* map_side2zone = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_ZONE);
    const int* map_side2pt1 = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT1);
    const int* map_side2edge = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_EDGE);
    const double* side_mass_frac = mesh_sides.getRawPtr<double>(FID_SMF);

    assert(task->regions[THREE].privilege_fields.size() == 1);
    LogicalStructured mesh_zone_pts(ctx, runtime, regions[THREE]);
    const int* zone_pts_ptr = mesh_zone_pts.getRawPtr<int>(FID_ZONE_PTS_PTR);

    assert(task->regions[FOUR].privilege_fields.size() == 1);
    LogicalStructured hydro_points(ctx, runtime, regions[FOUR]);
    const double2* pt_vel = hydro_points.getRawPtr<double2>(FID_PU);

    assert(task->regions[FIVE].privilege_fields.size() == 1);
    LogicalStructured mesh_zones(ctx, runtime, regions[FIVE]);
    const double* zone_vol = mesh_zones.getRawPtr<double>(FID_ZVOL);

    assert(task->regions[SEVEN].privilege_fields.size() == 4);
    LogicalStructured hydro_zones(ctx, runtime, regions[SEVEN]);
    const double* zone_rho = hydro_zones.getRawPtr<double>(FID_ZR);
    const double* zone_energy_density = hydro_zones.getRawPtr<double>(FID_ZE);
    const double* zone_mass = hydro_zones.getRawPtr<double>(FID_ZM);
    const double* zone_work_rate = hydro_zones.getRawPtr<double>(FID_ZWR);

    assert(task->regions[TEN].privilege_fields.size() == 1);
    LogicalStructured mesh_points(ctx, runtime, regions[TEN]);
    const double2* pt_x = mesh_points.getRawPtr<double2>(FID_PX);

    assert(task->regions[SIX].privilege_fields.size() == 1);
    LogicalStructured hydro_write_points(ctx, runtime, regions[SIX]);
    double2* pt_vel0 = hydro_write_points.getRawPtr<double2>(FID_PU0);

    double* edge_len = AbstractedMemory::alloc<double>(args.num_edges);
    double2* edge_x_pred = AbstractedMemory::alloc<double2>(args.num_edges);

    assert(task->regions[ONE].privilege_fields.size() == 2);
    LogicalStructured mesh_write_zones(ctx, runtime, regions[ONE]);
    double* zone_dl = mesh_write_zones.getRawPtr<double>(FID_ZDL);
    double* zone_vol0 = mesh_write_zones.getRawPtr<double>(FID_ZVOL0);
    double2* zone_x_pred = AbstractedMemory::alloc<double2>(mesh_write_zones.size());
    double* zone_vol_pred = AbstractedMemory::alloc<double>(mesh_write_zones.size());
    double* zone_area_pred = AbstractedMemory::alloc<double>(mesh_write_zones.size());

    assert(task->regions[EIGHT].privilege_fields.size() == 3);
    LogicalStructured hydro_write_zones(ctx, runtime, regions[EIGHT]);
    double* zone_dvel = hydro_write_zones.getRawPtr<double>(FID_ZDU);
    double* zone_sound_speed = hydro_write_zones.getRawPtr<double>(FID_ZSS);
    double* zone_pressure = hydro_write_zones.getRawPtr<double>(FID_ZP);
    double* zone_rho_pred = AbstractedMemory::alloc<double>(hydro_write_zones.size());

    assert(task->regions[TWO].privilege_fields.size() == 5);
    LogicalStructured write_sides_and_corners(ctx, runtime, regions[TWO]);
    double2* crnr_force_tot = write_sides_and_corners.getRawPtr<double2>(FID_CFTOT);
    double2* side_force_visc = write_sides_and_corners.getRawPtr<double2>(FID_SFQ);
    double2* side_force_tts = write_sides_and_corners.getRawPtr<double2>(FID_SFT);
    double2* side_force_pres = write_sides_and_corners.getRawPtr<double2>(FID_SFP);
    double* crnr_weighted_mass = write_sides_and_corners.getRawPtr<double>(FID_CMASWT);

    double* side_area_pred = AbstractedMemory::alloc<double>(args.num_sides);
    double2* side_surfp = AbstractedMemory::alloc<double2>(args.num_sides);

    assert(task->regions[NINE].privilege_fields.size() == 2);
    LogicalStructured mesh_write_points(ctx, runtime, regions[NINE]);
    double2* pt_x_pred = mesh_write_points.getRawPtr<double2>(FID_PXP);
    double2* pt_x0 = mesh_write_points.getRawPtr<double2>(FID_PX0);

    InputParameters input_params;
    input_params.meshtype = args.meshtype;
    input_params.directs.nzones_x = args.nzones_x;
    input_params.directs.nzones_y = args.nzones_y;
    input_params.directs.ntasks = args.num_subregions;
    input_params.directs.task_id = args.my_color;
    GenerateMesh* generate_mesh = new GenerateMesh(input_params);

    for (int pch = 0; pch < (args.point_chunk_CRS.size()-1); ++pch) {
        int pfirst = args.point_chunk_CRS[pch];
        int plast = args.point_chunk_CRS[pch+1];

        // save off point variable values from previous cycle
        std::copy(&pt_x[pfirst], &pt_x[plast], &pt_x0[pfirst]);
        std::copy(&pt_vel[pfirst], &pt_vel[plast], &pt_vel0[pfirst]);

        // ===== Predictor step =====
        // 1. advance mesh to center of time step
        Hydro::advPosHalf(args.dt, pfirst, plast,
                pt_x0, pt_vel0,
                pt_x_pred);
    }

    for (int side_chunk = 0; side_chunk < (args.side_chunk_CRS.size()-1); ++side_chunk) {
        int sfirst = args.side_chunk_CRS[side_chunk];
        int slast = args.side_chunk_CRS[side_chunk+1];
        int zfirst = LocalMesh::side_zone_chunks_first(side_chunk, map_side2zone, args.side_chunk_CRS);
        int zlast = LocalMesh::side_zone_chunks_last(side_chunk, map_side2zone, args.side_chunk_CRS);

        // save off zone variable values from previous cycle
        std::copy(&zone_vol[zfirst], &zone_vol[zlast], &zone_vol0[zfirst]);

        // 1a. compute new mesh geometry
        LocalMesh::calcCtrs(sfirst, slast, pt_x_pred,
                map_side2zone, args.num_sides, args.num_zones, map_side2pt1, map_side2edge, zone_pts_ptr,
                edge_x_pred, zone_x_pred);

        LocalMesh::calcVols(sfirst, slast, pt_x_pred, zone_x_pred,
                map_side2zone, args.num_sides, args.num_zones, map_side2pt1, zone_pts_ptr,
                side_area_pred, nullptr, zone_area_pred, zone_vol_pred);

        LocalMesh::calcMedianMeshSurfVecs(sfirst, slast, map_side2zone,
                map_side2edge, edge_x_pred, zone_x_pred,
                side_surfp);

        LocalMesh::calcEdgeLen(sfirst, slast, map_side2pt1, map_side2edge,
                map_side2zone, zone_pts_ptr, pt_x_pred,
                edge_len);

        LocalMesh::calcCharacteristicLen(sfirst, slast, map_side2zone, map_side2edge,
                zone_pts_ptr, side_area_pred, edge_len, args.num_sides, args.num_zones,
                zone_dl);

        // 2. compute point masses
        Hydro::calcRho(zone_vol_pred, zone_mass, zone_rho_pred, zfirst, zlast);

        Hydro::calcCrnrMass(sfirst, slast, zone_area_pred, side_mass_frac,
                map_side2zone, zone_pts_ptr, zone_rho_pred,
                crnr_weighted_mass);

        // 3. compute material state (half-advanced)
        PolyGas::calcStateAtHalf(zone_rho, zone_vol_pred, zone_vol0, zone_energy_density, zone_work_rate, zone_mass, args.dt,
                zone_pressure, zone_sound_speed, zfirst, zlast, args.gamma,
                        args.ssmin);

        // 4. compute forces
        PolyGas::calcForce(zone_pressure, side_surfp, side_force_pres, sfirst, slast,
                map_side2zone);

        TTS::calcForce(zone_area_pred, zone_rho_pred, zone_sound_speed, side_area_pred,
                side_mass_frac, side_surfp, side_force_tts,
                sfirst, slast,
                map_side2zone, args.ssmin, args.alpha);

        QCS::calcForce(side_force_visc, sfirst, slast,
                args.num_sides, args.num_zones, pt_vel, edge_x_pred,
                zone_x_pred, edge_len, map_side2zone, map_side2pt1,
                zone_pts_ptr, map_side2edge, pt_x_pred,
                zone_rho_pred, zone_sound_speed, args.qgamma,
                args.q1, args.q2,
                map_side2zone[sfirst],
                (slast < args.num_sides ? map_side2zone[slast] : args.num_zones),
                zone_dvel);

        Hydro::sumCrnrForce(side_force_pres, side_force_visc, side_force_tts,
                map_side2zone, zone_pts_ptr, sfirst, slast,
                crnr_force_tot);
    } // side chunk

    AbstractedMemory::free(side_surfp);
    AbstractedMemory::free(side_area_pred);
    AbstractedMemory::free(zone_rho_pred);
    AbstractedMemory::free(zone_area_pred);
    AbstractedMemory::free(zone_vol_pred);
    AbstractedMemory::free(zone_x_pred);
    AbstractedMemory::free(edge_x_pred);
    AbstractedMemory::free(edge_len);
}

