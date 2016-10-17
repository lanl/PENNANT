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
#include "Vec2.hh"


PredictorTask::PredictorTask(LogicalRegion mesh_zones,
        LogicalRegion mesh_sides,
        LogicalRegion mesh_zone_pts,
        LogicalRegion mesh_points,
        LogicalRegion mesh_edges,
        LogicalRegion hydro_zones,
        LogicalRegion hydro_sides_and_corners,
        LogicalRegion hydro_points,
        void *args, const size_t &size)
	 : TaskLauncher(PredictorTask::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(mesh_sides, READ_ONLY, EXCLUSIVE, mesh_sides));
    add_field(0/*idx*/, FID_MAP_CRN2CRN_NEXT);
    add_field(0/*idx*/, FID_SMAP_SIDE_TO_ZONE);
    add_field(0/*idx*/, FID_SMAP_SIDE_TO_PT1);
    add_field(0/*idx*/, FID_SMAP_SIDE_TO_EDGE);
    add_field(0/*idx*/, FID_SMF);
    add_region_requirement(RegionRequirement(mesh_zones, READ_WRITE, EXCLUSIVE, mesh_zones));
    add_field(1/*idx*/, FID_ZDL);
    add_field(1/*idx*/, FID_ZAREAP);
    add_field(1/*idx*/, FID_ZVOLP);
    add_field(1/*idx*/, FID_ZXP);
    add_field(1/*idx*/, FID_ZVOL0);
    add_region_requirement(RegionRequirement(hydro_sides_and_corners, READ_WRITE, EXCLUSIVE, hydro_sides_and_corners));
    add_field(2/*idx*/, FID_CFTOT);
    add_field(2/*idx*/, FID_SFQ);
    add_field(2/*idx*/, FID_SFT);
    add_field(2/*idx*/, FID_SFP);
    add_field(2/*idx*/, FID_CMASWT);
    add_region_requirement(RegionRequirement(mesh_zone_pts, READ_ONLY, EXCLUSIVE, mesh_zone_pts));
    add_field(3/*idx*/, FID_ZONE_PTS_PTR);
    add_region_requirement(RegionRequirement(hydro_points, READ_ONLY, EXCLUSIVE, hydro_points));
    add_field(4/*idx*/, FID_PU);
    add_region_requirement(RegionRequirement(mesh_edges, READ_WRITE, EXCLUSIVE, mesh_edges));
    add_field(5/*idx*/, FID_EXP);
    add_field(5/*idx*/, FID_ELEN);
    add_region_requirement(RegionRequirement(mesh_zones, READ_ONLY, EXCLUSIVE, mesh_zones));
    add_field(6/*idx*/, FID_ZVOL);
    add_region_requirement(RegionRequirement(hydro_points, READ_WRITE, EXCLUSIVE, hydro_points));
    add_field(7/*idx*/, FID_PU0);
    add_region_requirement(RegionRequirement(hydro_zones, READ_ONLY, EXCLUSIVE, hydro_zones));
    add_field(8/*idx*/, FID_ZR);
    add_field(8/*idx*/, FID_ZE);
    add_field(8/*idx*/, FID_ZM);
    add_field(8/*idx*/, FID_ZWR);
    add_region_requirement(RegionRequirement(hydro_zones, READ_WRITE, EXCLUSIVE, hydro_zones));
    add_field(9/*idx*/, FID_ZDU);
    add_field(9/*idx*/, FID_ZSS);
    add_field(9/*idx*/, FID_ZP);
    add_field(9/*idx*/, FID_ZRP);
    add_region_requirement(RegionRequirement(mesh_sides, READ_WRITE, EXCLUSIVE, mesh_sides));
    add_field(10/*idx*/, FID_SSURFP);
    add_field(10/*idx*/, FID_SAREAP);
    add_field(10/*idx*/, FID_SVOLP);
    add_region_requirement(RegionRequirement(mesh_points, READ_WRITE, EXCLUSIVE, mesh_points));
    add_field(11/*idx*/, FID_PXP);
    add_field(11/*idx*/, FID_PX0);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(12/*idx*/, FID_PX);
}

/*static*/ const char * const PredictorTask::TASK_NAME = "PredictorTask";


/*static*/
void PredictorTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == 13);
	assert(task->regions.size() == 13);

    assert(task->regions[0].privilege_fields.size() == 5);
    LogicalStructured mesh_sides(ctx, runtime, regions[0]);
    const int* map_crn2crn_next = mesh_sides.getRawPtr<int>(FID_MAP_CRN2CRN_NEXT);
    const int* map_side2zone = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_ZONE);
    const int* map_side2pt1 = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT1);
    const int* map_side2edge = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_EDGE);
    const double* side_mass_frac = mesh_sides.getRawPtr<double>(FID_SMF);

    assert(task->regions[3].privilege_fields.size() == 1);
    LogicalStructured mesh_zone_pts(ctx, runtime, regions[3]);
    const int* zone_pts_ptr = mesh_zone_pts.getRawPtr<int>(FID_ZONE_PTS_PTR);

    assert(task->regions[4].privilege_fields.size() == 1);
    LogicalStructured hydro_points(ctx, runtime, regions[4]);
    const double2* pt_vel = hydro_points.getRawPtr<double2>(FID_PU);

    assert(task->regions[6].privilege_fields.size() == 1);
    LogicalStructured mesh_zones(ctx, runtime, regions[6]);
    const double* zone_vol = mesh_zones.getRawPtr<double>(FID_ZVOL);

    assert(task->regions[8].privilege_fields.size() == 4);
    LogicalStructured hydro_zones(ctx, runtime, regions[8]);
    const double* zone_rho = hydro_zones.getRawPtr<double>(FID_ZR);
    const double* zone_energy_density = hydro_zones.getRawPtr<double>(FID_ZE);
    const double* zone_mass = hydro_zones.getRawPtr<double>(FID_ZM);
    const double* zone_work_rate = hydro_zones.getRawPtr<double>(FID_ZWR);

    assert(task->regions[12].privilege_fields.size() == 1);
    LogicalStructured mesh_points(ctx, runtime, regions[12]);
    const double2* pt_x = mesh_points.getRawPtr<double2>(FID_PX);

    assert(task->regions[7].privilege_fields.size() == 1);
    LogicalStructured hydro_write_points(ctx, runtime, regions[7]);
    double2* pt_vel0 = hydro_write_points.getRawPtr<double2>(FID_PU0);

    assert(task->regions[5].privilege_fields.size() == 2);
    LogicalStructured mesh_write_edges(ctx, runtime, regions[5]);
    double2* edge_x_pred = mesh_write_edges.getRawPtr<double2>(FID_EXP);
    double* edge_len = mesh_write_edges.getRawPtr<double>(FID_ELEN);

    assert(task->regions[1].privilege_fields.size() == 5);
    LogicalStructured mesh_write_zones(ctx, runtime, regions[1]);
    double* zone_dl = mesh_write_zones.getRawPtr<double>(FID_ZDL);
    double* zone_area_pred = mesh_write_zones.getRawPtr<double>(FID_ZAREAP);
    double* zone_vol_pred = mesh_write_zones.getRawPtr<double>(FID_ZVOLP);
    double2* zone_x_pred = mesh_write_zones.getRawPtr<double2>(FID_ZXP);
    double* zone_vol0 = mesh_write_zones.getRawPtr<double>(FID_ZVOL0);

    assert(task->regions[9].privilege_fields.size() == 4);
    LogicalStructured hydro_write_zones(ctx, runtime, regions[9]);
    double* zone_dvel = hydro_write_zones.getRawPtr<double>(FID_ZDU);
    double* zone_sound_speed = hydro_write_zones.getRawPtr<double>(FID_ZSS);
    double* zone_pressure = hydro_write_zones.getRawPtr<double>(FID_ZP);
    double* zone_rho_pred = hydro_write_zones.getRawPtr<double>(FID_ZRP);

    assert(task->regions[2].privilege_fields.size() == 5);
    LogicalStructured write_sides_and_corners(ctx, runtime, regions[2]);
    double2* crnr_force_tot = write_sides_and_corners.getRawPtr<double2>(FID_CFTOT);
    double2* side_force_visc = write_sides_and_corners.getRawPtr<double2>(FID_SFQ);
    double2* side_force_tts = write_sides_and_corners.getRawPtr<double2>(FID_SFT);
    double2* side_force_pres = write_sides_and_corners.getRawPtr<double2>(FID_SFP);
    double* crnr_weighted_mass = write_sides_and_corners.getRawPtr<double>(FID_CMASWT);

    assert(task->regions[10].privilege_fields.size() == 3);
    LogicalStructured mesh_write_sides(ctx, runtime, regions[10]);
    double2* side_surfp = mesh_write_sides.getRawPtr<double2>(FID_SSURFP);
    double* side_area_pred = mesh_write_sides.getRawPtr<double>(FID_SAREAP);
    double* side_vol_pred = mesh_write_sides.getRawPtr<double>(FID_SVOLP);

    assert(task->regions[11].privilege_fields.size() == 2);
    LogicalStructured mesh_write_points(ctx, runtime, regions[11]);
    double2* pt_x_pred = mesh_write_points.getRawPtr<double2>(FID_PXP);
    double2* pt_x0 = mesh_write_points.getRawPtr<double2>(FID_PX0);

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
                side_area_pred, side_vol_pred, zone_area_pred, zone_vol_pred);

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
}

