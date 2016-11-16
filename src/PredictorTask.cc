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
#include <cmath>

#include "GenerateMesh.hh"
#include "Hydro.hh"
#include "InputParameters.hh"
#include "LocalMesh.hh"
#include "LogicalStructured.hh"
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
    ELEVEN
};

using namespace std;


PredictorTask::PredictorTask(LogicalRegion mesh_zones,
        LogicalRegion mesh_sides,
        LogicalRegion mesh_zone_pts,
        LogicalRegion mesh_points,
        LogicalRegion side_chunks,
        LogicalRegion mesh_edges,
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
    add_field(ZERO, FID_SMAP_SIDE_TO_PT2);
    add_field(ZERO, FID_SMAP_SIDE_TO_EDGE);
    add_field(ZERO, FID_SMF);
    add_region_requirement(RegionRequirement(mesh_zones, READ_WRITE, EXCLUSIVE, mesh_zones));
    add_field(ONE, FID_ZDL);
    add_field(ONE, FID_ZVOL0);
    add_field(ONE, FID_Z_DBL2_TEMP);
    add_field(ONE, FID_Z_DBL_TEMP1);
    add_field(ONE, FID_Z_DBL_TEMP2);
    add_region_requirement(RegionRequirement(hydro_sides_and_corners, READ_WRITE, EXCLUSIVE, hydro_sides_and_corners));
    add_field(TWO, FID_CFTOT);
    add_field(TWO, FID_SFQ);
    add_field(TWO, FID_SFT);
    add_field(TWO, FID_SFP);
    add_field(TWO, FID_CMASWT);
    add_field(TWO, FID_S_DBL_TEMP);
    add_region_requirement(RegionRequirement(mesh_zone_pts, READ_ONLY, EXCLUSIVE, mesh_zone_pts));
    add_field(THREE, FID_ZONE_PTS_PTR);
    add_region_requirement(RegionRequirement(hydro_points, READ_ONLY, EXCLUSIVE, hydro_points));
    add_field(FOUR, FID_PU);
    add_region_requirement(RegionRequirement(mesh_zones, READ_ONLY, EXCLUSIVE, mesh_zones));
    add_field(FIVE, FID_ZVOL);
    add_region_requirement(RegionRequirement(hydro_zones, READ_ONLY, EXCLUSIVE, hydro_zones));
    add_field(SIX, FID_ZR);
    add_field(SIX, FID_ZE);
    add_field(SIX, FID_ZM);
    add_field(SIX, FID_ZWR);
    add_region_requirement(RegionRequirement(hydro_zones, READ_WRITE, EXCLUSIVE, hydro_zones));
    add_field(SEVEN, FID_ZDU);
    add_field(SEVEN, FID_ZSS);
    add_field(SEVEN, FID_ZP);
    add_region_requirement(RegionRequirement(mesh_points, READ_ONLY, EXCLUSIVE, mesh_points));
    add_field(EIGHT, FID_PXP);
    add_region_requirement(RegionRequirement(side_chunks, READ_ONLY, EXCLUSIVE, side_chunks));
    add_field(NINE, FID_SIDE_CHUNKS_CRS);
    add_region_requirement(RegionRequirement(mesh_edges, WRITE_DISCARD, EXCLUSIVE, mesh_edges));
    add_field(TEN, FID_E_DBL2_TEMP);
    add_field(TEN, FID_E_DBL_TEMP);
}

/*static*/ const char * const PredictorTask::TASK_NAME = "PredictorTask";


static
void PolyGascalcEOS(
        const double* zr,
        const double* ze,
        double* zp,
        double* z0per,
        double* zss,
        const int zfirst,
        const int zlast,
        const double gamma,
        const double ssmin) {

    const double gm1 = gamma - 1.;
    const double ss2 = max(ssmin * ssmin, 1.e-99);

    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        int z0 = z - zfirst;
        double rx = zr[z];
        double ex = max(ze[z], 0.0);
        double px = gm1 * rx * ex;
        double prex = gm1 * ex;
        double perx = gm1 * rx;
        double csqd = max(ss2, prex + perx * px / (rx * rx));
        zp[z] =  px;
        z0per[z0] = perx;
        zss[z] = sqrt(csqd);
    }

}


static
void PolyGascalcStateAtHalf(
        const double* zr0,
        const double* zvolp,
        const double* zvol0,
        const double* ze,
        const double* zwrate,
        const double* zm,
        const double dt,
        double* zp,
        double* zss,
        const int zfirst,
        const int zlast,
        const double gamma,
        const double ssmin) {

    double* z0per = AbstractedMemory::alloc<double>(zlast - zfirst);

    const double dth = 0.5 * dt;

    // compute EOS at beginning of time step
    PolyGascalcEOS(zr0, ze, zp, z0per, zss, zfirst, zlast, gamma, ssmin);

    // now advance pressure to the half-step
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        int z0 = z - zfirst;
        double zminv = 1. / zm[z];
        double dv = (zvolp[z] - zvol0[z]) * zminv;
        double bulk = zr0[z] * zss[z] * zss[z];
        double denom = 1. + 0.5 * z0per[z0] * dv;
        double src = zwrate[z] * dth * zminv;
        double value = zp[z] + (z0per[z0] * src - zr0[z] * bulk * dv) / denom;
        zp[z] = value;
    }

    AbstractedMemory::free(z0per);
}


static void Force(
        double2* sfq,
        const int sfirst,
        const int slast,
        const int nums,
        const int numz,
        const double2* pu,
        const double2* edge_x_pred,
        const double2* zone_x_pred,
        const double* elen,
        const int* map_side2zone,
        const int* map_side2pt1,
        const int* map_side2pt2,
        const int* zone_pts_ptr,
        const int* map_side2edge,
        const double2* pt_x_pred,
        const double* zss,
        const double qgamma,
        const double q1,
        const double q2,
        int zfirst,
        int zlast,
        double* zdu,
        const double* zone_area_pred,
        const double* side_area_pred,
        const double* side_mass_frac,
        double2* sf,
        const double ssmin,
        const double alfa,
        double* crnr_weighted_mass,
        const double* zone_vol_pred,
        const double* zone_mass,
        const double* zone_pressure,
        double2* side_force_pres)
{
    //  Side density:
    //    srho = sm/sv = zr (sm/zm) / (sv/zv)
    //  Side pressure:
    //    sp   = zp + alfa dpdr (srho-zr)
    //         = zp + sdp
    //  Side delta pressure:
    //    sdp  = alfa dpdr (srho-zr)
    //         = alfa c**2 (srho-zr)
    //
    //    Notes: smf stores (sm/zm)
    //           svfac stores (sv/zv)

    // declare temporary variables
    double* c0area = AbstractedMemory::alloc<double>(slast - sfirst);
    double* c0cos = AbstractedMemory::alloc<double>(slast - sfirst);
    double2* c0qe = AbstractedMemory::alloc<double2>(2 * (slast - sfirst));
    double* c0w = AbstractedMemory::alloc<double>(slast - sfirst);


    // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    // [2.1] Find the corner divergence
    // [2.2] Compute the cos angle for c
    // [2.3] Find the evolution factor c0evol(c) and the Delta u(c) = du(c)
    // [2.4] Find the weights c0w(c)


    // Routine number [2]  in the full algorithm
    //     [2.1] Find the corner divergence
    //     [2.2] Compute the cos angle for c
    //     [2.3] Find the evolution factor c0evol(c)
    //           and the Delta u(c) = du(c)

        double2* z0uc = AbstractedMemory::alloc<double2>(zlast - zfirst);

        // [1] Compute a zone-centered velocity
        fill(&z0uc[0], &z0uc[zlast-zfirst], double2(0., 0.));
        for (int s = sfirst; s < slast; ++s) {
            int p = map_side2pt1[s];
            int z = map_side2zone[s];
            int z0 = z - zfirst;
            z0uc[z0] += pu[p];
        }

        for (int z = zfirst; z < zlast; ++z) {
            int z0 = z - zfirst;
            z0uc[z0] /= (double) LocalMesh::zoneNPts(z, zone_pts_ptr);
        }

        // [2] Divergence at the corner
        const double2* pt_x_pred_ = pt_x_pred;
        #pragma ivdep
        for (int s = sfirst; s < slast; ++s) {
            int s2 = s;
            int sprev = LocalMesh::mapSideToSidePrev(s2, map_side2zone, zone_pts_ptr);
            // Associated zone, corner, point
            int z = map_side2zone[sprev];
            int z0 = z - zfirst;
            int c0 = s - sfirst;
            int p = map_side2pt2[sprev];
            // Points
            int p1 = map_side2pt1[sprev];
            int p2 = map_side2pt2[s2];
            // Edges
            int e1 = map_side2edge[sprev];
            int e2 = map_side2edge[s2];

            // Velocities and positions
            // 0 = point p
            double2 xp0 = pt_x_pred_[p];
            // 1 = edge e2
            double2 xp1 = edge_x_pred[e2];
            // 2 = zone center z
            double2 xp2 = zone_x_pred[z];
            // 3 = edge e1
            double2 xp3 = edge_x_pred[e1];

            // compute 2d cartesian volume of corner
            double cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1);
            c0area[c0] = cvolume;

            // compute cosine angle
            double2 v1 = xp3 - xp0;
            double2 v2 = xp1 - xp0;
            double de1 = elen[e1];
            double de2 = elen[e2];
            double minelen = min(de1, de2);
            c0cos[c0] = ((minelen < 1.e-12) ?
                    0. :
                    4. * dot(v1, v2) / (de1 * de2));

            // [5.1] Preparation of extra variables
            double csin2 = 1.0 - c0cos[c0] * c0cos[c0];
            c0w[c0]   = ((csin2 < 1.e-4) ? 0. : c0area[c0] / csin2);
            c0cos[c0] = ((csin2 < 1.e-4) ? 0. : c0cos[c0]);
        }  // for s


    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    // [4.1] Compute cmu = (1-psi) . crho . zKUR . c0evol
    // [4.2] Compute the q vector associated with c on edges
    //       e1=[n0,n1], e2=[n1,n2]
    //       c0qe(2,c) = cmu(c).( u(n2)-u(n1) ) / l_{n1->n2}
    //       c0qe(1,c) = cmu(c).( u(n1)-u(n0) ) / l_{n0->n1}


    // Routine number [4]  in the full algorithm CS2DQforce(...)
    const double gammap1 = qgamma + 1.0;

    double* z0tmp = AbstractedMemory::alloc<double>(zlast - zfirst);

    fill(&z0tmp[0], &z0tmp[zlast-zfirst], 0.);

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int sprev = LocalMesh::mapSideToSidePrev(s, map_side2zone, zone_pts_ptr);
        int c10 = s - sfirst;
        int p = map_side2pt2[sprev];
        // Associated point and edge 1
        int p1 = map_side2pt1[sprev];
        int e1 = map_side2edge[sprev];
        // Associated point and edge 2
        int p2 = map_side2pt2[s];
        int e2 = map_side2edge[s];
        int z = map_side2zone[s];

        // Associated zone, corner, point
        int zprev = map_side2zone[sprev];
        int z0 = zprev - zfirst;

        // Velocities and positions
        // 0 = point p
        double2 up0 = pu[p];
        double2 xp0 = pt_x_pred_[p];
        // 1 = edge e2
        double2 up1 = 0.5 * (pu[p] + pu[p2]);
        double2 xp1 = edge_x_pred[e2];
        // 2 = zone center z
        double2 up2 = z0uc[z0];
        double2 xp2 = zone_x_pred[z];
        // 3 = edge e1
        double2 up3 = 0.5 * (pu[p1] + pu[p]);
        double2 xp3 = edge_x_pred[e1];

        // compute cosine angle
        double2 v1 = xp3 - xp0;
        double2 v2 = xp1 - xp0;
        double de1 = elen[e1];
        double de2 = elen[e2];
        double minelen = min(de1, de2);

        // compute delta velocity
        double dv1 = length2(up1 + up2 - up0 - up3);
        double dv2 = length2(up2 + up3 - up0 - up1);
        double du = sqrt(max(dv1, dv2));

        // average corner-centered velocity
        double2 duav = 0.25 * (up0 + up1 + up2 + up3);

        // compute evolution factor
        double2 dxx1 = 0.5 * (xp1 + xp2 - xp0 - xp3);
        double2 dxx2 = 0.5 * (xp2 + xp3 - xp0 - xp1);
        double dx1 = length(dxx1);
        double dx2 = length(dxx2);

        double test1 = abs(dot(dxx1, duav) * dx2);
        double test2 = abs(dot(dxx2, duav) * dx1);
        double num = (test1 > test2 ? dx1 : dx2);
        double den = (test1 > test2 ? dx2 : dx1);
        double r = num / den;
        double evol = sqrt(4.0 * c0area[c10] * r);
        evol = min(evol, 2.0 * minelen);

        // compute divergence of corner
        double c0div = (cross(up2 - up0, xp3 - xp1) -
                cross(up3 - up1, xp2 - xp0)) /
                (2.0 * c0area[c10]);

        double c0evol = (c0div < 0.0 ? evol : 0.);
        double c0du = (c0div < 0.0 ? du   : 0.);

        // Hydro::calcRho
        double zone_rho_pred = zone_mass[z] / zone_vol_pred[z];

        // Hydro::calcCrnrMass
        double m = zone_rho_pred * zone_area_pred[z] * 0.5 * (side_mass_frac[s] + side_mass_frac[sprev]);
        crnr_weighted_mass[s] = m;

        // LocalMesh::calcMedianMeshSurfVecs
        double2 side_surfp = rotateCCW(edge_x_pred[e2] - zone_x_pred[z]);

        // TTS
        double svfacinv = zone_area_pred[z] / side_area_pred[s];
        double srho = zone_rho_pred * side_mass_frac[s] * svfacinv;
        double sstmp = max(zss[z], ssmin);
        sstmp = alfa * sstmp * sstmp;
        double sdp = sstmp * (srho - zone_rho_pred);
        double2 sqq = -sdp * side_surfp;
        sf[s] = sqq;

        // PolyGas::calcForce
        double2 sfx = -zone_pressure[z] * side_surfp;
        side_force_pres[s] = sfx;

        // [4.1] Compute the c0rmu (real Kurapatenko viscous scalar)
        // Kurapatenko form of the viscosity
        double ztmp2 = q2 * 0.25 * gammap1 * c0du;
        double ztmp1 = q1 * zss[z];
        double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
        // Compute c0rmu for each corner
        double rmu = zkur * zone_rho_pred * c0evol;
        double c0rmu = ((c0div > 0.0) ? 0. : rmu);

        // [4.2] Compute the c0qe for each corner

        // Compute: c0qe(1,2,3)=edge 1, y component (2nd), 3rd corner
        //          c0qe(2,1,3)=edge 2, x component (1st)
        c0qe[2 * c10]     = c0rmu * (pu[p] - pu[p1]) / elen[e1];
        c0qe[2 * c10 + 1] = c0rmu * (pu[p2] - pu[p]) / elen[e2];

    } // for s

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        // [5] Compute the Q forces
        // Routine number [5]  in the full algorithm CS2DQforce(...)
        // Associated corners 1 and 2, and edge
        int c10 = s - sfirst;
        int c2 = LocalMesh::mapSideToSideNext(s, map_side2zone, zone_pts_ptr);
        int c20 = c2 - sfirst;
        int e = map_side2edge[s];
        // Edge length for s, c2 contribution to s
        double el = elen[e];

        // [5.2] Set-Up the forces on corners
        sfq[s] = (c0w[c10] * (c0qe[2*c10+1] + c0cos[c10] * c0qe[2*c10]) +
                  c0w[c20] * (c0qe[2*c20] + c0cos[c20] * c0qe[2*c20+1]))
            / el;

        int p3 = map_side2pt1[s];
        int p4 = map_side2pt2[s];
        int z = map_side2zone[s];
        int z0 = z - zfirst;

        // [6] Set velocity difference to use to compute timestep
        double2 dx = pt_x_pred[p4] - pt_x_pred[p3];
        double2 du = pu[p4] - pu[p3];
        double lenx = elen[e];
        double dux = dot(du, dx);
        // Routine number [6] in the full algorithm
        dux = (lenx > 0. ? abs(dux) / lenx : 0.);
        z0tmp[z0] = max(z0tmp[z0], dux);
    }

    AbstractedMemory::free(z0uc);
    AbstractedMemory::free(c0w);
    AbstractedMemory::free(c0area);
    AbstractedMemory::free(c0cos);
    AbstractedMemory::free(c0qe);

    for (int z = zfirst; z < zlast; ++z) {
        int z0 = z - zfirst;
        zdu[z] = q1 * zss[z] + 2. * q2 * z0tmp[z0];
    }

    AbstractedMemory::free(z0tmp);
}


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

    assert(task->regions[ZERO].privilege_fields.size() == 6);
    LogicalStructured mesh_sides(ctx, runtime, regions[ZERO]);
    const int* map_crn2crn_next = mesh_sides.getRawPtr<int>(FID_MAP_CRN2CRN_NEXT);
    const int* map_side2zone = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_ZONE);
    const int* map_side2pt1 = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT1);
    const int* map_side2pt2 = mesh_sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT2);
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

    assert(task->regions[SIX].privilege_fields.size() == 4);
    LogicalStructured hydro_zones(ctx, runtime, regions[SIX]);
    const double* zone_rho = hydro_zones.getRawPtr<double>(FID_ZR);
    const double* zone_energy_density = hydro_zones.getRawPtr<double>(FID_ZE);
    const double* zone_mass = hydro_zones.getRawPtr<double>(FID_ZM);
    const double* zone_work_rate = hydro_zones.getRawPtr<double>(FID_ZWR);

    assert(task->regions[EIGHT].privilege_fields.size() == 1);
    LogicalStructured mesh_points(ctx, runtime, regions[EIGHT]);
    const double2* pt_x_pred = mesh_points.getRawPtr<double2>(FID_PXP);

    assert(task->regions[NINE].privilege_fields.size() == 1);
    LogicalStructured side_chunks(ctx, runtime, regions[NINE]);
    const int* side_chunks_CRS = side_chunks.getRawPtr<int>(FID_SIDE_CHUNKS_CRS);

    assert(task->regions[TEN].privilege_fields.size() == 2);
    LogicalStructured mesh_write_edges(ctx, runtime, regions[TEN]);
    double* edge_len = mesh_write_edges.getRawPtr<double>(FID_E_DBL_TEMP);
    double2* edge_x_pred = mesh_write_edges.getRawPtr<double2>(FID_E_DBL2_TEMP);

    assert(task->regions[ONE].privilege_fields.size() == 5);
    LogicalStructured mesh_write_zones(ctx, runtime, regions[ONE]);
    double* zone_dl = mesh_write_zones.getRawPtr<double>(FID_ZDL);
    double* zone_vol0 = mesh_write_zones.getRawPtr<double>(FID_ZVOL0);
    double2* zone_x_pred = mesh_write_zones.getRawPtr<double2>(FID_Z_DBL2_TEMP);
    double* zone_vol_pred = mesh_write_zones.getRawPtr<double>(FID_Z_DBL_TEMP1);
    double* zone_area_pred = mesh_write_zones.getRawPtr<double>(FID_Z_DBL_TEMP2);

    assert(task->regions[SEVEN].privilege_fields.size() == 3);
    LogicalStructured hydro_write_zones(ctx, runtime, regions[SEVEN]);
    double* zone_dvel = hydro_write_zones.getRawPtr<double>(FID_ZDU);
    double* zone_sound_speed = hydro_write_zones.getRawPtr<double>(FID_ZSS);
    double* zone_pressure = hydro_write_zones.getRawPtr<double>(FID_ZP);

    assert(task->regions[TWO].privilege_fields.size() == 6);
    LogicalStructured write_sides_and_corners(ctx, runtime, regions[TWO]);
    double2* crnr_force_tot = write_sides_and_corners.getRawPtr<double2>(FID_CFTOT);
    double2* side_force_visc = write_sides_and_corners.getRawPtr<double2>(FID_SFQ);
    double2* side_force_tts = write_sides_and_corners.getRawPtr<double2>(FID_SFT);
    double2* side_force_pres = write_sides_and_corners.getRawPtr<double2>(FID_SFP);
    double* crnr_weighted_mass = write_sides_and_corners.getRawPtr<double>(FID_CMASWT);

    double* side_area_pred = write_sides_and_corners.getRawPtr<double>(FID_S_DBL_TEMP);

    assert(task->futures.size() == 1);
    Future f1 = task->futures[0];
    TimeStep time_step = f1.get_result<TimeStep>();

    for (int side_chunk = 0; side_chunk < args.num_side_chunks; ++side_chunk) {
        int sfirst = side_chunks_CRS[side_chunk];
        int slast = side_chunks_CRS[side_chunk+1];
        int zfirst = LocalMesh::side_zone_chunks_first(side_chunk, map_side2zone, side_chunks_CRS);
        int zlast = LocalMesh::side_zone_chunks_last(side_chunk, map_side2zone, side_chunks_CRS);

        // save off zone variable values from previous cycle
        std::copy(&zone_vol[zfirst], &zone_vol[zlast], &zone_vol0[zfirst]);

        // 1a. compute new mesh geometry
        LocalMesh::calcCtrs(sfirst, slast, pt_x_pred,
                map_side2zone, args.num_sides, args.num_zones, map_side2pt1, map_side2pt2, map_side2edge, zone_pts_ptr,
                edge_x_pred, zone_x_pred);

        LocalMesh::calcVols(sfirst, slast, pt_x_pred, zone_x_pred,
                map_side2zone, args.num_sides, args.num_zones, map_side2pt1, map_side2pt2, zone_pts_ptr,
                side_area_pred, nullptr, zone_area_pred, zone_vol_pred);

        LocalMesh::calcEdgeLen(sfirst, slast, map_side2pt1, map_side2pt2, map_side2edge,
                map_side2zone, zone_pts_ptr, pt_x_pred,
                edge_len);

        LocalMesh::calcCharacteristicLen(sfirst, slast, map_side2zone, map_side2edge,
                zone_pts_ptr, side_area_pred, edge_len, args.num_sides, args.num_zones,
                zone_dl);

        // 3. compute material state (half-advanced)
        PolyGascalcStateAtHalf(zone_rho, zone_vol_pred, zone_vol0, zone_energy_density, zone_work_rate, zone_mass, time_step.dt,
                zone_pressure, zone_sound_speed, zfirst, zlast, args.gamma,
                        args.ssmin);

        // 2. compute point masses
        // 4. compute forces
        Force(side_force_visc, sfirst, slast,
                args.num_sides, args.num_zones, pt_vel, edge_x_pred,
                zone_x_pred, edge_len, map_side2zone, map_side2pt1, map_side2pt2,
                zone_pts_ptr, map_side2edge, pt_x_pred,
                zone_sound_speed, args.qgamma,
                args.q1, args.q2,
                map_side2zone[sfirst],
                (slast < args.num_sides ? map_side2zone[slast] : args.num_zones),
                zone_dvel,
                zone_area_pred, side_area_pred, side_mass_frac, side_force_tts,
                args.ssmin, args.alpha,
                crnr_weighted_mass,
                zone_vol_pred, zone_mass,
                zone_pressure, side_force_pres);

        Hydro::sumCrnrForce(side_force_pres, side_force_visc, side_force_tts,
                map_side2zone, zone_pts_ptr, sfirst, slast,
                crnr_force_tot);
    } // side chunk
}

