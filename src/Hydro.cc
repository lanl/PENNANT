/*
 * Hydro.cc
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Hydro.hh"

#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

#include "CorrectorTask.hh"
#include "HydroBC.hh"
#include "LocalMesh.hh"
#include "Memory.hh"
#include "PredictorPointTask.hh"
#include "PredictorTask.hh"


using namespace std;

Hydro::Hydro(const InputParameters& params, LocalMesh* m,
        DynamicCollective add_reduction,
        DynamicCollective min_reduction,
        Context ctx, HighLevelRuntime* rt) :
		mesh(m),
		cfl(params.directs.cfl),
		cflv(params.directs.cflv),
		rho_init(params.directs.rho_init),
		energy_init(params.directs.energy_init),
		rho_init_sub(params.directs.rho_init_sub),
		energy_init_sub(params.directs.energy_init_sub),
		vel_init_radial(params.directs.vel_init_radial),
		bcx(params.bcx),
		bcy(params.bcy),
        add_reduction(add_reduction),
        min_reduction(min_reduction),
		ctx(ctx),
		runtime(rt),
        zones(ctx, rt),
        sides_and_corners(ctx, rt),
        edges(ctx, rt),
        points(ctx, rt),
        bcx_chunks(ctx, rt),
        bcy_chunks(ctx, rt),
        params(params),
		my_color(params.directs.task_id)
{
    init();
}


void Hydro::init() {

    const int numpch = mesh->num_pt_chunks;
    const int numzch = mesh->num_zone_chunks;
    const int nump = mesh->num_pts;
    const int numz = mesh->num_zones;
    const int nums = mesh->num_sides;

    const double2* zx = mesh->zones.getRawPtr<double2>(FID_ZX);
    const double* zvol = mesh->zones.getRawPtr<double>(FID_ZVOL);

    const int* zone_chunks_CRS = mesh->zone_chunks.getRawPtr<int>(FID_ZONE_CHUNKS_CRS);
    const int* pt_chunks_CRS = mesh->point_chunks.getRawPtr<int>(FID_POINT_CHUNKS_CRS);

    // allocate arrays
    allocateFields();

    points.allocate(nump);
    double2* pt_vel = points.getRawPtr<double2>(FID_PU);

    sides_and_corners.allocate(nums);

    zones.allocate(numz);
    double* zone_rho = zones.getRawPtr<double>(FID_ZR);
    double* zone_energy_density = zones.getRawPtr<double>(FID_ZE);
    double* zone_mass = zones.getRawPtr<double>(FID_ZM);
    double* zone_energy_tot = zones.getRawPtr<double>(FID_ZETOT);
    double* zone_work_rate = zones.getRawPtr<double>(FID_ZWR);

    // initialize hydro vars
    for (int zch = 0; zch < numzch; ++zch) {
        int zfirst = zone_chunks_CRS[zch];
        int zlast = zone_chunks_CRS[zch+1];

        fill(&zone_rho[zfirst], &zone_rho[zlast], rho_init);
        fill(&zone_energy_density[zfirst], &zone_energy_density[zlast], energy_init);
        fill(&zone_work_rate[zfirst], &zone_work_rate[zlast], 0.);

        const double& subrgn_xmin = mesh->subregion_xmin;
        const double& subrgn_xmax = mesh->subregion_xmax;
        const double& subrgn_ymin = mesh->subregion_ymin;
        const double& subrgn_ymax = mesh->subregion_ymax;
        if (subrgn_xmin != std::numeric_limits<double>::max()) {
            const double eps = 1.e-12;
            #pragma ivdep
            for (int z = zfirst; z < zlast; ++z) {
                if (zx[z].x > (subrgn_xmin - eps) &&
                    zx[z].x < (subrgn_xmax + eps) &&
                    zx[z].y > (subrgn_ymin - eps) &&
                    zx[z].y < (subrgn_ymax + eps)) {
                    zone_rho[z]  = rho_init_sub;
                    zone_energy_density[z] = energy_init_sub;
                }
            }
        }

        #pragma ivdep
        for (int z = zfirst; z < zlast; ++z) {
        		zone_mass[z] = zone_rho[z] * zvol[z];
        		zone_energy_tot[z] = zone_energy_density[z] * zone_mass[z];
        }
    }  // for sch

    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = pt_chunks_CRS[pch];
        int plast = pt_chunks_CRS[pch+1];
        if (vel_init_radial != 0.)
            initRadialVel(vel_init_radial, pfirst, plast, pt_vel);
        else
            fill(&pt_vel[pfirst], &pt_vel[plast], double2(0., 0.));
    }  // for pch

    points.unMapPRegion();
    sides_and_corners.unMapPRegion();
    zones.unMapPRegion();
    mesh->zones.unMapPRegion();
    mesh->zone_chunks.unMapPRegion();

    args.num_point_chunks = mesh->num_pt_chunks;
    double2* pt_x = mesh->points.getRawPtr<double2>(FID_PX);
    int count = 0;
    std::vector<int> bcx_point_chunk_CRS_concatenated;
    for (int i = 0; i < bcx.size(); ++i) {
        args.boundary_conditions_x.push_back(LocalMesh::getXPlane(bcx[i], mesh->num_pts, pt_x));
        std::vector<int> pchb_CRS;
        LocalMesh::getPlaneChunks(args.boundary_conditions_x[i], pt_chunks_CRS, numpch, pchb_CRS);
        args.bcx_point_chunk_CRS_offsets.push_back(count);
        count += pchb_CRS.size();
        for (int j = 0; j < pchb_CRS.size(); ++j)
            bcx_point_chunk_CRS_concatenated.push_back(pchb_CRS[j]);
    }
    bcx_chunks.allocate(bcx_point_chunk_CRS_concatenated.size());
    int* bcx_chunks_CRS = bcx_chunks.getRawPtr<int>(FID_BCX_CHUNKS_CRS);
    std::copy(bcx_point_chunk_CRS_concatenated.begin(), bcx_point_chunk_CRS_concatenated.end(),
            bcx_chunks_CRS);
    bcx_chunks.unMapPRegion();

    count = 0;
    std::vector<int> bcy_point_chunk_CRS_concatenated;
    for (int i = 0; i < bcy.size(); ++i){
        args.boundary_conditions_y.push_back(LocalMesh::getYPlane(bcy[i], mesh->num_pts, pt_x));
        std::vector<int> pchb_CRS;
        LocalMesh::getPlaneChunks(args.boundary_conditions_y[i], pt_chunks_CRS, numpch, pchb_CRS);
        args.bcy_point_chunk_CRS_offsets.push_back(count);
        count += pchb_CRS.size();
        for (int j = 0; j < pchb_CRS.size(); ++j)
            bcy_point_chunk_CRS_concatenated.push_back(pchb_CRS[j]);
    }
    bcy_chunks.allocate(bcy_point_chunk_CRS_concatenated.size());
    int* bcy_chunks_CRS = bcy_chunks.getRawPtr<int>(FID_BCY_CHUNKS_CRS);
    std::copy(bcy_point_chunk_CRS_concatenated.begin(), bcy_point_chunk_CRS_concatenated.end(),
            bcy_chunks_CRS);
    bcy_chunks.unMapPRegion();
    mesh->points.unMapPRegion();
    mesh->point_chunks.unMapPRegion();


    args.min_reduction = min_reduction;
    args.cfl = cfl;
    args.cflv = cflv;
    args.num_points = mesh->num_pts;
    args.num_sides = mesh->num_sides;
    args.num_zones = mesh->num_zones;
    args.num_edges = mesh->num_edges;
    args.num_zone_chunks = mesh->num_zone_chunks;
    args.num_side_chunks = mesh->num_side_chunks;
    args.meshtype = params.meshtype;
    args.nzones_x = params.directs.nzones_x;
    args.nzones_y = params.directs.nzones_y;
    args.num_subregions = params.directs.ntasks;
    args.my_color = my_color;
    args.qgamma = params.directs.qgamma;
    args.q1 = params.directs.q1;
    args.q2 = params.directs.q2;
    args.ssmin = params.directs.ssmin;
    args.alpha = params.directs.alpha;
    args.gamma = params.directs.gamma;

    serial.archive(&args);
}


void Hydro::allocateFields()
{
    points.addField<double2>(FID_PU);
    points.addField<double2>(FID_PU0);
    sides_and_corners.addField<double>(FID_CMASWT);
    sides_and_corners.addField<double2>(FID_SFP);
    sides_and_corners.addField<double2>(FID_SFQ);
    sides_and_corners.addField<double2>(FID_SFT);
    sides_and_corners.addField<double2>(FID_CFTOT);
    sides_and_corners.addField<double>(FID_S_DBL_TEMP);
    zones.addField<double>(FID_ZR);
    zones.addField<double>(FID_ZE);
    zones.addField<double>(FID_ZP);
    zones.addField<double>(FID_ZM);
    zones.addField<double>(FID_ZETOT);
    zones.addField<double>(FID_ZWR);
    zones.addField<double>(FID_ZSS);
    zones.addField<double>(FID_ZDU);
    bcx_chunks.addField<int>(FID_BCX_CHUNKS_CRS);
    bcy_chunks.addField<int>(FID_BCY_CHUNKS_CRS);
}


void Hydro::initRadialVel(
        const double vel,
        const int pfirst,
        const int plast,
        double2* pt_vel)
{
    const double eps = 1.e-12;

    double2* pt_x = mesh->points.getRawPtr<double2>(FID_PX);

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        double pmag = length(pt_x[p]);
        if (pmag > eps)
            pt_vel[p] = vel * pt_x[p] / pmag;
        else
            pt_vel[p] = double2(0., 0.);
    }
    mesh->points.unMapPRegion();
}


Future Hydro::doCycle(Future future_step)
{
    PredictorPointTask predictor_point_launcher(
            mesh->points.getLRegion(),
            mesh->point_chunks.getLRegion(),
            points.getLRegion(),
            serial.getBitStream(), serial.getBitStreamSize());
    predictor_point_launcher.add_future(future_step);
    runtime->execute_task(ctx, predictor_point_launcher);

    PredictorTask predictor_launcher(mesh->zones.getLRegion(),
            mesh->sides.getLRegion(),
            mesh->zone_pts.getLRegion(),
            mesh->points.getLRegion(),
            mesh->side_chunks.getLRegion(),
            mesh->edges.getLRegion(),
            zones.getLRegion(),
            sides_and_corners.getLRegion(),
            points.getLRegion(),
            serial.getBitStream(), serial.getBitStreamSize());
    predictor_launcher.add_future(future_step);
    runtime->execute_task(ctx, predictor_launcher);

    // sum corner masses, forces to points
    mesh->sumCornersToPoints(sides_and_corners, serial);

    CorrectorTask corrector_launcher(mesh->zones.getLRegion(),
            mesh->sides.getLRegion(),
            mesh->zone_pts.getLRegion(),
            mesh->points.getLRegion(),
            mesh->edges.getLRegion(),
            mesh->local_points_by_gid.getLRegion(),
            mesh->point_chunks.getLRegion(),
            mesh->side_chunks.getLRegion(),
            mesh->zone_chunks.getLRegion(),
            zones.getLRegion(),
            sides_and_corners.getLRegion(),
            points.getLRegion(),
            bcx_chunks.getLRegion(),
            bcy_chunks.getLRegion(),
            serial.getBitStream(), serial.getBitStreamSize());
    corrector_launcher.add_future(future_step);

    Future future = runtime->execute_task(ctx, corrector_launcher);
    Future min_future = Parallel::globalMin(future, min_reduction, runtime, ctx);

    // Future dt = calcGlobalDt(min_future);
    return min_future;
}


/*static*/
void Hydro::advPosHalf(
        const double dt,
        const int pfirst,
        const int plast,
        const double2* pt_x0,
        const double2* pt_vel0,
        double2* pt_x_pred)
{
    double dth = 0.5 * dt;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        pt_x_pred[p] = pt_x0[p] + pt_vel0[p] * dth;
    }
}


/*static*/
void Hydro::advPosFull(
        const double dt,
        const double2* pt_vel0,
        const double2* pt_accel,
        const double2* pt_x0,
        double2* pt_vel,
        double2* pt_x,
        const int pfirst,
        const int plast) {

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        pt_vel[p] = pt_vel0[p] + pt_accel[p] * dt;
        pt_x[p] = pt_x0[p] + 0.5 * (pt_vel[p] + pt_vel0[p]) * dt;
    }

}


/*static*/
void Hydro::calcCrnrMass(
        const int sfirst,
        const int slast,
        const double* zone_area_pred,
        const double* side_mass_frac,
        const int* map_side2zone,
        const int* zone_pts_ptr,
        const double* zone_rho_pred,
        double* crnr_weighted_mass)
{
    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int s3 = LocalMesh::mapSideToSidePrev(s, map_side2zone, zone_pts_ptr);
        int z = map_side2zone[s];

        double m = zone_rho_pred[z] * zone_area_pred[z] * 0.5 * (side_mass_frac[s] + side_mass_frac[s3]);
        crnr_weighted_mass[s] = m;
    }
}


/*static*/
void Hydro::sumCrnrForce(
        const double2* side_force_pres,
        const double2* side_force_visc,
        const double2* side_force_tts,
        const int* map_side2zone,
        const int* zone_pts_ptr,
        const int sfirst,
        const int slast,
        double2* crnr_force_tot) {

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int s3 = LocalMesh::mapSideToSidePrev(s, map_side2zone, zone_pts_ptr);

        double2 f = (side_force_pres[s] + side_force_visc[s] + side_force_tts[s]) -
                    (side_force_pres[s3] + side_force_visc[s3] + side_force_tts[s3]);
        crnr_force_tot[s] = f;
    }
}


/*static*/
void Hydro::calcAccel(
        const ptr_t* pt_local2globalID,
        const Double2SOAAccessor pf,
        const DoubleSOAAccessor pmass,
        double2* pt_accel,
        const int pfirst,
        const int plast) {

    const double fuzz = 1.e-99;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        ptr_t pt_ptr = pt_local2globalID[p];
        pt_accel[p] = pf.read(pt_ptr) / max(pmass.read(pt_ptr), fuzz);
    }

}


/*static*/
void Hydro::calcRho(
        const double* zvol,
        const double* zm,
        double* zr,
        const int zfirst,
        const int zlast)
{
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        zr[z] = zm[z] / zvol[z];
    }
}


/*static*/
void Hydro::calcWork(
        const double dt,
        const int* map_side2pt1,
        const int* map_side2pt2,
        const int* map_side2zone,
        const int* zone_pts_ptr,
        const double2* side_force_pres,
        const double2* side_force_visc,
        const double2* pt_vel,
        const double2* pt_vel0,
        const double2* pt_x_pred,
        double* zone_energy_tot,
        double* zone_work,
        const int side_first,
        const int side_last) {
    // Compute the work done by finding, for each element/node pair,
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const double dth = 0.5 * dt;

    for (int side = side_first; side < side_last; ++side) {
        int p1 = map_side2pt1[side];
        int p2 = map_side2pt2[side];
        int z = map_side2zone[side];

        double2 sftot = side_force_pres[side] + side_force_visc[side];
        double sd1 = dot( sftot, (pt_vel0[p1] + pt_vel[p1]));
        double sd2 = dot(-sftot, (pt_vel0[p2] + pt_vel[p2]));
        double dwork = -dth * (sd1 * pt_x_pred[p1].x + sd2 * pt_x_pred[p2].x);

        zone_energy_tot[z] += dwork;
        zone_work[z] += dwork;

    }

}


/*static*/
void Hydro::calcWorkRate(
        const double dt,
        const double* zone_vol,
        const double* zone_vol0,
        const double* zone_work,
        const double* zone_pressure,
        double* zone_work_rate,
        const int zfirst,
        const int zlast) {
    double dtinv = 1. / dt;
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        double dvol = zone_vol[z] - zone_vol0[z];
        zone_work_rate[z] = (zone_work[z] + zone_pressure[z] * dvol) * dtinv;
    }

}


/*static*/
void Hydro::calcEnergy(
        const double* zone_energy_tot,
        const double* zone_mass,
        double* zone_energy_density,
        const int zfirst,
        const int zlast)
{
    const double fuzz = 1.e-99;
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        zone_energy_density[z] = zone_energy_tot[z] / (zone_mass[z] + fuzz);
    }
}


void Hydro::sumEnergy(
        const double* zetot,
        const double* zarea,
        const double* zvol,
        const double* zm,
        const double* side_mass_frac,
        const double2* px,
        const double2* pu,
        const int* map_side2pt1,
        const int* map_side2zone,
        const int* zone_pts_ptr,
        double& ei,
        double& ek,
        const int zfirst,
        const int zlast,
        const int sfirst,
        const int slast) {

    // compute internal energy
    double sumi = 0.; 
    for (int z = zfirst; z < zlast; ++z) {
        sumi += zetot[z];
    }
    // multiply by 2\pi for cylindrical geometry
    ei += sumi * 2 * M_PI;

    // compute kinetic energy
    // in each individual zone:
    // zone ke = zone mass * (volume-weighted average of .5 * u ^ 2)
    //         = zm sum(c in z) [cvol / zvol * .5 * u ^ 2]
    //         = sum(c in z) [zm * cvol / zvol * .5 * u ^ 2]
    double sumk = 0.; 
    for (int s = sfirst; s < slast; ++s) {
        int s3 = LocalMesh::mapSideToSidePrev(s, map_side2zone, zone_pts_ptr);
        int p1 = map_side2pt1[s];
        int z = map_side2zone[s];

        double cvol = zarea[z] * px[p1].x * 0.5 * (side_mass_frac[s] + side_mass_frac[s3]);
        double cke = zm[z] * cvol / zvol[z] * 0.5 * length2(pu[p1]);
        sumk += cke;
    }
    // multiply by 2\pi for cylindrical geometry
    ek += sumk * 2 * M_PI;
}


/*static*/
void Hydro::calcDtCourant(
        double& dtrec,
        char* msgdtrec,
        const int zfirst,
        const int zlast,
        const double* zdl,
        const double* zone_dvel,
        const double* zone_sound_speed,
        const double cfl)
{
    const double fuzz = 1.e-99;
    double dtnew = 1.e99;
    int zmin = -1;
    for (int z = zfirst; z < zlast; ++z) {
        double cdu = std::max(zone_dvel[z], std::max(zone_sound_speed[z], fuzz));
        double zdthyd = zdl[z] * cfl / cdu;
        zmin = (zdthyd < dtnew ? z : zmin);
        dtnew = (zdthyd < dtnew ? zdthyd : dtnew);
    }

    if (dtnew < dtrec) {
        dtrec = dtnew;
        snprintf(msgdtrec, 80, "Hydro Courant limit for z = %d", zmin);
    }

}


/*static*/
void Hydro::calcDtVolume(
        const double dtlast,
        double& dtrec,
        char* msgdtrec,
        const int zfirst,
        const int zlast,
        const double* zvol,
        const double* zvol0,
        const double cflv)
{
    double dvovmax = 1.e-99;
    int zmax = -1;
    for (int z = zfirst; z < zlast; ++z) {
        double zdvov = std::abs((zvol[z] - zvol0[z]) / zvol0[z]);
        zmax = (zdvov > dvovmax ? z : zmax);
        dvovmax = (zdvov > dvovmax ? zdvov : dvovmax);
    }
    double dtnew = dtlast * cflv / dvovmax;
    if (dtnew < dtrec) {
        dtrec = dtnew;
        snprintf(msgdtrec, 80, "Hydro dV/V limit for z = %d", zmax);
    }

}


/*static*/
void Hydro::calcDtHydro(
        const double dtlast,
        const int zfirst,
        const int zlast,
        const double* zone_dl,
        const double* zone_dvel,
        const double* zone_sound_speed,
        const double cfl,
        const double* zone_vol,
        const double* zone_vol0,
        const double cflv,
        TimeStep& recommend)
{
    double dtchunk = 1.e99;
    char msgdtchunk[80];

    calcDtCourant(dtchunk, msgdtchunk, zfirst, zlast, zone_dl,
            zone_dvel, zone_sound_speed, cfl);
    calcDtVolume(dtlast, dtchunk, msgdtchunk, zfirst, zlast,
            zone_vol, zone_vol0, cflv);
    if (dtchunk < recommend.dt) {
        {
            // redundant test needed to avoid race condition
            if (dtchunk < recommend.dt) {
                recommend.dt = dtchunk;
                strncpy(recommend.message, msgdtchunk, 80);
            }
        }
    }

}


void Hydro::writeEnergyCheck() {


    const int* side_chunks_CRS = mesh->side_chunks.getRawPtr<int>(FID_SIDE_CHUNKS_CRS);
    const double2* pt_x = mesh->points.getRawPtr<double2>(FID_PX);
    const int* map_side2zone = mesh->sides.getRawPtr<int>(FID_SMAP_SIDE_TO_ZONE);
    const double* side_mass_frac = mesh->sides.getRawPtr<double>(FID_SMF);
    const int* map_side2pt1 = mesh->sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT1);
    const double* zone_area = mesh->zones.getRawPtr<double>(FID_ZAREA);
    const double* zone_vol = mesh->zones.getRawPtr<double>(FID_ZVOL);
    const int* zone_pts_ptr = mesh->zone_pts.getRawPtr<int>(FID_ZONE_PTS_PTR);
    const double* zone_energy_tot = zones.getRawPtr<double>(FID_ZETOT);
    const double* zone_mass = zones.getRawPtr<double>(FID_ZM);
    const double2* pt_vel = points.getRawPtr<double2>(FID_PU);

    double ei = 0.;
    double ek = 0.;
    for (int sch = 0; sch < mesh->num_side_chunks; ++sch) {
        int sfirst = side_chunks_CRS[sch];
        int slast = side_chunks_CRS[sch+1];
        int zfirst = LocalMesh::side_zone_chunks_first(sch, map_side2zone, side_chunks_CRS);
        int zlast = LocalMesh::side_zone_chunks_last(sch, map_side2zone, side_chunks_CRS);

        double eichunk = 0.;
        double ekchunk = 0.;
        sumEnergy(zone_energy_tot, zone_area, zone_vol, zone_mass, side_mass_frac,
                pt_x, pt_vel, map_side2pt1, map_side2zone, zone_pts_ptr,
                eichunk, ekchunk,
                zfirst, zlast, sfirst, slast);
        {
            ei += eichunk;
            ek += ekchunk;
        }
    }
    points.unMapPRegion();
    zones.unMapPRegion();
    mesh->side_chunks.unMapPRegion();
    mesh->points.unMapPRegion();
    mesh->sides.unMapPRegion();
    mesh->zones.unMapPRegion();
    mesh->zone_pts.unMapPRegion();

    Future future_sum = Parallel::globalSum(ei, add_reduction, runtime, ctx);
    ei = future_sum.get_result<double>();

    future_sum = Parallel::globalSum(ek, add_reduction, runtime, ctx);
    ek = future_sum.get_result<double>();

    if (my_color == 0) {
        cout << scientific << setprecision(6);
        cout << "Energy check:  "
             << "total energy  = " << setw(14) << ei + ek << endl;
        cout << "(internal = " << setw(14) << ei
             << ", kinetic = " << setw(14) << ek << ")" << endl;
    }

}


void Hydro::copyZonesToLegion(LogicalUnstructured& global_zones)
{
    IndexSpace ispace_zones = global_zones.getISpace();
    DoubleSOAAccessor rho_acc = global_zones.getRegionSOAAccessor<double>(FID_ZR);
    DoubleSOAAccessor energy_density_acc = global_zones.getRegionSOAAccessor<double>(FID_ZE);
    DoubleSOAAccessor pressure_acc = global_zones.getRegionSOAAccessor<double>(FID_ZP);

    const double* zone_rho = zones.getRawPtr<double>(FID_ZR);
    const double* zone_energy_density = zones.getRawPtr<double>(FID_ZE);
    const double* zone_pressure_ = zones.getRawPtr<double>(FID_ZP);

    IndexIterator zone_itr(runtime,ctx, ispace_zones);
    int z = 0;
    while (zone_itr.has_next()) {
        ptr_t zone_ptr = zone_itr.next();
        rho_acc.write(zone_ptr, zone_rho[z]);
        energy_density_acc.write(zone_ptr, zone_energy_density[z]);
        pressure_acc.write(zone_ptr, zone_pressure_[z]);
        z++;
    }

    global_zones.unMapPRegion();
    zones.unMapPRegion();

    assert(z == mesh->num_zones);
}
