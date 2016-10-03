/*
 * HydroTask2.cc
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

#include "HydroTask2.hh"

#include<algorithm>
#include<cmath>

#include "LogicalStructured.hh"


HydroTask2::HydroTask2(LogicalRegion mesh_zones,
        LogicalRegion hydro_zones,
		void *args, const size_t &size)
	 : TaskLauncher(HydroTask2::TASK_ID, TaskArgument(args, size))
{
    add_region_requirement(RegionRequirement(mesh_zones, READ_ONLY, EXCLUSIVE, mesh_zones));
    add_field(0/*idx*/, FID_ZDL);
    add_field(0/*idx*/, FID_ZVOL);
    add_field(0/*idx*/, FID_ZVOL0);
    add_region_requirement(RegionRequirement(hydro_zones, READ_ONLY, EXCLUSIVE, hydro_zones));
    add_field(1/*idx*/, FID_ZDU);
    add_field(1/*idx*/, FID_ZSS);
}

/*static*/ const char * const HydroTask2::TASK_NAME = "HydroTask2";

static void calcDtCourant(
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


static void calcDtVolume(
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


static void calcDtHydro(
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


/*static*/
TimeStep HydroTask2::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() == 2);
	assert(task->regions.size() == 2);
    assert(task->regions[0].privilege_fields.size() == 3);
    assert(task->regions[1].privilege_fields.size() == 2);

    LogicalStructured mesh_zones(ctx, runtime, regions[0]);
    double* zone_dl = mesh_zones.getRawPtr<double>(FID_ZDL);
    double* zone_vol = mesh_zones.getRawPtr<double>(FID_ZVOL);
    double* zone_vol0 = mesh_zones.getRawPtr<double>(FID_ZVOL0);

    LogicalStructured hydro_zones(ctx, runtime, regions[1]);
    double* zone_dvel = hydro_zones.getRawPtr<double>(FID_ZDU);
    double* zone_sound_speed = hydro_zones.getRawPtr<double>(FID_ZSS);

    HydroTask2Args args;
    HydroTask2ArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    TimeStep recommend;
    recommend.dt = 1.e99;
    strcpy(recommend.message, "Hydro default");

    for (int zch = 0; zch < args.num_zone_chunks; ++zch) {
        int zfirst = args.zone_chunk_CRS[zch];
        int zlast = args.zone_chunk_CRS[zch+1];
        calcDtHydro(args.dtlast, zfirst, zlast, zone_dl,
                zone_dvel, zone_sound_speed, args.cfl,
                zone_vol, zone_vol0, args.cflv, recommend);
    }

    return recommend;
}

