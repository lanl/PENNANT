/*
 * Driver.cc
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Driver.hh"

#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <iomanip>

#include "Hydro.hh"
#include "LocalMesh.hh"

using namespace std;

DriverTask::DriverTask(int my_color,
		LogicalRegion my_zones,
		LogicalRegion all_zones,
		std::vector<LogicalRegion> halo_pts,
		void *args, const size_t &size)
	 : TaskLauncher(DriverTask::TASK_ID, TaskArgument(args, size))
{
	add_region_requirement(RegionRequirement(my_zones, WRITE_DISCARD, EXCLUSIVE, all_zones));
	add_field(0/*idx*/, FID_ZR);
	add_field(0/*idx*/, FID_ZE);
	add_field(0/*idx*/, FID_ZP);
	for (int i=0; i < halo_pts.size(); ++i) {
		add_region_requirement(RegionRequirement(halo_pts[i], READ_WRITE, SIMULTANEOUS, halo_pts[i]));
		if (i != 0)
			region_requirements[1+i].add_flags(NO_ACCESS_FLAG);
		add_field(1+i, FID_GHOST_PF);
		add_field(1+i, FID_GHOST_PMASWT);
	}
}

/*static*/ const char * const DriverTask::TASK_NAME = "DriverTask";

/*static*/
RunStat DriverTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	assert(regions.size() > 1);
	assert(task->regions.size() > 1);
	assert(task->regions[0].privilege_fields.size() == 3);
	assert(task->regions[1].privilege_fields.size() == 2);

	runtime->unmap_all_regions(ctx);

    std::vector<LogicalUnstructured> halos_points;
    std::vector<PhysicalRegion> pregions_halos;
	for (int i = 1; i < task->regions.size(); i++) {
	    halos_points.push_back(LogicalUnstructured(ctx, runtime, task->regions[i].region));
	    pregions_halos.push_back(regions[i]);
	}

    SPMDArgs args;
    SPMDArgsSerializer args_serializer;
    args_serializer.setBitStream(task->args);
    args_serializer.restore(&args);

    InputParameters params;
    params.directs = args.direct_input_params;
    params.meshtype = args.meshtype;
    params.probname = args.probname;
    params.bcx = args.bcx;
	params.bcy = args.bcy;

    IndexSpace ghost_is = task->regions[1].region.get_index_space();
    IndexPartition ghost_ip = runtime->get_parent_index_partition(ctx, ghost_is);
    IndexSpace points = runtime->get_parent_index_space(ctx, ghost_ip);
    Driver drv(params, args.add_reduction, args.add_int64_reduction, args.min_reduction,
            args.pbarrier_as_master, args.masters_pbarriers,
            regions[0], points, halos_points, pregions_halos,
            ctx, runtime);

    RunStat value=drv.run();
    return value;
}

Driver::Driver(const InputParameters& params,
        DynamicCollective add_reduct,
        DynamicCollective add_int64_reduct,
		DynamicCollective min_reduct,
        PhaseBarrier pbarrier_as_master,
        std::vector<PhaseBarrier> masters_pbarriers,
        const PhysicalRegion& zones,
        IndexSpace pts,
        std::vector<LogicalUnstructured>& halos_points,
        std::vector<PhysicalRegion>& pregions_halos,
        Context ctx, HighLevelRuntime* rt)
        : probname(params.probname),
		  tstop(params.directs.tstop),
		  cstop(params.directs.cstop),
		  dtmax(params.directs.dtmax),
		  dtinit(params.directs.dtinit),
		  dtfac(params.directs.dtfac),
		  dtreport(params.directs.dtreport),
          add_reduction(add_reduct),
          add_int64_reduction(add_int64_reduct),
		  min_reduction(min_reduct),
		  ctx(ctx),
		  runtime(rt),
		  my_color(params.directs.task_id),
          global_zones(ctx, runtime, zones)

{
    global_zones.unMapPRegion();
    mesh = new LocalMesh(params, pts, halos_points, pregions_halos,
            pbarrier_as_master, masters_pbarriers,
            add_int64_reduction, ctx, runtime);
    hydro = new Hydro(params, mesh, add_reduction, ctx, runtime);
}


RunStat Driver::run() {
    run_stat.time = 0.0;
    run_stat.cycle = 0;

    // do energy check
    hydro->writeEnergyCheck();

    double tbegin = std::numeric_limits<double>::max();
    double tlast = std::numeric_limits<double>::min();
    if (my_color == 0) {
        // get starting timestamp
        struct timeval sbegin;
        gettimeofday(&sbegin, NULL);
        tbegin = sbegin.tv_sec + sbegin.tv_usec * 1.e-6;
        tlast = tbegin;
    }

    // main event loop
    while (run_stat.cycle < cstop && run_stat.time < tstop) {

    	run_stat.cycle += 1;

        // get timestep
        calcGlobalDt();

        // begin hydro cycle
        dt_hydro = hydro->doCycle(dt);

        run_stat.time += dt;

        if (my_color == 0 &&
                (run_stat.cycle == 1 || run_stat.cycle % dtreport == 0)) {
            struct timeval scurr;
            gettimeofday(&scurr, NULL);
            double tcurr = scurr.tv_sec + scurr.tv_usec * 1.e-6;
            double tdiff = tcurr - tlast;

            cout << scientific << setprecision(5);
            cout << "End cycle " << setw(6) << run_stat.cycle
                 << ", time = " << setw(11) << run_stat.time
                 << ", dt = " << setw(11) << dt
                 << ", wall = " << setw(11) << tdiff << endl;
            cout << "dt limiter: " << msgdt << endl;

            tlast = tcurr;
        } // if mype_...

    } // while cycle...

    // copy Hydro zone data to legion global regions
    hydro->copyZonesToLegion(global_zones);

    if (my_color == 0) {

        // get stopping timestamp
        struct timeval send;
        gettimeofday(&send, NULL);
        double tend = send.tv_sec + send.tv_usec * 1.e-6;
        double runtime = tend - tbegin;

        // write end message
        cout << endl;
        cout << "Run complete" << endl;
        cout << scientific << setprecision(6);
        cout << "cycle = " << setw(6) << run_stat.cycle
             << ",         cstop = " << setw(6) << cstop << endl;
        cout << "time  = " << setw(14) << run_stat.time
             << ", tstop = " << setw(14) << tstop << endl;

        cout << endl;
        cout << "************************************" << endl;
        cout << "hydro cycle run time= " << setw(14) << runtime << endl;
        cout << "************************************" << endl;

    } // if mype_

    // do energy check
    hydro->writeEnergyCheck();

    return run_stat;
}

void Driver::calcGlobalDt() {

    // Save timestep from last cycle
    dtlast = dt;
    msgdtlast = msgdt;

    // Compute timestep for this cycle
    dt = dtmax;
    msgdt = "Global maximum (dtmax)";

    if (run_stat.cycle == 1) {
        // compare to initial timestep
        if (dtinit < dt) {
            dt = dtinit;
            msgdt = "Initial timestep";
        }
    } else {
        // compare to factor * previous timestep
        double dtrecover = dtfac * dtlast;
        if (dtrecover < dt) {
            dt = dtrecover;
            if (msgdtlast.substr(0, 8) == "Recovery")
                msgdt = msgdtlast;
            else
                msgdt = "Recovery: " + msgdtlast;
        }
    }

    // compare to time-to-end
    if ((tstop - run_stat.time) < dt) {
        dt = tstop - run_stat.time;
        msgdt = "Global (tstop - time)";
    }

    // compare to hydro dt
    if (dt_hydro.dt < dt) {
        dt = dt_hydro.dt;
        msgdt = string(dt_hydro.message);
    }

	TimeStep recommend;
	recommend.dt = dt;
	snprintf(recommend.message, 80, "%s", msgdt.c_str());
	Future future_min = Parallel::globalMin(recommend, min_reduction, runtime, ctx);

	TimeStep ts = future_min.get_result<TimeStep>();
	dt = ts.dt;
	msgdt = string(ts.message);

}

