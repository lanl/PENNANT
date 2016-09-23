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
#include <sstream>
#include <iomanip>

#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;

DriverTask::DriverTask(int my_color,
		LogicalRegion my_zones,
		LogicalRegion all_zones,
		LogicalRegion my_pts,
		LogicalRegion all_pts,
		std::vector<LogicalRegion> ghost_pts,
		void *args, const size_t &size)
	 : TaskLauncher(DriverTask::TASK_ID, TaskArgument(args, size))
{
	add_region_requirement(RegionRequirement(my_zones, WRITE_DISCARD, EXCLUSIVE, all_zones));
	add_field(0/*idx*/, FID_ZR);
	add_field(0/*idx*/, FID_ZE);
	add_field(0/*idx*/, FID_ZP);
	add_region_requirement(RegionRequirement(my_pts, READ_ONLY, EXCLUSIVE, all_pts));
	add_field(1/*idx*/, FID_GHOST_PF);  // TODO until real ghost regions give access to index space
/*	for (int color=0; color < ghost_pts.size(); ++color) {
		add_region_requirement(RegionRequirement(ghost_pts[color], READ_WRITE, SIMULTANEOUS, ghost_pts[color]));
		if (color != my_color)
			region_requirements[2+color].add_flags(NO_ACCESS_FLAG);
		add_field(2+color, FID_GHOST_PF);
		add_field(2+color, FID_GHOST_PMASWT);
	}*/
}

/*static*/ const char * const DriverTask::TASK_NAME = "DriverTask";

/*static*/
RunStat DriverTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
	//assert(regions.size() > 2);
	//assert(task->regions.size() > 2);
	assert(task->regions[0].privilege_fields.size() == 3);
	assert(task->regions[1].privilege_fields.size() == 1);
	//assert(task->regions[2].privilege_fields.size() == 2);

	// Legion Zones
	LogicalUnstructured zones(ctx, runtime, regions[0]);

	// Legion cannot handle data structures with indirections in them
	// TODO hide this behing a static member function
	unsigned char *serialized_args = (unsigned char *) task->args;
    SPMDArgs args;
	size_t next_size = sizeof(SPMDArgs);
    memcpy((void*)(&args), (void*)serialized_args, next_size);
	serialized_args += next_size;

    InputParameters params;
    params.directs_ = args.direct_input_params_;

    // Legion cannot handle data structures with indirections in them
    {
      next_size = args.n_meshtype_ * sizeof(char);
	  char *buffer = (char *)malloc(next_size+1);
	  memcpy((void *)buffer, (void *)serialized_args, next_size);
	  buffer[next_size] = '\0';
	  params.meshtype_ = string(buffer);
	  free(buffer);
	  serialized_args += next_size;
    }
    {
	  next_size = args.n_probname_ * sizeof(char);
	  char *buffer = (char *)malloc(next_size+1);
	  memcpy((void *)buffer, (void *)serialized_args, next_size);
	  buffer[next_size] = '\0';
	  params.probname_ = string(buffer);
	  free(buffer);
	  serialized_args += next_size;
    }
    {
	  params.bcx_.resize(args.n_bcx_);
	  next_size = args.n_bcx_ * sizeof(double);
	  memcpy((void *)&(params.bcx_[0]), (void *)serialized_args, next_size);
	  serialized_args += next_size;
    }
    {
	  params.bcy_.resize(args.n_bcy_);
	  next_size = args.n_bcy_ * sizeof(double);
	  memcpy((void *)&(params.bcy_[0]), (void *)serialized_args, next_size);
    }

    // For some reason Legion explodes if I do this in the Hydro object or use LogicalUnstructured
    DoubleAccessor zone_rho = zones.getRegionAccessor<double>(FID_ZR); // TODO can I do this in driver object or combine objects?
    DoubleAccessor zone_energy_density = zones.getRegionAccessor<double>(FID_ZE);
    DoubleAccessor zone_pressure = zones.getRegionAccessor<double>(FID_ZP);

    Driver drv(params, args.add_reduction_, args.min_reduction_,
            args.pbarrier_as_master, //args.masters_pbarriers,
            &zone_rho, &zone_energy_density, &zone_pressure, zones,
            regions[1], //regions[2],
            ctx, runtime);

    RunStat value=drv.run();
    return value;
}

Driver::Driver(const InputParameters& params,
		DynamicCollective add_reduction,
		DynamicCollective min_reduction,
        PhaseBarrier pbarrier_as_master,
        //std::vector<PhaseBarrier> masters_pbarriers,
		DoubleAccessor* zone_rho,
		DoubleAccessor* zone_energy_density,
		DoubleAccessor* zone_pressure,
        LogicalUnstructured& global_comm_zones,
		const PhysicalRegion& pts,
        Context ctx, HighLevelRuntime* rt)
        : probname(params.probname_),
		  tstop(params.directs_.tstop_),
		  cstop(params.directs_.cstop_),
		  dtmax(params.directs_.dtmax_),
		  dtinit(params.directs_.dtinit_),
		  dtfac(params.directs_.dtfac_),
		  dtreport(params.directs_.dtreport_),
		  add_reduction_(add_reduction),
		  min_reduction_(min_reduction),
		  ctx_(ctx),
		  runtime_(rt),
		  mype_(params.directs_.task_id_),
          points(ctx, rt, pts),
          zone_rho(zone_rho),
          zone_energy_density(zone_energy_density),
          zone_pressure(zone_pressure),
          ispace_zones(global_comm_zones.getISpace())
{
    std::cout << mype_ << " driver " << pbarrier_as_master.get_barrier().id << std::endl;

    mesh = new LocalMesh(params, points, //ghost_pts,
    		ctx_, runtime_);
    hydro = new Hydro(params, mesh, add_reduction_, ctx_, runtime_);
}

RunStat Driver::run() {
    run_stat.time = 0.0;
    run_stat.cycle = 0;

    // do energy check
    hydro->writeEnergyCheck();

    double tbegin, tlast;
    if (mype_ == 0) {
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
        hydro->doCycle(dt);

        run_stat.time += dt;

        if (mype_ == 0 &&
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

    // copy Hydro zone data to legion regions
    hydro->copyZonesToLegion(zone_rho, zone_energy_density, zone_pressure, ispace_zones);

    if (mype_ == 0) {

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

// TODO make this a task and collapse Driver into DriverTask
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
    hydro->getDtHydro(dt, msgdt);

	TimeStep recommend;
	recommend.dt_ = dt;
	snprintf(recommend.message_, 80, "%s", msgdt.c_str());
	Future future_min = Parallel::globalMin(recommend, min_reduction_, runtime_, ctx_);

	TimeStep ts = future_min.get_result<TimeStep>();
	dt = ts.dt_;
	msgdt = string(ts.message_);

}

