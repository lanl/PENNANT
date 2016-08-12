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

DriverTask::DriverTask(//LogicalRegion lregion_my_zones,
		LogicalRegion lregion_global_zones,
		void *args, const size_t &size)
	 : TaskLauncher(DriverTask::TASK_ID, TaskArgument(args, size))
{
	add_region_requirement(RegionRequirement(lregion_global_zones, WRITE_DISCARD, EXCLUSIVE, lregion_global_zones));
	add_field(0/*idx*/, FID_ZR);
	add_field(0/*idx*/, FID_ZE);
	add_field(0/*idx*/, FID_ZP);
}

/*static*/ const char * const DriverTask::TASK_NAME = "DriverTask";

/*static*/
void DriverTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* rt)
{
	assert(regions.size() == 1);
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 3);

	// Legion cannot handle data structures with indirections in them
    unsigned char *serialized_args = (unsigned char *) task->args;
    SPMDArgs args;
	size_t next_size = sizeof(SPMDArgs);
    memcpy((void*)(&args), (void*)serialized_args, next_size);
	serialized_args += sizeof(SPMDArgs);

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

    Driver drv(params, args.add_reduction_, args.min_reduction_,
    		regions[0],
		ctx, rt);
    drv.run();

}

Driver::Driver(const InputParameters& params,
		DynamicCollective add_reduction,
		DynamicCollective min_reduction,
		const PhysicalRegion &zones,
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
		  runtime_(rt)
{

    // initialize mesh, hydro
    mesh = new Mesh(params);
    hydro = new Hydro(params, mesh, add_reduction_, zones, ctx_, runtime_);

}

Driver::~Driver() {

    delete hydro;
    delete mesh;

}

void Driver::run() {
    time = 0.0;
    cycle = 0;

    // do energy check
    hydro->writeEnergyCheck();

    double tbegin, tlast;
    if (Parallel::mype() == 0) {
        // get starting timestamp
        struct timeval sbegin;
        gettimeofday(&sbegin, NULL);
        tbegin = sbegin.tv_sec + sbegin.tv_usec * 1.e-6;
        tlast = tbegin;
    }

    // main event loop
    while (cycle < cstop && time < tstop) {

        cycle += 1;

        // get timestep
        calcGlobalDt();

        // begin hydro cycle
        hydro->doCycle(dt);

        time += dt;

        if (Parallel::mype() == 0 &&
                (cycle == 1 || cycle % dtreport == 0)) {
            struct timeval scurr;
            gettimeofday(&scurr, NULL);
            double tcurr = scurr.tv_sec + scurr.tv_usec * 1.e-6;
            double tdiff = tcurr - tlast;

            cout << scientific << setprecision(5);
            cout << "End cycle " << setw(6) << cycle
                 << ", time = " << setw(11) << time
                 << ", dt = " << setw(11) << dt
                 << ", wall = " << setw(11) << tdiff << endl;
            cout << "dt limiter: " << msgdt << endl;

            tlast = tcurr;
        } // if Parallel::mype()...

    } // while cycle...

    if (Parallel::mype() == 0) {

        // get stopping timestamp
        struct timeval send;
        gettimeofday(&send, NULL);
        double tend = send.tv_sec + send.tv_usec * 1.e-6;
        double runtime = tend - tbegin;

        // write end message
        cout << endl;
        cout << "Run complete" << endl;
        cout << scientific << setprecision(6);
        cout << "cycle = " << setw(6) << cycle
             << ",         cstop = " << setw(6) << cstop << endl;
        cout << "time  = " << setw(14) << time
             << ", tstop = " << setw(14) << tstop << endl;

        cout << endl;
        cout << "************************************" << endl;
        cout << "hydro cycle run time= " << setw(14) << runtime << endl;
        cout << "************************************" << endl;

    } // if Parallel::mype()

    // do energy check
    hydro->writeEnergyCheck();

    // do final mesh output
    mesh->write(probname, cycle, time,
            hydro->zone_rho_, hydro->zone_energy_density_, hydro->zone_pres);

}

// TODO make this a task and collapse Driver into DriverTask
void Driver::calcGlobalDt() {

    // Save timestep from last cycle
    dtlast = dt;
    msgdtlast = msgdt;

    // Compute timestep for this cycle
    dt = dtmax;
    msgdt = "Global maximum (dtmax)";

    if (cycle == 1) {
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
    if ((tstop - time) < dt) {
        dt = tstop - time;
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

