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

DriverTask::DriverTask(SPMDArgs *args)
	 : TaskLauncher(DriverTask::TASK_ID, TaskArgument(args, sizeof(SPMDArgs)))
{
}

/*static*/ const char * const DriverTask::TASK_NAME = "DriverTask";

/*static*/
void DriverTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* rt)
{
	// Unmap all the regions we were given since we won't actually use them
	rt->unmap_all_regions(ctx);

    SPMDArgs *args = (SPMDArgs *)(task->args);

}

Driver::Driver(const InputParameters& params)
        : probname(params.probname),
		  cstop(params.cstop_),
		  tstop(params.tstop_),
		  dtmax(params.dtmax_),
		  dtinit(params.dtinit_),
		  dtfac(params.dtfac_),
		  dtreport(params.dtreport_)
{

    // initialize mesh, hydro
    mesh = new Mesh(params);
    hydro = new Hydro(params, mesh);

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
            hydro->zone_rho, hydro->zone_energy_density, hydro->zone_pres);

}


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

#ifdef USE_MPI
    int pedt;
    Parallel::globalMinLoc(dt, pedt);

    // if the global min isn't on this PE, get the right message
    if (pedt > 0) {
        const int tagmpi = 300;
        if (Parallel::mype() == pedt) {
            char cmsgdt[80];
            strncpy(cmsgdt, msgdt.c_str(), 80);
            MPI_Send(cmsgdt, 80, MPI_CHAR, 0, tagmpi,
                    MPI_COMM_WORLD);
        }
        else if (Parallel::mype() == 0) {
            char cmsgdt[80];
            MPI_Status status;
            MPI_Recv(cmsgdt, 80, MPI_CHAR, pedt, tagmpi,
                    MPI_COMM_WORLD, &status);
            cmsgdt[79] = '\0';
            msgdt = string(cmsgdt);
        }
    }  // if pedt > 0

    // if timestep was determined by hydro, report which PE
    // caused it
    if (Parallel::mype() == 0 && msgdt.substr(0, 5) == "Hydro") {
        ostringstream oss;
        oss << "PE " << pedt << ", " << msgdt;
        msgdt = oss.str();
    }
#endif

}

