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

DriverTask::DriverTask(void *args, const size_t &size)
	 : TaskLauncher(DriverTask::TASK_ID, TaskArgument(args, size))
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

	// Legion cannot handle data structures with indirections in them
    void *indirect_args = task->args;
    SPMDArgs args;
    memcpy((void*)(&args), indirect_args, sizeof(SPMDArgs));
	unsigned char *next = (unsigned char*)indirect_args ;
	next += sizeof(SPMDArgs);

    InputParameters barf;

	  barf.ntasks_ = args.ntasks_;
	  barf.task_id_ = args.task_id_;
	  barf.tstop_ = args.tstop_;
	  barf.cstop_ = args.cstop_;
	  barf.dtmax_ = args.dtmax_;
	  barf.dtinit_ = args.dtinit_;
	  barf.dtfac_ = args.dtfac_;
	  barf.dtreport_ = args.dtreport_;
	  barf.chunk_size_ = args.chunk_size_;
	  barf.write_xy_file_ = args.write_xy_file_;
	  barf.write_gold_file_ = args.write_gold_file_;
	  barf.nzones_x_ = args.nzones_x_;
	  barf.nzones_y_ = args.nzones_y_;
	  barf.len_x_ = args.len_x_;
	  barf.len_y_ = args.len_y_;
	  barf.cfl_ = args.cfl_;
	  barf.cflv_ = args.cflv_;
	  barf.rho_init_ = args.rho_init_;
	  barf.energy_init_ = args.energy_init_;
	  barf.rho_init_sub_ = args.rho_init_sub_;
	  barf.energy_init_sub_ = args.energy_init_sub_;
	  barf.vel_init_radial_ = args.vel_init_radial_;
	  barf.gamma_ = args.gamma_;
	  barf.ssmin_ = args.ssmin_;
	  barf.alfa_ = args.alfa_;
	  barf.qgamma_ = args.qgamma_;
	  barf.q1_ = args.q1_;
	  barf.q2_ = args.q2_;
	  barf.subregion_xmin_ = args.subregion_xmin_;
	  barf.subregion_xmax_ = args.subregion_xmax_;
	  barf.subregion_ymin_ = args.subregion_ymin_;
	  barf.subregion_ymax_ = args.subregion_ymax_;

	  // Legion cannot handle data structures with indirections in them

	  size_t next_size = args.n_meshtype_ * sizeof(char);
	  char *love_legion1 = (char *)malloc(next_size+1);
	  memcpy((void *)love_legion1, (void *)next, next_size);
	  love_legion1[next_size] = '\0';
	  cout << "Love legion " << love_legion1 << endl;
	  barf.meshtype_ = string(love_legion1);
	  free(love_legion1);
	  next += next_size;

	  next_size = args.n_probname_ * sizeof(char);
	  char *love_legion2 = (char *)malloc(next_size+1);
	  memcpy((void *)love_legion2, (void *)next, next_size);
	  love_legion2[next_size] = '\0';
	  cout << "Love legion " << love_legion2 << endl;
	  barf.probname_ = string(love_legion2);
	  free(love_legion2);
	  next += next_size;

	  barf.bcx_.resize(args.n_bcx_);
	  next_size = args.n_bcx_ * sizeof(double);
	  memcpy((void *)&(barf.bcx_[0]), (void *)next, next_size);
	  next += next_size;

	  barf.bcy_.resize(args.n_bcy_);
	  next_size = args.n_bcy_ * sizeof(double);
	  memcpy((void *)&(barf.bcy_[0]), (void *)next, next_size);

    cout << "shard_id = " << args.shard_id_ << endl;
    cout << "cstop = " << barf.cstop_ << endl;
    cout << "tstop = " << barf.tstop_ << endl;
    cout << "meshtype = " << barf.meshtype_ << endl;
    cout << "probname = " << barf.probname_ << endl;
    cout << "meshparams = " << barf.nzones_x_ << ","
    		<< barf.nzones_y_ << ","
		<< barf.len_x_ << ","
		<< barf.len_y_ << ","
     	<< endl;
    cout << "subregion = " << barf.subregion_xmax_ << endl;
    cout << "rinitsub = " << barf.rho_init_sub_ << endl;
    cout << "einitsub = " << barf.energy_init_sub_ << endl;
    cout << barf.bcx_.size() << " bcx = " << barf.bcx_[0] << endl;
    cout << "bcx = " << barf.bcx_[1] << endl;
    cout << barf.bcy_.size() << " bcy = " << barf.bcy_[0] << endl;
    cout << "bcy = " << barf.bcy_[1] << endl;
    cout << "ssmin = " << barf.ssmin_ << endl;
    cout << "q1 = " << barf.q1_ << endl;
    cout << "q2 = " << barf.q2_ << endl;
    cout << "dtinit = " << barf.dtinit_ << endl;
    cout << "writexy = " << barf.write_xy_file_ << endl;
    cout << "chunksize = " << barf.chunk_size_ << endl;

    Driver drv(barf);

    drv.run();

}

Driver::Driver(const InputParameters& params)
        : probname(params.probname_),
		  tstop(params.tstop_),
		  cstop(params.cstop_),
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

