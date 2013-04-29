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
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#ifdef _OPENMP
#include "omp.h"
#endif

#include "InputFile.hh"
#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;


Driver::Driver(const InputFile* inp, const string& pname)
        : probname(pname) {
    cout << "********************" << endl;
    cout << "Running PENNANT v0.4" << endl;
    cout << "********************" << endl;
    cout << endl;

#ifdef _OPENMP
    cout << "Running on " << omp_get_max_threads() << " threads" << endl;
#endif

    cstop = inp->getInt("cstop", 999999);
    tstop = inp->getDouble("tstop", 1.e99);
    if (cstop == 999999 && tstop == 1.e99) {
        cerr << "Must specify either cstop or tstop" << endl;
        exit(1);
    }
    dtmax = inp->getDouble("dtmax", 1.e99);
    dtinit = inp->getDouble("dtinit", 1.e99);
    dtfac = inp->getDouble("dtfac", 1.2);
    dtreport = inp->getInt("dtreport", 10);

    // initialize mesh, hydro
    mesh = new Mesh(inp);
    hydro = new Hydro(inp, mesh);

}

Driver::~Driver() {

    delete hydro;
    delete mesh;

}

void Driver::run() {

    const int numz = mesh->numz;

    time = 0.0;
    cycle = 0;
    double* zr = hydro->zr;
    double* ze = hydro->ze;
    double* zp = hydro->zp;

    // get starting timestamp
    struct timeval sbegin;
    gettimeofday(&sbegin, NULL);
    double tbegin = sbegin.tv_sec + sbegin.tv_usec * 1.e-6;

    // main event loop
    while (cycle < cstop && time < tstop) {

        cycle += 1;

        // get timestep
        calcGlobalDt();

        // begin hydro cycle
        hydro->doCycle(dt);

        time += dt;

        if (cycle == 1 || cycle % dtreport == 0) {
            cout << scientific << setprecision(5);
            cout << "End cycle " << setw(6) << cycle
                 << ", time = " << setw(11) << time
                 << ", dt = " << setw(11) << dt << endl;
            cout << "dt limiter: " << msgdt << endl;
        }

    } // while cycle...

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
    cout << "**************************************" << endl;
    cout << "total problem run time= " << setw(14) << runtime << endl;
    cout << "**************************************" << endl;


    // write output data files
    string xyname = probname + ".xy";
    ofstream ofs(xyname.c_str());
    ofs << scientific << setprecision(8);
    ofs << "#  zr" << endl;
    for (int z = 0; z < numz; ++z) {
        ofs << setw(5) << (z + 1) << setw(18) << zr[z] << endl;
    }
    ofs << "#  ze" << endl;
    for (int z = 0; z < numz; ++z) {
        ofs << setw(5) << (z + 1) << setw(18) << ze[z] << endl;
    }
    ofs << "#  zp" << endl;
    for (int z = 0; z < numz; ++z) {
        ofs << setw(5) << (z + 1) << setw(18) << zp[z] << endl;
    }
    ofs.close();
    cycle += 1;

    mesh->write(probname, cycle, time, zr, ze, zp);

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
            if (msgdtlast.substr(0, 10) == "Recovery: ")
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

}

