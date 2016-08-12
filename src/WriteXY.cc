/*
 * WriteXY.cc
 *
 *  Created on: Dec 16, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "WriteXY.hh"

#include <fstream>
#include <iomanip>

#include "Mesh.hh"

using namespace std;


WriteXY::WriteXY(Mesh* m) : mesh(m) {}

WriteXY::~WriteXY() {}


void WriteXY::write(
        const string& basename,
        const DoubleAccessor& zr,
        const DoubleAccessor& ze,
        const double* zp) {

    const int numz = mesh->num_zones_;

    int gnumz = numz;
    Parallel::globalSum(gnumz);
    gnumz = (Parallel::mype() == 0 ? gnumz : 0);
    vector<int> penumz(Parallel::mype() == 0 ? Parallel::num_subregions() : 0);
    Parallel::gather(numz, &penumz[0]);

    vector<double> gzp(gnumz);
    //Parallel::gatherv(&zr[0], numz, &gzr[0], &penumz[0]);
    //Parallel::gatherv(&ze[0], numz, &gze[0], &penumz[0]);
    Parallel::gatherv(&zp[0], numz, &gzp[0], &penumz[0]);

    if (Parallel::mype() == 0) {
        string xyname = basename + ".xy";
        ofstream ofs(xyname.c_str());
        ofs << scientific << setprecision(8);
        ofs << "#  zr" << endl;
        for (int z = 0; z < gnumz; ++z) {
    		    ptr_t zone_ptr(z);
            ofs << setw(5) << (z + 1) << setw(18) << zr.read(zone_ptr) << endl;
        }
        ofs << "#  ze" << endl;
        for (int z = 0; z < gnumz; ++z) {
        		ptr_t zone_ptr(z);
            ofs << setw(5) << (z + 1) << setw(18) << ze.read(zone_ptr) << endl;
        }
        ofs << "#  zp" << endl;
        for (int z = 0; z < gnumz; ++z) {
            ofs << setw(5) << (z + 1) << setw(18) << gzp[z] << endl;
        }
        ofs.close();

    } // if Parallel::mype()

}

