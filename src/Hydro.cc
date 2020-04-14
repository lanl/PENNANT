/*
 * Hydro.cc
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Triad National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Hydro.hh"

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include <iostream>
#ifdef USE_MPI
#include "Parallel.hh"
#endif

#include "Memory.hh"
#include "InputFile.hh"
#include "Mesh.hh"
#include "PolyGas.hh"
#include "TTS.hh"
#include "QCS.hh"
#include "HydroBC.hh"
#include "HydroGPU.hh"

using namespace std;


Hydro::Hydro(const InputFile* inp, Mesh* m) : mesh(m) {
    cfl = inp->getDouble("cfl", 0.6);
    cflv = inp->getDouble("cflv", 0.1);
    rinit = inp->getDouble("rinit", 1.);
    einit = inp->getDouble("einit", 0.);
    rinitsub = inp->getDouble("rinitsub", 1.);
    einitsub = inp->getDouble("einitsub", 0.);
    uinitradial = inp->getDouble("uinitradial", 0.);
    bcx = inp->getDoubleList("bcx", vector<double>());
    bcy = inp->getDoubleList("bcy", vector<double>());

    pgas = new PolyGas(inp, this);
    tts = new TTS(inp, this);
    qcs = new QCS(inp, this);

    init();
    hydroInitGPU();

#ifdef USE_MPI
    if (Parallel::numpe > 1){
      hydroInitMPI(mesh->nummstrpe, mesh->numslvpe, mesh->numprx, mesh->numslv,
		   mesh->mapslvpepe, mesh->mapslvpeprx1, mesh->mapprxp,
		   mesh->slvpenumprx, mesh->mapmstrpepe, mesh->mstrpenumslv,
		   mesh->mapmstrpeslv1, mesh->mapslvp);
    }
#endif

    hydroInit(mesh->nump, mesh->numz, mesh->nums, mesh->numc, mesh->nume,
	      pgas->gamma, pgas->ssmin,
	      tts->alfa, tts->ssmin,
	      qcs->qgamma, qcs->q1, qcs->q2,
	      cfl, cflv,
	      bcx.size(), &bcx[0], bcy.size(), &bcy[0],
	      mesh->px,
	      pu,
	      zm,
	      zr,
	      mesh->zvol,
	      ze, zetot,
	      zwrate,
	      mesh->smf,
	      mesh->mapsp1,
	      mesh->mapsp2,
	      mesh->mapsz,
	      mesh->mapss4,
	      mesh->mapse,
	      mesh->znump);
}


Hydro::~Hydro() {

    hydroFinalGPU();

    delete tts;
    delete qcs;
    for (int i = 0; i < bcs.size(); ++i) {
        delete bcs[i];
    }
}


void Hydro::init() {

    dtrec = 1.e99;
    msgdtrec = "Hydro default";

    const int nump = mesh->nump;
    const int numz = mesh->numz;
    const int nums = mesh->nums;

    const double2* zx = mesh->zx;
    const double* zvol = mesh->zvol;




    // allocate arrays
    pu = Memory::alloc<double2>(nump);
    zm = Memory::alloc<double>(numz);
    zr = Memory::alloc<double>(numz);
    ze = Memory::alloc<double>(numz);
    zetot = Memory::alloc<double>(numz);
    zwrate = Memory::alloc<double>(numz);
    zp = Memory::alloc<double>(numz);
    pmaswt = Memory::alloc<double>(nump);
    cmaswt = Memory::alloc<double>(nums);
    cftot = Memory::alloc<double2>(nums);
    pf = Memory::alloc<double2>(nump);
    // initialize hydro vars
    fill(&zr[0], &zr[numz], rinit);
    fill(&ze[0], &ze[numz], einit);
    fill(&zwrate[0], &zwrate[numz], 0.);

    const vector<double>& subrgn = mesh->subregion;
    if (!subrgn.empty()) {
        const double eps = 1.e-12;
        for (int z = 0; z < numz; ++z) {
            if (zx[z].x > (subrgn[0] - eps) &&
                zx[z].x < (subrgn[1] + eps) &&
                zx[z].y > (subrgn[2] - eps) &&
                zx[z].y < (subrgn[3] + eps)) {
                zr[z] = rinitsub;
                ze[z] = einitsub;
            }
        }
    }

    for (int z = 0; z < numz; ++z) {
        zm[z] = zr[z] * zvol[z];
        zetot[z] = ze[z] * zm[z];
    }

    if (uinitradial != 0.)
        initRadialVel(uinitradial);
    else
        fill(&pu[0], &pu[nump], double2(0., 0.));
}


void Hydro::getData() {

    hydroGetData( mesh->zarea,zetot,mesh->zvol,
            mesh->nump, mesh->numz,
            mesh->px,
            zr, ze, zp,pu);
}


void Hydro::initRadialVel(const double vel) {
    const int nump = mesh->nump;
    const double2* px = mesh->px;
    const double eps = 1.e-12;

    for (int p = 0; p < nump; ++p) {
        double pmag = length(px[p]);
        if (pmag > eps)
            pu[p] = vel * px[p] / pmag;
        else
            pu[p] = double2(0., 0.);
    }
}


void Hydro::doCycle(
            const double dt) {

    int idtrec;
    const int nump = mesh->nump;
    const int numz = mesh->numz;
    const int nums = mesh->nums;

    hydroDoCycle(dt, dtrec, idtrec);
    int z = idtrec >> 1;
    bool dtfromvol = idtrec & 1;
    ostringstream oss;
    if (dtfromvol)
        oss << "Hydro dV/V limit for z = " << setw(6) << z;
    else
        oss << "Hydro Courant limit for z = " << setw(6) << z;
    msgdtrec = oss.str();
}


void Hydro::getDtHydro(
        double& dtnew,
        string& msgdtnew) {

    if (dtrec < dtnew) {
        dtnew = dtrec;
        msgdtnew = msgdtrec;
    }

}

void Hydro::sumEnergy(
        const double* zetot,
        const double* zarea,
        const double* zvol,
        const double* zm,
        const double* smf,
        const double2* px,
        const double2* pu,
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
        int s3 = mesh->mapss3[s];
        int p1 = mesh->mapsp1[s];
        int z = mesh->mapsz[s];

        double cvol = zarea[z] * px[p1].x * 0.5 * (smf[s] + smf[s3]);
        double cke = zm[z] * cvol / zvol[z] * 0.5 * length2(pu[p1]);
        sumk += cke;
    }
    // multiply by 2\pi for cylindrical geometry
    ek += sumk * 2 * M_PI;
}


void Hydro::writeEnergyCheck() {

#ifdef USE_MPI
    using Parallel::mype;
#else
    constexpr int mype = 0;
#endif
    
    double ei = 0.;
    double ek = 0.;
    #pragma omp parallel for schedule(static)
    for (int sch = 0; sch < mesh->numsch; ++sch) {
        int sfirst = mesh->schsfirst[sch];
        int slast = mesh->schslast[sch];
        int zfirst = mesh->schzfirst[sch];
        int zlast = mesh->schzlast[sch];

        double eichunk = 0.;
        double ekchunk = 0.;
        sumEnergy(zetot, mesh->zarea, mesh->zvol, zm, mesh->smf,
                mesh->px, pu, eichunk, ekchunk,
                zfirst, zlast, sfirst, slast);
        #pragma omp critical
        {
            ei += eichunk;
            ek += ekchunk;
        }
    }
#ifdef USEMPI
    Parallel::globalSum(ei);
    Parallel::globalSum(ek);
#endif

    if (mype == 0) {
        cout << scientific << setprecision(6);
        cout << "Energy check:  "
             << "total energy  = " << setw(14) << ei + ek << endl;
        cout << "(internal = " << setw(14) << ei
             << ", kinetic = " << setw(14) << ek << ")" << endl;
    }
 
}
