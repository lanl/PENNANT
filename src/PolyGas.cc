/*
 * PolyGas.cc
 *
 *  Created on: Mar 26, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "PolyGas.hh"

#include "Memory.hh"

using namespace std;


/*static*/
void PolyGas::calcStateAtHalf(
        const double* zr0,
        const double* zvolp,
        const double* zvol0,
        const double* ze,
        const double* zwrate,
        const double* zm,
        const double dt,
        double* zp,
        double* zss,
        const int zfirst,
        const int zlast,
        const double gamma,
        const double ssmin) {

    double* z0per = AbstractedMemory::alloc<double>(zlast - zfirst);

    const double dth = 0.5 * dt;

    // compute EOS at beginning of time step
    calcEOS(zr0, ze, zp, z0per, zss, zfirst, zlast, gamma, ssmin);

    // now advance pressure to the half-step
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        int z0 = z - zfirst;
        double zminv = 1. / zm[z];
        double dv = (zvolp[z] - zvol0[z]) * zminv;
        double bulk = zr0[z] * zss[z] * zss[z];
        double denom = 1. + 0.5 * z0per[z0] * dv;
        double src = zwrate[z] * dth * zminv;
        double value = zp[z] + (z0per[z0] * src - zr0[z] * bulk * dv) / denom;
        zp[z] = value;
    }

    AbstractedMemory::free(z0per);
}


/*static*/
void PolyGas::calcEOS(
        const double* zr,
        const double* ze,
        double* zp,
        double* z0per,
        double* zss,
        const int zfirst,
        const int zlast,
        const double gamma,
        const double ssmin) {

    const double gm1 = gamma - 1.;
    const double ss2 = max(ssmin * ssmin, 1.e-99);

    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        int z0 = z - zfirst;
        double rx = zr[z];
        double ex = max(ze[z], 0.0);
        double px = gm1 * rx * ex;
        double prex = gm1 * ex;
        double perx = gm1 * rx;
        double csqd = max(ss2, prex + perx * px / (rx * rx));
        zp[z] =  px;
        z0per[z0] = perx;
        zss[z] = sqrt(csqd);
    }

}


/*static*/
void PolyGas::calcForce(
        const double* zp,
        const double2* ssurfp,
        double2* sf,
        const int sfirst,
        const int slast,
        const int* map_side2zone) {

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        double2 sfx = -zp[z] * ssurfp[s];
        sf[s] = sfx;

    }
}




