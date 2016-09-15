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
#include "Hydro.hh"
#include "Mesh.hh"

using namespace std;


PolyGas::PolyGas(const InputParameters& params, Hydro* h) :
		hydro(h),
		gamma(params.directs_.gamma_),
		ssmin(params.directs_.ssmin_)
{
}


void PolyGas::calcStateAtHalf(
        const DoubleAccessor* zr0,
        const double* zvolp,
        const double* zvol0,
        const DoubleAccessor* ze,
        const double* zwrate,
        const double* zm,
        const double dt,
        DoubleAccessor* zp,
        double* zss,
        const int zfirst,
        const int zlast) {

    double* z0per = AbstractedMemory::alloc<double>(zlast - zfirst);

    const double dth = 0.5 * dt;

    // compute EOS at beginning of time step
    calcEOS(zr0, ze, zp, z0per, zss, zfirst, zlast);

    // now advance pressure to the half-step
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
        int z0 = z - zfirst;
        double zminv = 1. / zm[z];
        double dv = (zvolp[z] - zvol0[z]) * zminv;
        ptr_t zone_ptr(z);
        double bulk = zr0->read(zone_ptr) * zss[z] * zss[z];
        double denom = 1. + 0.5 * z0per[z0] * dv;
        double src = zwrate[z] * dth * zminv;
        double value = zp->read(zone_ptr) + (z0per[z0] * src - zr0->read(zone_ptr) * bulk * dv) / denom;
        zp->write(zone_ptr, value);
    }

    AbstractedMemory::free(z0per);
}


void PolyGas::calcEOS(
        const DoubleAccessor* zr,
        const DoubleAccessor* ze,
        DoubleAccessor* zp,
        double* z0per,
        double* zss,
        const int zfirst,
        const int zlast) {

    const double gm1 = gamma - 1.;
    const double ss2 = max(ssmin * ssmin, 1.e-99);

    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
    		ptr_t zone_ptr(z);
        int z0 = z - zfirst;
        double rx = zr->read(zone_ptr);
        double ex = max(ze->read(zone_ptr), 0.0);
        double px = gm1 * rx * ex;
        double prex = gm1 * ex;
        double perx = gm1 * rx;
        double csqd = max(ss2, prex + perx * px / (rx * rx));
        zp->write(zone_ptr,  px);
        z0per[z0] = perx;
        zss[z] = sqrt(csqd);
    }

}


void PolyGas::calcForce(
        const DoubleAccessor* zp,
        const double2* ssurfp,
        double2* sf,
        const int sfirst,
        const int slast) {

    const LocalMesh* mesh = hydro->mesh;

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = mesh->map_side2zone_[s];
        ptr_t zone_ptr(z);
        double2 sfx = -zp->read(zone_ptr) * ssurfp[s];
        sf[s] = sfx;

    }
}




