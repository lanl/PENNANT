/*
 * TTS.cc
 *
 *  Created on: Feb 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "TTS.hh"

#include "Vec2.hh"
#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;


TTS::TTS(const InputParameters& params, Hydro* h) :
		hydro(h),
		alfa(params.directs_.alfa_),
		ssmin(params.directs_.ssmin_)
{
}


TTS::~TTS() {}


void TTS::calcForce(
        const double* zarea,
        const DoubleAccessor* zr,
        const double* zss,
        const double* sarea,
        const double* smf,
        const double2* ssurfp,
        double2* sf,
        const int sfirst,
        const int slast) {

    //  Side density:
    //    srho = sm/sv = zr (sm/zm) / (sv/zv)
    //  Side pressure:
    //    sp   = zp + alfa dpdr (srho-zr)
    //         = zp + sdp
    //  Side delta pressure:
    //    sdp  = alfa dpdr (srho-zr)
    //         = alfa c**2 (srho-zr)
    //
    //    Notes: smf stores (sm/zm)
    //           svfac stores (sv/zv)

    const Mesh* mesh = hydro->mesh;

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = mesh->map_side2zone_[s];
        ptr_t zone_ptr(z);

        double svfacinv = zarea[z] / sarea[s];
        double srho = zr->read(zone_ptr) * smf[s] * svfacinv;
        double sstmp = max(zss[z], ssmin);
        sstmp = alfa * sstmp * sstmp;
        double sdp = sstmp * (srho - zr->read(zone_ptr));
        double2 sqq = -sdp * ssurfp[s];
        sf[s] = sqq;

    }

}

