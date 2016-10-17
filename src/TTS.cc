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

#include<algorithm>

using namespace std;


void TTS::calcForce(
        const double* zarea,
        const double* zr,
        const double* zss,
        const double* sarea,
        const double* smf,
        const double2* ssurfp,
        double2* sf,
        const int sfirst,
        const int slast,
        const int* map_side2zone,
        const double ssmin,
        const double alfa) {

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

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];

        double svfacinv = zarea[z] / sarea[s];
        double srho = zr[z] * smf[s] * svfacinv;
        double sstmp = max(zss[z], ssmin);
        sstmp = alfa * sstmp * sstmp;
        double sdp = sstmp * (srho - zr[z]);
        double2 sqq = -sdp * ssurfp[s];
        sf[s] = sqq;

    }

}

