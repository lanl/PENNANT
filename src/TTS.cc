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

#include <algorithm>

using namespace std;

// TTS = Temporary Triangular Subzoning
//
// This algorithm prevents certain kinds of grid-shape distortions (such as
// "hourglassing" by estimating a pressure for each side and adding a force
// to each side based on the difference between the zone and side pressures

void TTS::calcForce(const double*__restrict__ zarea,
    const double*__restrict__ zr, const double*__restrict__ zss,
    const double*__restrict__ sarea, const double*__restrict__ smf,
    const double2*__restrict__ ssurfp, double2*__restrict__ sf,
    const int sfirst, const int slast, const int*__restrict__ map_side2zone,
    const double ssmin, const double alpha) {

  //  Side density:
  //    srho = sm/sv = zr (sm/zm) / (sv/zv)
  //  Side pressure:
  //    sp   = zp + alpha dpdr (srho-zr)
  //         = zp + sdp
  //  Side delta pressure:
  //    sdp  = alpha dpdr (srho-zr)
  //         = alpha c**2 (srho-zr)
  //
  //    Notes: smf stores (sm/zm)
  //           svfac stores (sv/zv)

#pragma ivdep
  for (int s = sfirst; s < slast; ++s) {
    int z = map_side2zone[s];

    double svfacinv = zarea[z] / sarea[s];
    double srho = zr[z] * smf[s] * svfacinv;
    double sstmp = max(zss[z], ssmin);
    sstmp = alpha * sstmp * sstmp;
    double sdp = sstmp * (srho - zr[z]);
    double2 sqq = -sdp * ssurfp[s];
    sf[s] = sqq;
  }
}

