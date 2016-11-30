/*
 * HydroBC.cc
 *
 *  Created on: Jan 13, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "HydroBC.hh"

#include "LocalMesh.hh"
#include "Memory.hh"

using namespace std;


/*static*/
void HydroBC::applyFixedBC(
        const ptr_t* pt_local2globalID,
        const double2 vfix,
        const vector<int>& mapbp,
        double2* pu,
        Double2SOAAccessor& pf,
        const int bfirst,
        const int blast) {

    #pragma ivdep
    for (int b = bfirst; b < blast; ++b) {
        int p = mapbp[b];
        ptr_t pt_ptr = pt_local2globalID[p];

        pu[p] = project(pu[p], vfix);
        double2 old_pf = pf.read(pt_ptr);
        pf.write(pt_ptr, project(old_pf, vfix));
    }

}

