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

#include "Memory.hh"
#include "Mesh.hh"

using namespace std;


HydroBC::HydroBC(
        LocalMesh* msh,
        const double2 v,
        const vector<int>& mbp)
    : mesh(msh), numb(mbp.size()), vfix(v) {

    mapbp = AbstractedMemory::alloc<int>(numb);
    copy(mbp.begin(), mbp.end(), mapbp);

    mesh->getPlaneChunks(numb, mapbp, pchbfirst, pchblast);

}


HydroBC::~HydroBC() {}


void HydroBC::applyFixedBC(
        double2* pu,
        Double2Accessor& pf,
        const int bfirst,
        const int blast) {

    #pragma ivdep
    for (int b = bfirst; b < blast; ++b) {
        int p = mapbp[b];
        ptr_t pt_ptr = mesh->point_local_to_globalID[p];

        pu[p] = project(pu[p], vfix);
        double2 old_pf = pf.read(pt_ptr);
        pf.write(pt_ptr, project(old_pf, vfix));
    }

}

