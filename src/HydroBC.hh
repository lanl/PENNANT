/*
 * HydroBC.hh
 *
 *  Created on: Jan 13, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDROBC_HH_
#define HYDROBC_HH_

#include <vector>

#include "GenerateMesh.hh"
#include "Parallel.hh"
#include "Vec2.hh"

// forward declarations
class LocalMesh;


class HydroBC {
public:

    static void applyFixedBC(
            const ptr_t* pt_local2globalID,
            const double2 vfix,
            const std::vector<int>& mapbp,
            double2* pu,
            Double2SOAAccessor& pf,
            const int bfirst,
            const int blast);

}; // class HydroBC


#endif /* HYDROBC_HH_ */
