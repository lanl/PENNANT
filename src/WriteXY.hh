/*
 * WriteXY.hh
 *
 *  Created on: Dec 16, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef WRITEXY_HH_
#define WRITEXY_HH_

#include <string>

#include "Parallel.hh"

// forward declarations
class LocalMesh;


class WriteXY {
public:

    static void write(
            const std::string& basename,
            const DoubleAccessor& zr,
            const DoubleAccessor& ze,
            const DoubleAccessor& zp,
			IndexIterator& zr_itr,
			IndexIterator& ze_itr,
			IndexIterator& zp_itr);

};


#endif /* WRITEXY_HH_ */
