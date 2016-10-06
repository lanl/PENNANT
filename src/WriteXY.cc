/*
 * WriteXY.cc
 *
 *  Created on: Dec 16, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "WriteXY.hh"

#include <fstream>
#include <iomanip>

#include "LocalMesh.hh"

using namespace std;

// static
void WriteXY::write(
        const string& basename,
        const DoubleAccessor& zr,
        const DoubleAccessor& ze,
        const DoubleAccessor& zp,
		IndexIterator& zr_itr,
		IndexIterator& ze_itr,
		IndexIterator& zp_itr)
{

        string xyname = basename + ".xy";
        ofstream ofs(xyname.c_str());
        ofs << scientific << setprecision(8);
        ofs << "#  zr" << endl;
        while (zr_itr.has_next()) {
    		    ptr_t zone_ptr = zr_itr.next();
            ofs << setw(5) << (zone_ptr.value + 1) << setw(18) << zr.read(zone_ptr) << endl;
        }
        ofs << "#  ze" << endl;
        while (ze_itr.has_next()) {
		    ptr_t zone_ptr = ze_itr.next();
            ofs << setw(5) << (zone_ptr.value + 1) << setw(18) << ze.read(zone_ptr) << endl;
        }
        ofs << "#  zp" << endl;
        while (zp_itr.has_next()) {
		    ptr_t zone_ptr = zp_itr.next();
            ofs << setw(5) << (zone_ptr.value + 1) << setw(18) << zp.read(zone_ptr) << endl;
        }
        ofs.close();

}

