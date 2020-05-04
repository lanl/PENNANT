/*
 * HydroMPI.hh
 *
 *  Created on: Aug 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Triad National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDROMPI_H_
#define HYDROMPI_H_

#ifdef USE_MPI
#include <hip/hip_runtime.h>

void parallelGather(const int numslvpe, const int nummstrpe,
                    const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1,
                    const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1, 
                    double* pmaswt_proxy_buffer, double2* pf_proxy_buffer,
		    double* pmaswt_slave_buffer, double2* pf_slave_buffer);

void parallelScatter(const int numslvpe, const int nummstrpe,
		     const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1,
		     const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1, const int *mapslvp,
		     double* pmaswt_proxy_buffer, double2* pf_proxy_buffer,
		     double* pmaswt_slave_buffer, double2* pf_slave_buffer);
#endif // USE_MPI
#endif // HYDROMPI_H_
