/*
 * HydroGPU.cu
 *
 *  Created on: Aug 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Triad National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Parallel.hh"
#include "Memory.hh"
#include "HydroMPI.hh"



void parallelGather(const int numslv, const int numslvpe, const int nummstrpe,
		    const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1, 
		    const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1,	
        	    const double* pvar, double* prxvar, double2* prxvar1, double *slvvar, double2 *slvvar1) {
#ifdef USE_MPI

using Parallel::numpe;
    using Parallel::mype;
    // This routine gathers slave values for which MYPE owns the masters.
    const int tagmpi = 100;
    // Post receives for incoming messages from slaves.
    // Store results in proxy buffer.
    MPI_Request* request = Memory::alloc<MPI_Request>(numslvpe);
    MPI_Request* request1 = Memory::alloc<MPI_Request>(numslvpe);
    //printf("%d: numslvpe=%d nummstrpe=%d\n",mype, numslvpe,nummstrpe);
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];

        MPI_Irecv(&prxvar[prx1], nprx * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request[slvpe]);

	MPI_Irecv(&prxvar1[prx1], nprx * sizeof(double2), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request1[slvpe]);
    }
    // Send slave data to master PEs.
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Send(&slvvar[slv1], nslv * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
	MPI_Send(&slvvar1[slv1], nslv * sizeof(double2), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
    MPI_Status* status = Memory::alloc<MPI_Status>(numslvpe);
    MPI_Status* status1 = Memory::alloc<MPI_Status>(numslvpe);
    int ierr = MPI_Waitall(numslvpe, &request[0], &status[0]);
    int ierr1 = MPI_Waitall(numslvpe, &request1[0], &status1[0]);

    Memory::free(request);
    Memory::free(status);
    Memory::free(request1);
    Memory::free(status1);
#endif
}







void parallelScatter(const int numslv, const int numslvpe, const int nummstrpe,
                    const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1,
                    const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1, const int *mapslvp,
                    double* pvar, double* prxvar, double2* prxvar1, double* slvvar, double2* slvvar1){
#ifdef USE_MPI
    const int tagmpi = 200;
using Parallel::mype;
    // Post receives for incoming messages from masters.
    // Store results in slave buffer.
    MPI_Request* request = Memory::alloc<MPI_Request>(nummstrpe);
    MPI_Request* request1 = Memory::alloc<MPI_Request>(nummstrpe);
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Irecv(&slvvar[slv1], nslv * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request[mstrpe]);
	MPI_Irecv(&slvvar1[slv1], nslv * sizeof(double2), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request1[mstrpe]);
//	printf("%d:scatter recving  from--> %d\n", mype, pe);
    }

    // Send updated slave data from proxy buffer back to slave PEs.
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];
        MPI_Send((void*)&prxvar[prx1], nprx * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
	MPI_Send((void*)&prxvar1[prx1], nprx * sizeof(double2), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
//	printf("%d:scatter sending to--> %d\n", mype, pe);
    }

    // Wait for all receives to complete.
    MPI_Status* status = Memory::alloc<MPI_Status>(nummstrpe);
    MPI_Status* status1 = Memory::alloc<MPI_Status>(nummstrpe);
    int ierr = MPI_Waitall(nummstrpe, &request[0], &status[0]);
    int ierr1 = MPI_Waitall(nummstrpe, &request1[0], &status1[0]);

    // Store slave data from buffer back to points.
    Memory::free(request);
    Memory::free(status);
     Memory::free(request1);
    Memory::free(status1);
#endif
}


