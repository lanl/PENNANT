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

#ifdef USE_MPI

#include "Parallel.hh"
#include "Memory.hh"
#include "HydroMPI.hh"

// TODO: check if the two functions below try to send from device buffers, or if they use host buffers
// In the former case, figure out if that requires a specific MPI stack and/or UCX support

void parallelGather(const int numslvpe, const int nummstrpe,
		    const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1, 
		    const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1,	
        	    double* pmaswt_proxy_buffer, double2* pf_proxy_buffer,
		    double *pmaswt_slave_buffer, double2 *pf_slave_buffer) {

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

        MPI_Irecv(&pmaswt_proxy_buffer[prx1], nprx * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request[slvpe]);

	MPI_Irecv(&pf_proxy_buffer[prx1], nprx * sizeof(double2), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request1[slvpe]);
    }
    // Send slave data to master PEs.
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Send(&pmaswt_slave_buffer[slv1], nslv * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
	MPI_Send(&pf_slave_buffer[slv1], nslv * sizeof(double2), MPI_BYTE,
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
}


void parallelScatter(const int numslvpe, const int nummstrpe,
		     const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1,
		     const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1, const int *mapslvp,
		     double* pmaswt_proxy_buffer, double2* pf_proxy_buffer,
		     double* pmaswt_slave_buffer, double2* pf_slave_buffer){
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
        MPI_Irecv(&pmaswt_slave_buffer[slv1], nslv * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request[mstrpe]);
	MPI_Irecv(&pf_slave_buffer[slv1], nslv * sizeof(double2), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request1[mstrpe]);
//	printf("%d:scatter recving  from--> %d\n", mype, pe);
    }

    // Send updated slave data from proxy buffer back to slave PEs.
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];
        MPI_Send((void*)&pmaswt_proxy_buffer[prx1], nprx * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
	MPI_Send((void*)&pf_proxy_buffer[prx1], nprx * sizeof(double2), MPI_BYTE,
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
}

#endif // USE_MPI
