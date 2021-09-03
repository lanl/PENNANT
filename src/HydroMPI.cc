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
// TODO: check result codes of all MPI calls

#include "Parallel.hh"
#include "Memory.hh"
#include "HydroMPI.hh"

// TODO: wrap static heap objects in a unique_ptr with a custom deleter

void parallelGather(const int numslvpe, const int nummstrpe,
                    const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1,
                    const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1,
                    double* pmaswt_pf_proxy_buffer, double* pmaswt_pf_slave_buffer){

    using Parallel::numpe;
    using Parallel::mype;
    // This routine gathers slave values for which MYPE owns the masters.
    const int tagmpi = 100;
    // Post receives for incoming messages from slaves.
    // Store results in proxy buffer.
    static MPI_Request* request = Memory::alloc<MPI_Request>(numslvpe);
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];

        MPI_Irecv(&pmaswt_pf_proxy_buffer[prx1 * 3], nprx * 3 * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request[slvpe]);
    }
    // Send slave data to master PEs.
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Send(&pmaswt_pf_slave_buffer[slv1 * 3], nslv * 3 * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
    static MPI_Status* status = Memory::alloc<MPI_Status>(numslvpe);
    MPI_Waitall(numslvpe, &request[0], &status[0]);
}

void parallelScatter(const int numslvpe, const int nummstrpe,
                     const int *mapslvpepe, const int *slvpenumprx, const int *mapslvpeprx1,
                     const int *mapmstrpepe, const int *mstrpenumslv, const int *mapmstrpeslv1, const int *mapslvp,
                     double* pmaswt_pf_proxy_buffer, double* pmaswt_pf_slave_buffer){
    const int tagmpi = 200;
    using Parallel::mype;
    // Post receives for incoming messages from masters.
    // Store results in slave buffer.
    static MPI_Request* request = Memory::alloc<MPI_Request>(nummstrpe);
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Irecv(&pmaswt_pf_slave_buffer[slv1*3], nslv * 3 * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request[mstrpe]);
    }

    // Send updated slave data from proxy buffer back to slave PEs.
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];
        MPI_Send((void*)&pmaswt_pf_proxy_buffer[prx1*3], nprx * 3 * sizeof(double), MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
    static MPI_Status* status = Memory::alloc<MPI_Status>(nummstrpe);
    MPI_Waitall(nummstrpe, &request[0], &status[0]);
}

#endif // USE_MPI
