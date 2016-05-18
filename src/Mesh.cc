/*
 * Mesh.cc
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Mesh.hh"

#include <stdint.h>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "Vec2.hh"
#include "Memory.hh"
#include "Parallel.hh"
#include "InputFile.hh"
#include "GenMesh.hh"
#include "WriteXY.hh"
#include "ExportGold.hh"

using namespace std;


Mesh::Mesh(const InputFile* inp) :
    gmesh(NULL), egold(NULL), wxy(NULL) {

    using Parallel::mype;

    chunksize = inp->getInt("chunksize", 0);
    if (chunksize < 0) {
        if (mype == 0)
            cerr << "Error: bad chunksize " << chunksize << endl;
        exit(1);
    }

    subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        if (mype == 0)
            cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    }

    writexy = inp->getInt("writexy", 0);
    writegold = inp->getInt("writegold", 0);

    gmesh = new GenMesh(inp);
    wxy = new WriteXY(this);
    egold = new ExportGold(this);

    init();
}


Mesh::~Mesh() {
    delete gmesh;
    delete wxy;
    delete egold;
}


void Mesh::init() {

    // generate mesh
    vector<double2> nodepos;
    vector<int> cellstart, cellsize, cellnodes;
    vector<int> slavemstrpes, slavemstrcounts, slavepoints;
    vector<int> masterslvpes, masterslvcounts, masterpoints;
    gmesh->generate(nodepos, cellstart, cellsize, cellnodes,
            slavemstrpes, slavemstrcounts, slavepoints,
            masterslvpes, masterslvcounts, masterpoints);

    num_pts = nodepos.size();
    num_zones = cellstart.size();
    num_sides = cellnodes.size();
    num_corners = num_sides;

    // copy cell sizes to mesh
    znump = Memory::alloc<int>(num_zones);
    copy(cellsize.begin(), cellsize.end(), znump);

    // populate maps:
    // use the cell* arrays to populate the side maps
    initSides(cellstart, cellsize, cellnodes);
    // release memory from cell* arrays
    cellstart.resize(0);
    cellsize.resize(0);
    cellnodes.resize(0);
    // now populate edge maps using side maps
    initEdges();

    // populate chunk information
    initChunks();

    // create inverse map for corner-to-point gathers
    initInvMap();

    // calculate parallel data structures
    initParallel(slavemstrpes, slavemstrcounts, slavepoints,
            masterslvpes, masterslvcounts, masterpoints);
    // release memory from parallel-related arrays
    slavemstrpes.resize(0);
    slavemstrcounts.resize(0);
    slavepoints.resize(0);
    masterslvpes.resize(0);
    masterslvcounts.resize(0);
    masterpoints.resize(0);

    // write mesh statistics
    writeStats();

    // allocate remaining arrays
    pt_x = Memory::alloc<double2>(num_pts);
    edge_x = Memory::alloc<double2>(num_edges);
    zone_x = Memory::alloc<double2>(num_zones);
    pt_x0 = Memory::alloc<double2>(num_pts);
    pt_x_pred = Memory::alloc<double2>(num_pts);
    edge_x_pred = Memory::alloc<double2>(num_edges);
    zone_x_pred = Memory::alloc<double2>(num_zones);
    side_area = Memory::alloc<double>(num_sides);
    side_vol = Memory::alloc<double>(num_sides);
    zone_area = Memory::alloc<double>(num_zones);
    zone_vol = Memory::alloc<double>(num_zones);
    side_area_pred = Memory::alloc<double>(num_sides);
    side_vol_pred = Memory::alloc<double>(num_sides);
    zone_area_pred = Memory::alloc<double>(num_zones);
    zone_vol_pred = Memory::alloc<double>(num_zones);
    zone_vol0 = Memory::alloc<double>(num_zones);
    side_surfp = Memory::alloc<double2>(num_sides);
    edege_len = Memory::alloc<double>(num_edges);
    zone_dl = Memory::alloc<double>(num_zones);
    side_mass_frac = Memory::alloc<double>(num_sides);

    // do a few initial calculations
    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < num_pt_chunks; ++pch) {
        int pfirst = pt_chunks_first[pch];
        int plast = pt_chunks_last[pch];
        // copy nodepos into px, distributed across threads
        for (int p = pfirst; p < plast; ++p)
            pt_x[p] = nodepos[p];

    }

    num_bad_sides = 0;
    #pragma omp parallel for schedule(static)
    for (int sch = 0; sch < num_side_chunks; ++sch) {
        int sfirst = side_chunks_first[sch];
        int slast = side_chunks_last[sch];
        calcCtrs(pt_x, edge_x, zone_x, sfirst, slast);
        calcVols(pt_x, zone_x, side_area, side_vol, zone_area, zone_vol, sfirst, slast);
        calcSideFracs(side_area, zone_area, side_mass_frac, sfirst, slast);
    }
    checkBadSides();

}


void Mesh::initSides(
        const vector<int>& cellstart,
        const vector<int>& cellsize,
        const vector<int>& cellnodes) {

    map_side2pt1 = Memory::alloc<int>(num_sides);
    map_side2pt2 = Memory::alloc<int>(num_sides);
    map_side2zone  = Memory::alloc<int>(num_sides);
    maps_side_prev = Memory::alloc<int>(num_sides);
    maps_side_next = Memory::alloc<int>(num_sides);

    for (int z = 0; z < num_zones; ++z) {
        int sbase = cellstart[z];
        int size = cellsize[z];
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            int snext = sbase + (n + 1 == size ? 0 : n + 1);
            int slast = sbase + (n == 0 ? size : n) - 1;
            map_side2zone[s] = z;
            map_side2pt1[s] = cellnodes[s];
            map_side2pt2[s] = cellnodes[snext];
            maps_side_prev[s] = slast;
            maps_side_next[s] = snext;
        } // for n
    } // for z

}


void Mesh::initEdges() {

    vector<vector<int> > edgepp(num_pts), edgepe(num_pts);

    map_side2edge = Memory::alloc<int>(num_sides);

    int e = 0;
    for (int s = 0; s < num_sides; ++s) {
        int p1 = min(map_side2pt1[s], map_side2pt2[s]);
        int p2 = max(map_side2pt1[s], map_side2pt2[s]);

        vector<int>& vpp = edgepp[p1];
        vector<int>& vpe = edgepe[p1];
        int i = find(vpp.begin(), vpp.end(), p2) - vpp.begin();
        if (i == vpp.size()) {
            // (p, p2) isn't in the edge list - add it
            vpp.push_back(p2);
            vpe.push_back(e);
            ++e;
        }
        map_side2edge[s] = vpe[i];
    }  // for s

    num_edges = e;

}


void Mesh::initChunks() {

    if (chunksize == 0) chunksize = max(num_pts, num_sides);

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s1, s2 = 0;
    while (s2 < num_sides) {
        s1 = s2;
        s2 = min(s2 + chunksize, num_sides);
        while (s2 < num_sides && map_side2zone[s2] == map_side2zone[s2-1])
            --s2;
        side_chunks_first.push_back(s1);
        side_chunks_last.push_back(s2);
        zone_chunks_first.push_back(map_side2zone[s1]);
        zone_chunks_last.push_back(map_side2zone[s2-1] + 1);
    }
    num_side_chunks = side_chunks_first.size();

    // compute point chunks
    int p1, p2 = 0;
    while (p2 < num_pts) {
        p1 = p2;
        p2 = min(p2 + chunksize, num_pts);
        pt_chunks_first.push_back(p1);
        pt_chunks_last.push_back(p2);
    }
    num_pt_chunks = pt_chunks_first.size();

    // compute zone chunks
    int z1, z2 = 0;
    while (z2 < num_zones) {
        z1 = z2;
        z2 = min(z2 + chunksize, num_zones);
        zone_chunk_first.push_back(z1);
        zone_chunk_last.push_back(z2);
    }
    num_zone_chunks = zone_chunk_first.size();

}


void Mesh::initInvMap() {
    map_pt2crn_first = Memory::alloc<int>(num_pts);
    map_crn2crn_next = Memory::alloc<int>(num_sides);

    vector<pair<int, int> > pcpair(num_sides);
    for (int c = 0; c < num_corners; ++c)
        pcpair[c] = make_pair(map_side2pt1[c], c);
    sort(pcpair.begin(), pcpair.end());
    for (int i = 0; i < num_corners; ++i) {
        int p = pcpair[i].first;
        int pp = pcpair[i+1].first;
        int pm = pcpair[i-1].first;
        int c = pcpair[i].second;
        int cp = pcpair[i+1].second;

        if (i == 0 || p != pm)  map_pt2crn_first[p] = c;
        if (i+1 == num_corners || p != pp)
            map_crn2crn_next[c] = -1;
        else
            map_crn2crn_next[c] = cp;
    }

}


void Mesh::initParallel(
        const vector<int>& slavemstrpes,
        const vector<int>& slavemstrcounts,
        const vector<int>& slavepoints,
        const vector<int>& masterslvpes,
        const vector<int>& masterslvcounts,
        const vector<int>& masterpoints) {
    if (Parallel::numpe == 1) return;

    num_mesg_send2master = slavemstrpes.size();
    map_master_pe2globale_pe = Memory::alloc<int>(num_mesg_send2master);
    copy(slavemstrpes.begin(), slavemstrpes.end(), map_master_pe2globale_pe);
    master_pe_num_slaves = Memory::alloc<int>(num_mesg_send2master);
    copy(slavemstrcounts.begin(), slavemstrcounts.end(), master_pe_num_slaves);
    map_master_pe2slave1 = Memory::alloc<int>(num_mesg_send2master);
    int count = 0;
    for (int mstrpe = 0; mstrpe < num_mesg_send2master; ++mstrpe) {
        map_master_pe2slave1[mstrpe] = count;
        count += master_pe_num_slaves[mstrpe];
    }
    num_slaves = slavepoints.size();
    map_slave2pt = Memory::alloc<int>(num_slaves);
    copy(slavepoints.begin(), slavepoints.end(), map_slave2pt);

    num_slave_pes = masterslvpes.size();
    map_slave_pe2global_pe = Memory::alloc<int>(num_slave_pes);
    copy(masterslvpes.begin(), masterslvpes.end(), map_slave_pe2global_pe);
    slave_pe_num_prox = Memory::alloc<int>(num_slave_pes);
    copy(masterslvcounts.begin(), masterslvcounts.end(), slave_pe_num_prox);
    map_slave_pe2prox1 = Memory::alloc<int>(num_slave_pes);
    count = 0;
    for (int slvpe = 0; slvpe < num_slave_pes; ++slvpe) {
        map_slave_pe2prox1[slvpe] = count;
        count += slave_pe_num_prox[slvpe];
    }
    num_proxies = masterpoints.size();
    map_prox2master_pt = Memory::alloc<int>(num_proxies);
    copy(masterpoints.begin(), masterpoints.end(), map_prox2master_pt);

}


void Mesh::writeStats() {

    int64_t gnump = num_pts;
    // make sure that boundary points aren't double-counted;
    // only count them if they are masters
    if (Parallel::numpe > 1) gnump -= num_slaves;
    int64_t gnumz = num_zones;
    int64_t gnums = num_sides;
    int64_t gnume = num_edges;
    int gnumpch = num_pt_chunks;
    int gnumzch = num_zone_chunks;
    int gnumsch = num_side_chunks;

    Parallel::globalSum(gnump);
    Parallel::globalSum(gnumz);
    Parallel::globalSum(gnums);
    Parallel::globalSum(gnume);
    Parallel::globalSum(gnumpch);
    Parallel::globalSum(gnumzch);
    Parallel::globalSum(gnumsch);

    if (Parallel::mype > 0) return;

    cout << "--- Mesh Information ---" << endl;
    cout << "Points:  " << gnump << endl;
    cout << "Zones:  "  << gnumz << endl;
    cout << "Sides:  "  << gnums << endl;
    cout << "Edges:  "  << gnume << endl;
    cout << "Side chunks:  " << gnumsch << endl;
    cout << "Point chunks:  " << gnumpch << endl;
    cout << "Zone chunks:  " << gnumzch << endl;
    cout << "Chunk size:  " << chunksize << endl;
    cout << "------------------------" << endl;

}


void Mesh::write(
        const string& probname,
        const int cycle,
        const double time,
        const double* zr,
        const double* ze,
        const double* zp) {

    if (writexy) {
        if (Parallel::mype == 0)
            cout << "Writing .xy file..." << endl;
        wxy->write(probname, zr, ze, zp);
    }
    if (writegold) {
        if (Parallel::mype == 0) 
            cout << "Writing gold file..." << endl;
        egold->write(probname, cycle, time, zr, ze, zp);
    }

}


vector<int> Mesh::getXPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts; ++p) {
        if (fabs(pt_x[p].x - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


vector<int> Mesh::getYPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts; ++p) {
        if (fabs(pt_x[p].y - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


void Mesh::getPlaneChunks(
        const int numb,
        const int* mapbp,
        vector<int>& pchbfirst,
        vector<int>& pchblast) {

    pchbfirst.resize(0);
    pchblast.resize(0);

    // compute boundary point chunks
    // (boundary points contained in each point chunk)
    int bf, bl = 0;
    for (int pch = 0; pch < num_pt_chunks; ++pch) {
         int pl = pt_chunks_last[pch];
         bf = bl;
         bl = lower_bound(&mapbp[bf], &mapbp[numb], pl) - &mapbp[0];
         pchbfirst.push_back(bf);
         pchblast.push_back(bl);
    }

}


void Mesh::calcCtrs(
        const double2* px,
        double2* ex,
        double2* zx,
        const int sfirst,
        const int slast) {

    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1[s];
        int p2 = map_side2pt2[s];
        int e = map_side2edge[s];
        int z = map_side2zone[s];
        ex[e] = 0.5 * (px[p1] + px[p2]);
        zx[z] += px[p1];
    }

    for (int z = zfirst; z < zlast; ++z) {
        zx[z] /= (double) znump[z];
    }

}


void Mesh::calcVols(
        const double2* px,
        const double2* zx,
        double* sarea,
        double* svol,
        double* zarea,
        double* zvol,
        const int sfirst,
        const int slast) {

    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    const double third = 1. / 3.;
    int count = 0;
    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1[s];
        int p2 = map_side2pt2[s];
        int z = map_side2zone[s];

        // compute side volumes, sum to zone
        double sa = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
        double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
        sarea[s] = sa;
        svol[s] = sv;
        zarea[z] += sa;
        zvol[z] += sv;

        // check for negative side volumes
        if (sv <= 0.) count += 1;

    } // for s

    if (count > 0) {
        #pragma omp atomic
        num_bad_sides += count;
    }

}


void Mesh::checkBadSides() {

    // if there were negative side volumes, error exit
    if (num_bad_sides > 0) {
        cerr << "Error: " << num_bad_sides << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


void Mesh::calcSideFracs(
        const double* sarea,
        const double* zarea,
        double* smf,
        const int sfirst,
        const int slast) {

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        smf[s] = sarea[s] / zarea[z];
    }
}


void Mesh::calcSurfVecs(
        const double2* zx,
        const double2* ex,
        double2* ssurf,
        const int sfirst,
        const int slast) {

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        int e = map_side2edge[s];

        ssurf[s] = rotateCCW(ex[e] - zx[z]);

    }

}


void Mesh::calcEdgeLen(
        const double2* px,
        double* elen,
        const int sfirst,
        const int slast) {

    for (int s = sfirst; s < slast; ++s) {
        const int p1 = map_side2pt1[s];
        const int p2 = map_side2pt2[s];
        const int e = map_side2edge[s];

        elen[e] = length(px[p2] - px[p1]);

    }
}


void Mesh::calcCharLen(
        const double* sarea,
        double* zdl,
        const int sfirst,
        const int slast) {

    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zdl[zfirst], &zdl[zlast], 1.e99);

    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        int e = map_side2edge[s];

        double area = sarea[s];
        double base = edege_len[e];
        double fac = (znump[z] == 3 ? 3. : 4.);
        double sdl = fac * area / base;
        zdl[z] = min(zdl[z], sdl);
    }
}


template <typename T>
void Mesh::parallelGather(
        const T* pvar,
        T* prxvar) {
#ifdef USE_MPI
    // This routine gathers slave values for which MYPE owns the masters.
    const int tagmpi = 100;
    const int type_size = sizeof(T);
//    std::vector<T> slvvar(numslv);
    T* slvvar = Memory::alloc<T>(num_slaves);

    // Post receives for incoming messages from slaves.
    // Store results in proxy buffer.
//    vector<MPI_Request> request(numslvpe);
    MPI_Request* request = Memory::alloc<MPI_Request>(num_slave_pes);
    for (int slvpe = 0; slvpe < num_slave_pes; ++slvpe) {
        int pe = map_slave_pe2global_pe[slvpe];
        int nprx = slave_pe_num_prox[slvpe];
        int prx1 = map_slave_pe2prox1[slvpe];
        MPI_Irecv(&prxvar[prx1], nprx * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request[slvpe]);
    }

    // Load slave data buffer from points.
    for (int slv = 0; slv < num_slaves; ++slv) {
        int p = map_slave2pt[slv];
        slvvar[slv] = pvar[p];
    }

    // Send slave data to master PEs.
    for (int mstrpe = 0; mstrpe < num_mesg_send2master; ++mstrpe) {
        int pe = map_master_pe2globale_pe[mstrpe];
        int nslv = master_pe_num_slaves[mstrpe];
        int slv1 = map_master_pe2slave1[mstrpe];
        MPI_Send(&slvvar[slv1], nslv * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
//    vector<MPI_Status> status(numslvpe);
    MPI_Status* status = Memory::alloc<MPI_Status>(num_slave_pes);
    int ierr = MPI_Waitall(num_slave_pes, &request[0], &status[0]);
    if (ierr != 0) {
        cerr << "Error: parallelGather MPI error " << ierr <<
                " on PE " << Parallel::mype << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    Memory::free(slvvar);
    Memory::free(request);
    Memory::free(status);
#endif
}


template <typename T>
void Mesh::parallelSum(
        T* pvar,
        T* prxvar) {
#ifdef USE_MPI
    // Compute sum of all (proxy/master) sets.
    // Store results in master.
    for (int prx = 0; prx < num_proxies; ++prx) {
        int p = map_prox2master_pt[prx];
        pvar[p] += prxvar[prx];
    }

    // Copy updated master data back to proxies.
    for (int prx = 0; prx < num_proxies; ++prx) {
        int p = map_prox2master_pt[prx];
        prxvar[prx] = pvar[p];
    }
#endif
}


template <typename T>
void Mesh::parallelScatter(
        T* pvar,
        const T* prxvar) {
#ifdef USE_MPI
    // This routine scatters master values on MYPE to all slave copies
    // owned by other PEs.
    const int tagmpi = 200;
    const int type_size = sizeof(T);
//    std::vector<T> slvvar(numslv);
    T* slvvar = Memory::alloc<T>(num_slaves);

    // Post receives for incoming messages from masters.
    // Store results in slave buffer.
//    vector<MPI_Request> request(nummstrpe);
    MPI_Request* request = Memory::alloc<MPI_Request>(num_mesg_send2master);
    for (int mstrpe = 0; mstrpe < num_mesg_send2master; ++mstrpe) {
        int pe = map_master_pe2globale_pe[mstrpe];
        int nslv = master_pe_num_slaves[mstrpe];
        int slv1 = map_master_pe2slave1[mstrpe];
        MPI_Irecv(&slvvar[slv1], nslv * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request[mstrpe]);
    }

    // Send updated slave data from proxy buffer back to slave PEs.
    for (int slvpe = 0; slvpe < num_slave_pes; ++slvpe) {
        int pe = map_slave_pe2global_pe[slvpe];
        int nprx = slave_pe_num_prox[slvpe];
        int prx1 = map_slave_pe2prox1[slvpe];
        MPI_Send((void*)&prxvar[prx1], nprx * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
//    vector<MPI_Status> status(nummstrpe);
    MPI_Status* status = Memory::alloc<MPI_Status>(num_mesg_send2master);
    int ierr = MPI_Waitall(num_mesg_send2master, &request[0], &status[0]);
    if (ierr != 0) {
        cerr << "Error: parallelScatter MPI error " << ierr <<
                " on PE " << Parallel::mype << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // Store slave data from buffer back to points.
    for (int slv = 0; slv < num_slaves; ++slv) {
        int p = map_slave2pt[slv];
        pvar[p] = slvvar[slv];
    }

    Memory::free(slvvar);
    Memory::free(request);
    Memory::free(status);
#endif
}


template <typename T>
void Mesh::sumAcrossProcs(T* pvar) {
    if (Parallel::numpe == 1) return;
//    std::vector<T> prxvar(numprx);
    T* prxvar = Memory::alloc<T>(num_proxies);
    parallelGather(pvar, &prxvar[0]);
    parallelSum(pvar, &prxvar[0]);
    parallelScatter(pvar, &prxvar[0]);
    Memory::free(prxvar);
}


template <typename T>
void Mesh::sumOnProc(
        const T* cvar,
        T* pvar) {

    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < num_pt_chunks; ++pch) {
        int pfirst = pt_chunks_first[pch];
        int plast = pt_chunks_last[pch];
        for (int p = pfirst; p < plast; ++p) {
            T x = T();
            for (int c = map_pt2crn_first[p]; c >= 0; c = map_crn2crn_next[c]) {
                x += cvar[c];
            }
            pvar[p] = x;
        }  // for p
    }  // for pch

}


template <>
void Mesh::sumToPoints(
        const double* cvar,
        double* pvar) {

    sumOnProc(cvar, pvar);
    if (Parallel::numpe > 1)
        sumAcrossProcs(pvar);

}


template <>
void Mesh::sumToPoints(
        const double2* cvar,
        double2* pvar) {

    sumOnProc(cvar, pvar);
    if (Parallel::numpe > 1)
        sumAcrossProcs(pvar);

}

