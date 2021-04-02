/*
 * Mesh.cc
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Triad National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Mesh.hh"

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
  gmesh(NULL), wxy(NULL), egold(NULL) {

    chunksize = inp->getInt("chunksize", 0);
    if (chunksize < 0) {
        cerr << "Error: bad chunksize " << chunksize << endl;
	exit(1);
    }
    subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    }

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

    // read mesh from gmv file
    vector<double2> nodepos;
    vector<int> cellstart, cellsize, cellnodes;
    vector<int> slavemstrpes, slavemstrcounts, slavepoints;
    vector<int> masterslvpes, masterslvcounts, masterpoints;
    gmesh->generate(nodepos, cellstart, cellsize, cellnodes,
            slavemstrpes, slavemstrcounts, slavepoints,
            masterslvpes, masterslvcounts, masterpoints);

    nump = nodepos.size();
    numz = cellstart.size();
    nums = cellnodes.size();

    // copy node positions to mesh, apply scaling factor
    px = Memory::alloc<double2>(nump);
    copy(nodepos.begin(), nodepos.end(), px);

    // copy cell sizes to mesh
    znump = Memory::alloc<int>(numz);
    copy(cellsize.begin(), cellsize.end(), znump);

    mapzs  = Memory::alloc<int>(numz);
    copy(cellstart.begin(), cellstart.end(), mapzs);

    // populate maps:
    // use the cell* arrays to populate the side maps
    initSides(cellstart, cellsize, cellnodes);
    // release memory from cell* arrays
    cellstart.resize(0);
    cellsize.resize(0);
    cellnodes.resize(0);
    // now populate other maps using side maps
    initEdges();
    initCorners();

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
    ex = Memory::alloc<double2>(nume);
    zx = Memory::alloc<double2>(numz);
    sarea = Memory::alloc<double>(nums);
    svol = Memory::alloc<double>(nums);
    carea = Memory::alloc<double>(numc);
    cvol = Memory::alloc<double>(numc);
    zarea = Memory::alloc<double>(numz);
    zvol = Memory::alloc<double>(numz);
    smf = Memory::alloc<double>(nums);

    // do a few initial calculations
    #pragma omp parallel for
    for (int ch = 0; ch < numsch; ++ch) {
        int sfirst = schsfirst[ch];
        int slast = schslast[ch];
        calcCtrs(px, ex, zx, sfirst, slast);
        calcVols(px, zx, sarea, svol, carea, cvol, zarea, zvol,
                sfirst, slast);
        calcSideFracs(sarea, zarea, smf, sfirst, slast);
    }

}


void Mesh::initSides(
        const std::vector<int>& cellstart,
        const std::vector<int>& cellsize,
        const std::vector<int>& cellnodes) {

    mapsp1 = Memory::alloc<int>(nums);
    mapsp2 = Memory::alloc<int>(nums);
    mapsz  = Memory::alloc<int>(nums);
    mapss3 = Memory::alloc<int>(nums);
    mapss4 = Memory::alloc<int>(nums);

    for (int z = 0; z < numz; ++z) {
        int sbase = cellstart[z];
        int size = cellsize[z];
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            int snext = sbase + (n + 1 == size ? 0 : n + 1);
            int slast = sbase + (n == 0 ? size : n) - 1;
            mapsz[s] = z;
            mapsp1[s] = cellnodes[s];
            mapsp2[s] = cellnodes[snext];
            mapss3[s] = slast;
            mapss4[s] = snext;
        } // for n
    } // for z

}


void Mesh::initEdges() {

    vector<vector<int> > edgepp(nump), edgepe(nump);

    mapse = Memory::alloc<int>(nums);
    // nums = upper bound for number of edges
    mapep1 = Memory::alloc<int>(nums);
    mapep2 = Memory::alloc<int>(nums);

    int e = 0;
    for (int s = 0; s < nums; ++s) {
        int p1 = min(mapsp1[s], mapsp2[s]);
        int p2 = max(mapsp1[s], mapsp2[s]);

        vector<int>& vpp = edgepp[p1];
        vector<int>& vpe = edgepe[p1];
        size_t i = find(vpp.begin(), vpp.end(), p2) - vpp.begin();
        if (i == vpp.size()) {
            // (p, p2) isn't in the edge list - add it
            vpp.push_back(p2);
            vpe.push_back(e);
            mapep1[e] = p1;
            mapep2[e] = p2;
            ++e;
        }
        mapse[s] = vpe[i];
    }  // for s

    nume = e;

}


void Mesh::initCorners() {

    numc = nums;

    mapcz = Memory::alloc<int>(numc);
    mapcp = Memory::alloc<int>(numc);
    mapsc1 = Memory::alloc<int>(nums);
    mapsc2 = Memory::alloc<int>(nums);

    for (int s = 0; s < nums; ++s) {
        int c = s;
        int c2 = mapss4[s];
        mapsc1[s] = c;
        mapsc2[s] = c2;
        mapcz[c] = mapsz[s];
        mapcp[c] = mapsp1[s];
    }

}


void Mesh::initChunks() {

    // check for bad chunksize
    if (chunksize <= 0) {
        cerr << "Error: bad chunksize " << chunksize << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s1, s2 = 0;
    while (s2 < nums) {
        s1 = s2;
        s2 = min(s2 + chunksize, nums);
        while (s2 < nums && mapsz[s2] == mapsz[s2-1])
            --s2;
        schsfirst.push_back(s1);
        schslast.push_back(s2);
        schzfirst.push_back(mapsz[s1]);
        schzlast.push_back(mapsz[s2-1] + 1);
    }
    numsch = schsfirst.size();

    // compute point chunks
    int p1, p2 = 0;
    while (p2 < nump) {
        p1 = p2;
        p2 = min(p2 + chunksize, nump);
        pchpfirst.push_back(p1);
        pchplast.push_back(p2);
    }
    numpch = pchpfirst.size();

/*     // compute zone chunks
    int z1, z2 = 0;
    while (z2 < numz) {
        z1 = z2;
        z2 = min(z2 + chunksize, numz);
        zchzfirst.push_back(z1);
        zchzlast.push_back(z2);
    }
    numzch = zchzfirst.size();
    */
}

void Mesh::initInvMap() {
    mappcfirst = Memory::alloc<int>(nump);
    mapccnext = Memory::alloc<int>(nums);

    vector<pair<int, int> > pcpair(nums);
    for (int c = 0; c < numc; ++c)
        pcpair[c] = make_pair(mapsp1[c], c);
    sort(pcpair.begin(), pcpair.end());
    for (int i = 0; i < numc; ++i) {
        int p = pcpair[i].first;
        int pp = pcpair[i+1].first;
        int pm = pcpair[i-1].first;
        int c = pcpair[i].second;
        int cp = pcpair[i+1].second;

        if (i == 0 || p != pm)  mappcfirst[p] = c;
        if (i+1 == numc || p != pp)
            mapccnext[c] = -1;
        else
            mapccnext[c] = cp;
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

    nummstrpe = slavemstrpes.size();
    mapmstrpepe = Memory::alloc<int>(nummstrpe);
    copy(slavemstrpes.begin(), slavemstrpes.end(), mapmstrpepe);
    mstrpenumslv = Memory::alloc<int>(nummstrpe);
    copy(slavemstrcounts.begin(), slavemstrcounts.end(), mstrpenumslv);
    mapmstrpeslv1 = Memory::alloc<int>(nummstrpe);
    int count = 0;
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        mapmstrpeslv1[mstrpe] = count;
        count += mstrpenumslv[mstrpe];
    }
    numslv = slavepoints.size();
    mapslvp = Memory::alloc<int>(numslv);
    copy(slavepoints.begin(), slavepoints.end(), mapslvp);

    numslvpe = masterslvpes.size();
    mapslvpepe = Memory::alloc<int>(numslvpe);
    copy(masterslvpes.begin(), masterslvpes.end(), mapslvpepe);
    slvpenumprx = Memory::alloc<int>(numslvpe);
    copy(masterslvcounts.begin(), masterslvcounts.end(), slvpenumprx);
    mapslvpeprx1 = Memory::alloc<int>(numslvpe);
    count = 0;
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        mapslvpeprx1[slvpe] = count;
        count += slvpenumprx[slvpe];
    }
    numprx = masterpoints.size();
    mapprxp = Memory::alloc<int>(numprx);
    copy(masterpoints.begin(), masterpoints.end(), mapprxp);
//    printf("%d:Mesh:nummstrpeH:%d numslvpeH:%d numprxH:%d numslvH:%d, \n",Parallel::mype, nummstrpe, numslvpe, numprx, numslv);
}

void Mesh::writeStats() {

    int gnump = nump;
    // make sure that boundary points aren't double-counted;
    // only count them if they are masters
    if (Parallel::numpe > 1) gnump -= numslv;
    gnumz = numz;
    int gnums = nums;
    int gnume = nume;
    int gnumpch = numpch;
    int gnumzch = numz;
    int gnumsch = numsch;

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

    wxy->write(probname, zr, ze, zp);
    egold->write(probname, cycle, time, zr, ze, zp);

}


void Mesh::calcCtrs(
        const double2* px,
        double2* ex,
        double2* zx,
        const int sfirst,
        const int slast) {

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&zx[zfirst], &zx[zlast], make_double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int e = mapse[s];
        int z = mapsz[s];
        ex[e] = 0.5 * (px[p1] + px[p2]);
        zx[z] += px[p1] / (double) znump[z];
    }

}


void Mesh::calcVols(
        const double2* px,
        const double2* zx,
        double* sarea,
        double* svol,
        double* carea,
        double* cvol,
        double* zarea,
        double* zvol,
        const int sfirst,
        const int slast) {

    int cfirst = sfirst;
    int clast = slast;
    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&cvol[cfirst], &cvol[clast], 0.);
    fill(&carea[cfirst], &carea[clast], 0.);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    int nserr = 0;

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int z = mapsz[s];

        // compute side volumes, sum to zone
        double sa = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
        double sv = sa * (px[p1].x + px[p2].x + zx[z].x) / 3.;
        sarea[s] = sa;
        svol[s] = sv;
        zarea[z] += sa;
        zvol[z] += sv;

        // check for negative side volumes
        if (sv <= 0.) nserr += 1;

        int c1 = mapsc1[s];
        int c2 = mapsc2[s];

        // sum side volumes to corners
        double hsa = 0.5 * sa;
        double ex = 0.5 * (px[p1].x + px[p2].x);
        double hsv1 = hsa * (px[p1].x + zx[z].x + ex) / 3.;
        double hsv2 = hsa * (px[p2].x + zx[z].x + ex) / 3.;
        carea[c1] += hsa;
        carea[c2] += hsa;
        cvol[c1] += hsv1;
        cvol[c2] += hsv2;

    } // for s

    // if there were negative side volumes, error exit
    if (nserr > 0) {
        cerr << "Error: " << nserr << " negative side volumes" << endl;
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

    for (int s = sfirst; s < slast; ++s) {
        int z = mapsz[s];
        smf[s] = sarea[s] / zarea[z];
    }
}

template <typename T>
void Mesh::parallelGather(
        const T* pvar, T* slvvar,
        T* prxvar) {
#ifdef USE_MPI
    // This routine gathers slave values for which MYPE owns the masters.
    const int tagmpi = 100;
    const int type_size = sizeof(T);
//    std::vector<T> slvvar(numslv);
/*    T* slvvar = Memory::alloc<T>(numslv);

// Load slave data buffer from points.
    for (int slv = 0; slv < numslv; ++slv) {
        int p = mapslvp[slv];
        slvvar[slv] = pvar[p];
    }

*/
    // Post receives for incoming messages from slaves.
    // Store results in proxy buffer.
//    vector<MPI_Request> request(numslvpe);
    MPI_Request* request = Memory::alloc<MPI_Request>(numslvpe);
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];
        MPI_Irecv(&prxvar[prx1], nprx * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD, &request[slvpe]);
    }

/*    // Load slave data buffer from points.
    for (int slv = 0; slv < numslv; ++slv) {
        int p = mapslvp[slv];
        slvvar[slv] = pvar[p];
    }
*/
    // Send slave data to master PEs.
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Send(&slvvar[slv1], nslv * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
//    vector<MPI_Status> status(numslvpe);
    MPI_Status* status = Memory::alloc<MPI_Status>(numslvpe);
    int ierr = MPI_Waitall(numslvpe, &request[0], &status[0]);
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
    for (int prx = 0; prx < numprx; ++prx) {
        int p = mapprxp[prx];
        pvar[p] += prxvar[prx];
    }

    // Copy updated master data back to proxies.
    for (int prx = 0; prx < numprx; ++prx) {
        int p = mapprxp[prx];
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
    T* slvvar = Memory::alloc<T>(numslv);

    // Post receives for incoming messages from masters.
    // Store results in slave buffer.
//    vector<MPI_Request> request(nummstrpe);
    MPI_Request* request = Memory::alloc<MPI_Request>(nummstrpe);
    for (int mstrpe = 0; mstrpe < nummstrpe; ++mstrpe) {
        int pe = mapmstrpepe[mstrpe];
        int nslv = mstrpenumslv[mstrpe];
        int slv1 = mapmstrpeslv1[mstrpe];
        MPI_Irecv(&slvvar[slv1], nslv * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD,  &request[mstrpe]);
    }

    // Send updated slave data from proxy buffer back to slave PEs.
    for (int slvpe = 0; slvpe < numslvpe; ++slvpe) {
        int pe = mapslvpepe[slvpe];
        int nprx = slvpenumprx[slvpe];
        int prx1 = mapslvpeprx1[slvpe];
        MPI_Send((void*)&prxvar[prx1], nprx * type_size, MPI_BYTE,
                pe, tagmpi, MPI_COMM_WORLD);
    }

    // Wait for all receives to complete.
//    vector<MPI_Status> status(nummstrpe);
    MPI_Status* status = Memory::alloc<MPI_Status>(nummstrpe);
    int ierr = MPI_Waitall(nummstrpe, &request[0], &status[0]);
    if (ierr != 0) {
        cerr << "Error: parallelScatter MPI error " << ierr <<
                " on PE " << Parallel::mype << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // Store slave data from buffer back to points.
    for (int slv = 0; slv < numslv; ++slv) {
        int p = mapslvp[slv];
        pvar[p] = slvvar[slv];
    }

    Memory::free(slvvar);
    Memory::free(request);
    Memory::free(status);
#endif
}


//template <typename T>
void Mesh::sumAcrossProcs(double* pvar) {
    if (Parallel::numpe == 1) return;
     double* slvvar = Memory::alloc<double>(numslv);
    for (int slv = 0; slv < numslv; ++slv) {
        int p = mapslvp[slv];
        slvvar[slv] = pvar[p];
    }

//    std::vector<T> prxvar(numprx);
    double* prxvar = Memory::alloc<double>(numprx);
    parallelGather(pvar, slvvar, &prxvar[0]);
    parallelSum(pvar, &prxvar[0]);
    parallelScatter(pvar, &prxvar[0]);
    Memory::free(prxvar);
}


//template <typename T>
void Mesh::sumAcrossProcs(double2* pvar) {
    if (Parallel::numpe == 1) return;
//    std::vector<T> prxvar(numprx);
    double2* prxvar = Memory::alloc<double2>(numprx);
    double2* slvvar = Memory::alloc<double2>(numslv);
    for (int slv = 0; slv < numslv; ++slv) {
        int p = mapslvp[slv];
        slvvar[slv] = pvar[p];
    }
    parallelGather(pvar, slvvar, &prxvar[0]);
    parallelSum(pvar, &prxvar[0]);
    parallelScatter(pvar, &prxvar[0]);
    Memory::free(prxvar);
}


/*template <typename T>
void Mesh::sumOnProc(
        const T* cvar,
        T* pvar) {

//    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = pchpfirst[pch];
        int plast = pchplast[pch];
        for (int p = pfirst; p < plast; ++p) {
            T x = T();
            for (int c = mappcfirst[p]; c >= 0; c = mapccnext[c]) {
                x += cvar[c];
            }
            pvar[p] = x;
        }  // for p
    }  // for pch

}
*/
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

/*	    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = pchpfirst[pch];
        int plast = pchplast[pch];
        for (int p = pfirst; p < plast; ++p) {
           double2 x = make_double2(0.,0.);
            for (int c = mappcfirst[p]; c >= 0; c = mapccnext[c]) {
                x += cvar[c];
            }
            pvar[p] = x;
        }  // for p
    }  // for pch
*/
    sumOnProc(cvar, pvar);
    if (Parallel::numpe > 1)
        sumAcrossProcs(pvar);

}




void Mesh::sumOnProc(
        const double* cvar,
        double* pvar) {

//    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = pchpfirst[pch];
        int plast = pchplast[pch];
        for (int p = pfirst; p < plast; ++p) {
            double x = 0.;
            for (int c = mappcfirst[p]; c >= 0; c = mapccnext[c]) {
                x += cvar[c];
            }
            pvar[p] = x;
        }  // for p
    }  // for pch

}



void Mesh::sumOnProc(
        const double2* cvar,
        double2* pvar) {

//    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = pchpfirst[pch];
        int plast = pchplast[pch];
        for (int p = pfirst; p < plast; ++p) {
            double2 x = make_double2(0.,0.);
            for (int c = mappcfirst[p]; c >= 0; c = mapccnext[c]) {
                x += cvar[c];
            }
            pvar[p] = x;
        }  // for p
    }  // for pch

}
