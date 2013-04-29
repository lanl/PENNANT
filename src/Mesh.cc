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

#include <cmath>
#include <iostream>
#include <algorithm>

#include "Vec2.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "ImportGMV.hh"
#include "ExportGold.hh"

using namespace std;


Mesh::Mesh(const InputFile* inp) {
    meshfile = inp->getString("meshfile", "");
    if (meshfile.empty()) {
        cerr << "Error:  must specify meshfile" << endl;
        exit(1);
    }
    meshscale = inp->getDouble("meshscale", 1.);
    chunksize = inp->getInt("chunksize", 99999999);
    subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    }

    init();
}


Mesh::~Mesh() {
    delete igmv;
    delete egold;
}


void Mesh::init() {

    igmv = new ImportGMV(this);
    egold = new ExportGold(this);

    // read mesh from gmv file
    vector<double2> nodepos;
    vector<int> cellstart, cellsize, cellnodes;
    igmv->read(meshfile, nodepos,
               cellstart, cellsize, cellnodes);

    nump = nodepos.size();
    numz = cellstart.size();
    nums = cellnodes.size();

    // copy node positions to mesh, apply scaling factor
    px = Memory::alloc<double2>(nump);
    for (int p = 0; p < nump; ++p)
        px[p] = nodepos[p] * meshscale;

    // copy cell sizes to mesh
    znump = Memory::alloc<int>(numz);
    copy(cellsize.begin(), cellsize.end(), znump);

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

    // write mesh statistics
    writeStats();

    // allocate remaining arrays
    ex = Memory::alloc<double2>(nume);
    zx = Memory::alloc<double2>(numz);
    px0 = Memory::alloc<double2>(nump);
    pxp = Memory::alloc<double2>(nump);
    exp = Memory::alloc<double2>(nume);
    zxp = Memory::alloc<double2>(numz);
    sarea = Memory::alloc<double>(nums);
    svol = Memory::alloc<double>(nums);
    zarea = Memory::alloc<double>(numz);
    zvol = Memory::alloc<double>(numz);
    sareap = Memory::alloc<double>(nums);
    svolp = Memory::alloc<double>(nums);
    zareap = Memory::alloc<double>(numz);
    zvolp = Memory::alloc<double>(numz);
    zvol0 = Memory::alloc<double>(numz);
    ssurfp = Memory::alloc<double2>(nums);
    elen = Memory::alloc<double>(nume);
    zdl = Memory::alloc<double>(numz);
    smf = Memory::alloc<double>(nums);

    // do a few initial calculations
    #pragma omp parallel for
    for (int ch = 0; ch < numsch; ++ch) {
        int sfirst = schsfirst[ch];
        int slast = schslast[ch];
        calcCtrs(px, ex, zx, sfirst, slast);
        calcVols(px, zx, sarea, svol, zarea, zvol, sfirst, slast);
        calcSideFracs(sarea, zarea, smf, sfirst, slast);
    }

}


void Mesh::initSides(
        std::vector<int>& cellstart,
        std::vector<int>& cellsize,
        std::vector<int>& cellnodes) {

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
        int i = find(vpp.begin(), vpp.end(), p2) - vpp.begin();
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

}


void Mesh::writeStats() {
    cout << "--- Mesh Information ---" << endl;
    cout << "Points:  " << nump << endl;
    cout << "Zones:  "  << numz << endl;
    cout << "Sides:  "  << nums << endl;
    cout << "Edges:  "  << nume << endl;
    cout << "Side chunks:  " << numsch << endl;
    cout << "Point chunks:  " << numpch << endl;
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

    egold->write(probname, cycle, time, zr, ze, zp);

}


vector<int> Mesh::getXPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < nump; ++p) {
        if (fabs(px[p].x - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


vector<int> Mesh::getYPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < nump; ++p) {
        if (fabs(px[p].y - c) < eps) {
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
    for (int pch = 0; pch < numpch; ++pch) {
         int pl = pchplast[pch];
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

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

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
        double* zarea,
        double* zvol,
        const int sfirst,
        const int slast) {

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
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


void Mesh::calcSurfVecs(
        const double2* zx,
        const double2* ex,
        double2* ssurf,
        const int sfirst,
        const int slast) {

    for (int s = sfirst; s < slast; ++s) {
        int z = mapsz[s];
        int e = mapse[s];

        ssurf[s] = rotateCCW(ex[e] - zx[z]);

    }

}


void Mesh::calcEdgeLen(
        const double2* px,
        double* elen,
        const int sfirst,
        const int slast) {

    for (int s = sfirst; s < slast; ++s) {
        const int p1 = mapsp1[s];
        const int p2 = mapsp2[s];
        const int e = mapse[s];

        elen[e] = length(px[p2] - px[p1]);

    }
}


void Mesh::calcCharLen(
        const double2* px,
        const double2* zx,
        double* zdl,
        const int sfirst,
        const int slast) {

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&zdl[zfirst], &zdl[zlast], 1.e99);

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int z = mapsz[s];

        double area = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
        double base = length(px[p2] - px[p1]);
        double fac = (znump[z] == 3 ? 3. : 4.);
        double sdl = fac * area / base;
        zdl[z] = min(zdl[z], sdl);
    }
}

