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
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "Vec2.hh"
#include "Memory.hh"
#include "GenerateMesh.hh"

using namespace std;


LocalMesh::LocalMesh(const InputParameters& params,
		LogicalUnstructured& points,
		const PhysicalRegion& ghost_pts,
		Context ctx, HighLevelRuntime* rt) :
			chunk_size(params.directs_.chunk_size_),
			subregion_xmin(params.directs_.subregion_xmin_),
			subregion_xmax(params.directs_.subregion_xmax_),
			subregion_ymin(params.directs_.subregion_ymin_),
			subregion_ymax(params.directs_.subregion_ymax_),
            local_points_by_gid(ctx, rt, points.getISpace()),
            pt_x_init_by_gid(points),
			generate_mesh(NULL),
			ctx(ctx),
			runtime(rt),
			ghost_points(ghost_pts),
			num_subregions(params.directs_.ntasks_),
			my_PE(params.directs_.task_id_)
	{

    generate_mesh = new GenerateLocalMesh(params);

    local_points_by_gid.addField<double>(FID_PMASWT);
    local_points_by_gid.addField<double2>(FID_PF);
    local_points_by_gid.allocate();

	init();
}


LocalMesh::~LocalMesh() {
    delete generate_mesh;
}


void LocalMesh::init() {

    // generate mesh
    std::vector<double2> nodepos;  // TODO pointpos
    std::vector<int> cellstart;  // TODO zone_pts_CRS_val
    std::vector<int> cellnodes;  // TODO zone_pts_CRS_ptr
    vector<int> slavemstrpes, slavemstrcounts, slavepoints;
    vector<int> masterslvpes, masterslvcounts, masterpoints;
    generate_mesh->generate(nodepos, cellstart, cellnodes,
            slavemstrpes, slavemstrcounts, slavepoints,
            masterslvpes, masterslvcounts, masterpoints);

    {
		IndexIterator pt_itr = pt_x_init_by_gid.getIterator();
		size_t npts;
		pt_itr.next_span(npts);
		num_pts_ = (int) npts;
		assert(num_pts_ == nodepos.size());
    }

		num_sides_ = cellnodes.size();
	    num_corners_ = num_sides_;

		num_zones_ = cellstart.size() - 1;

    // copy cell sizes to mesh

    std::cout << "Task: " << my_PE << " zones: " << num_zones_ << std::endl;
    zone_pts_ptr_ = AbstractedMemory::alloc<int>(num_zones_+1);
    copy(cellstart.begin(), cellstart.end(), zone_pts_ptr_);

    // populate maps:
    // use the cell* arrays to populate the side maps
    initSideMappingArrays(cellstart, cellnodes);

    // release memory from cell* arrays
    cellstart.resize(0);
    cellnodes.resize(0);

    // now populate edge maps using side maps
    initEdgeMappingArrays();

    // populate chunk information
    populateChunks();

    // create inverse map for corner-to-point gathers
    populateInverseMap();

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
    writeMeshStats();

    // allocate remaining arrays
    edge_x = AbstractedMemory::alloc<double2>(num_edges_);
    zone_x_ = AbstractedMemory::alloc<double2>(num_zones_);
    pt_x0 = AbstractedMemory::alloc<double2>(num_pts_);
    pt_x = AbstractedMemory::alloc<double2>(num_pts_);
    pt_x_pred = AbstractedMemory::alloc<double2>(num_pts_);
    edge_x_pred = AbstractedMemory::alloc<double2>(num_edges_);
    zone_x_pred = AbstractedMemory::alloc<double2>(num_zones_);
    side_area_ = AbstractedMemory::alloc<double>(num_sides_);
    side_vol_ = AbstractedMemory::alloc<double>(num_sides_);
    zone_area_ = AbstractedMemory::alloc<double>(num_zones_);
    zone_vol_ = AbstractedMemory::alloc<double>(num_zones_);
    side_area_pred = AbstractedMemory::alloc<double>(num_sides_);
    side_vol_pred = AbstractedMemory::alloc<double>(num_sides_);
    zone_area_pred = AbstractedMemory::alloc<double>(num_zones_);
    zone_vol_pred = AbstractedMemory::alloc<double>(num_zones_);
    zone_vol0 = AbstractedMemory::alloc<double>(num_zones_);
    side_surfp = AbstractedMemory::alloc<double2>(num_sides_);
    edge_len = AbstractedMemory::alloc<double>(num_edges_);
    zone_dl = AbstractedMemory::alloc<double>(num_zones_);
    side_mass_frac = AbstractedMemory::alloc<double>(num_sides_);

    IndexIterator itr = pt_x_init_by_gid.getIterator();
    Double2Accessor x_init_acc = pt_x_init_by_gid.getRegionAccessor<double2>(FID_PX_INIT);
    point_local_to_globalID = AbstractedMemory::alloc<long long int>(num_pts_);
    int i = 0;
    while (itr.has_next()) {
    		ptr_t pt_ptr = itr.next();
        pt_x[i] = x_init_acc.read(pt_ptr);
        assert(x_init_acc.read(pt_ptr) == nodepos[i]);
        point_local_to_globalID[i] = pt_ptr.value;
        i++;
    }

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
    for (int sch = 0; sch < num_side_chunks; ++sch) {
        //int sfirst = side_chunks_first[sch];
        //int slast = side_chunks_last[sch];
        calcCtrs(sch, false);
        calcVols(sch, false);
        calcSideMassFracs(sch);
    }
    checkBadSides();

}


void LocalMesh::initSideMappingArrays(
        const vector<int>& cellstart,
        const vector<int>& cellnodes) {

    map_side2pt1_ = AbstractedMemory::alloc<int>(num_sides_);
    zone_pts_val_ = map_side2pt1_;
    map_side2zone_  = AbstractedMemory::alloc<int>(num_sides_);

    for (int z = 0; z < num_zones_; ++z) {
        int sbase = cellstart[z];
        int size = cellstart[z+1] - sbase;
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            map_side2zone_[s] = z;
            map_side2pt1_[s] = cellnodes[s];
        } // for n
    } // for z
}


void LocalMesh::initEdgeMappingArrays() {

    vector<vector<int> > edgepp(num_pts_), edgepe(num_pts_);

    map_side2edge_ = AbstractedMemory::alloc<int>(num_sides_);

    int e = 0;
    for (int s = 0; s < num_sides_; ++s) {
        int p1 = min(map_side2pt1_[s], mapSideToPt2(s));
        int p2 = max(map_side2pt1_[s], mapSideToPt2(s));

        vector<int>& vpp = edgepp[p1];
        vector<int>& vpe = edgepe[p1];
        int i = find(vpp.begin(), vpp.end(), p2) - vpp.begin();
        if (i == vpp.size()) {
            // (p, p2) isn't in the edge list - add it
            vpp.push_back(p2);
            vpe.push_back(e);
            ++e;
        }
        map_side2edge_[s] = vpe[i];
    }  // for s

    num_edges_ = e;

}


void LocalMesh::populateChunks() {

    if (chunk_size == 0) chunk_size = max(num_pts_, num_sides_);

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s1, s2 = 0;
    while (s2 < num_sides_) {
        s1 = s2;
        s2 = min(s2 + chunk_size, num_sides_);
        while (s2 < num_sides_ && map_side2zone_[s2] == map_side2zone_[s2-1])
            --s2;
        side_chunks_first.push_back(s1);
        side_chunks_last.push_back(s2);
        zone_chunks_first.push_back(map_side2zone_[s1]);
        zone_chunks_last.push_back(map_side2zone_[s2-1] + 1);
    }
    num_side_chunks = side_chunks_first.size();

    // compute point chunks
    int p1, p2 = 0;
    while (p2 < num_pts_) {
        p1 = p2;
        p2 = min(p2 + chunk_size, num_pts_);
        pt_chunks_first.push_back(p1);
        pt_chunks_last.push_back(p2);
    }
    num_pt_chunks = pt_chunks_first.size();

    // compute zone chunks
    int z1, z2 = 0;
    while (z2 < num_zones_) {
        z1 = z2;
        z2 = min(z2 + chunk_size, num_zones_);
        zone_chunk_first.push_back(z1);
        zone_chunk_last.push_back(z2);
    }
    num_zone_chunks = zone_chunk_first.size();

}


void LocalMesh::populateInverseMap() {
    map_pt2crn_first = AbstractedMemory::alloc<int>(num_pts_);
    map_crn2crn_next = AbstractedMemory::alloc<int>(num_sides_);

    vector<pair<int, int> > pcpair(num_sides_);
    for (int c = 0; c < num_corners_; ++c)
        pcpair[c] = make_pair(map_side2pt1_[c], c);
    sort(pcpair.begin(), pcpair.end());
    for (int i = 0; i < num_corners_; ++i) {
        int p = pcpair[i].first;
        int pp = pcpair[i+1].first;
        int pm = pcpair[i-1].first;
        int c = pcpair[i].second;
        int cp = pcpair[i+1].second;

        if (i == 0 || p != pm)  map_pt2crn_first[p] = c;
        if (i+1 == num_corners_ || p != pp)
            map_crn2crn_next[c] = -1;
        else
            map_crn2crn_next[c] = cp;
    }

}


void LocalMesh::initParallel(
        const vector<int>& slavemstrpes,
        const vector<int>& slavemstrcounts,
        const vector<int>& slavepoints,
        const vector<int>& masterslvpes,
        const vector<int>& masterslvcounts,
        const vector<int>& masterpoints) {
    if (num_subregions == 1) return;

    num_mesg_send2master = slavemstrpes.size();
    map_master_pe2globale_pe = AbstractedMemory::alloc<int>(num_mesg_send2master);
    copy(slavemstrpes.begin(), slavemstrpes.end(), map_master_pe2globale_pe);
    master_pe_num_slaves = AbstractedMemory::alloc<int>(num_mesg_send2master);
    copy(slavemstrcounts.begin(), slavemstrcounts.end(), master_pe_num_slaves);
    map_master_pe2slave1 = AbstractedMemory::alloc<int>(num_mesg_send2master);
    int count = 0;
    for (int mstrpe = 0; mstrpe < num_mesg_send2master; ++mstrpe) {
        map_master_pe2slave1[mstrpe] = count;
        count += master_pe_num_slaves[mstrpe];
    }
    num_slaves = slavepoints.size();
    map_slave2pt = AbstractedMemory::alloc<int>(num_slaves);
    copy(slavepoints.begin(), slavepoints.end(), map_slave2pt);

    num_slave_pes = masterslvpes.size();
    map_slave_pe2global_pe = AbstractedMemory::alloc<int>(num_slave_pes);
    copy(masterslvpes.begin(), masterslvpes.end(), map_slave_pe2global_pe);
    slave_pe_num_prox = AbstractedMemory::alloc<int>(num_slave_pes);
    copy(masterslvcounts.begin(), masterslvcounts.end(), slave_pe_num_prox);
    map_slave_pe2prox1 = AbstractedMemory::alloc<int>(num_slave_pes);
    count = 0;
    for (int slvpe = 0; slvpe < num_slave_pes; ++slvpe) {
        map_slave_pe2prox1[slvpe] = count;
        count += slave_pe_num_prox[slvpe];
    }
    num_proxies = masterpoints.size();
    map_prox2master_pt = AbstractedMemory::alloc<int>(num_proxies);
    copy(masterpoints.begin(), masterpoints.end(), map_prox2master_pt);

}


void LocalMesh::writeMeshStats() {

    int64_t gnump = num_pts_;
    // make sure that boundary points aren't double-counted;
    // only count them if they are masters
    if (num_subregions > 1) gnump -= num_slaves;
    int64_t gnumz = num_zones_;
    int64_t gnums = num_sides_;
    int64_t gnume = num_edges_;
    int gnumpch = num_pt_chunks;
    int gnumzch = num_zone_chunks;
    int gnumsch = num_side_chunks;

    // TODO use Legion
    Parallel::globalSum(gnump);
    Parallel::globalSum(gnumz);
    Parallel::globalSum(gnums);
    Parallel::globalSum(gnume);
    Parallel::globalSum(gnumpch);
    Parallel::globalSum(gnumzch);
    Parallel::globalSum(gnumsch);

    if (my_PE > 0) return;

    cout << "--- Mesh Information ---" << endl;
    cout << "Points:  " << gnump << endl;
    cout << "Zones:  "  << gnumz << endl;
    cout << "Sides:  "  << gnums << endl;
    cout << "Edges:  "  << gnume << endl;
    cout << "Side chunks:  " << gnumsch << endl;
    cout << "Point chunks:  " << gnumpch << endl;
    cout << "Zone chunks:  " << gnumzch << endl;
    cout << "Chunk size:  " << chunk_size << endl;
    cout << "------------------------" << endl;

}




vector<int> LocalMesh::getXPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts_; ++p) {
    		if (fabs(pt_x[p].x - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


vector<int> LocalMesh::getYPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts_; ++p) {
        if (fabs(pt_x[p].y - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


void LocalMesh::getPlaneChunks(
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


void LocalMesh::calcCtrs(const int side_chunk, const bool pred) {
    const double2* px;
    double2* ex;
    double2* zx;

    if (pred) {
        px = pt_x_pred;
        ex = edge_x_pred;
        zx = zone_x_pred;
    } else {
        px = pt_x;
        ex = edge_x;
        zx = zone_x_;
    }
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    int zfirst = map_side2zone_[sfirst];
    int zlast = (slast < num_sides_ ? map_side2zone_[slast] : num_zones_);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1_[s];
        int p2 = mapSideToPt2(s);
        int e = map_side2edge_[s];
        int z = map_side2zone_[s];
        ex[e] = 0.5 * (px[p1] + px[p2]);
        zx[z] += px[p1];
    }

    for (int z = zfirst; z < zlast; ++z) {
        zx[z] /= (double) zoneNPts(z);
    }

}


void LocalMesh::calcVols(const int side_chunk, const bool pred) {
    const double2* px;
    const double2* zx;
    double* sarea;
    double* svol;
    double* zarea;
    double* zvol;

    if (pred) {
        px = pt_x_pred;
        zx = zone_x_pred;
        sarea = side_area_pred;
        svol = side_vol_pred;
        zarea = zone_area_pred;
        zvol = zone_vol_pred;
    } else {
        px = pt_x;
        zx = zone_x_;
        sarea = side_area_;
        svol = side_vol_;
        zarea = zone_area_;
        zvol = zone_vol_;
    }
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    int zfirst = map_side2zone_[sfirst];
    int zlast = (slast < num_sides_ ? map_side2zone_[slast] : num_zones_);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    const double third = 1. / 3.;
    int count = 0;
    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1_[s];
        int p2 = mapSideToPt2(s);
        int z = map_side2zone_[s];

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
        num_bad_sides += count;
    }

}


void LocalMesh::checkBadSides() {

    // if there were negative side volumes, error exit
    if (num_bad_sides > 0) {
        cerr << "Error: " << num_bad_sides << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}

void LocalMesh::calcSideMassFracs(const int side_chunk) {
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone_[s];
        side_mass_frac[s] = side_area_[s] / zone_area_[z];
    }
}


void LocalMesh::calcMedianMeshSurfVecs(const int side_chunk) {
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone_[s];
        int e = map_side2edge_[s];

        side_surfp[s] = rotateCCW(edge_x_pred[e] - zone_x_pred[z]);

    }

}


void LocalMesh::calcEdgeLen(const int side_chunk) {
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

	for (int s = sfirst; s < slast; ++s) {
        const int p1 = map_side2pt1_[s];
        const int p2 = mapSideToPt2(s);
        const int e = map_side2edge_[s];

        edge_len[e] = length(pt_x_pred[p2] - pt_x_pred[p1]);

    }
}


void LocalMesh::calcCharacteristicLen(const int side_chunk) {
    int sfirst = side_chunks_first[side_chunk];
    int slast = side_chunks_last[side_chunk];

    int zfirst = map_side2zone_[sfirst];
    int zlast = (slast < num_sides_ ? map_side2zone_[slast] : num_zones_);
    fill(&zone_dl[zfirst], &zone_dl[zlast], 1.e99);

    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone_[s];
        int e = map_side2edge_[s];

        double area = side_area_pred[s];
        double base = edge_len[e];
        double fac = (zoneNPts(z) == 3 ? 3. : 4.);
        double sdl = fac * area / base;
        zone_dl[z] = min(zone_dl[z], sdl);
    }
}


template <typename T>
void LocalMesh::parallelGather(
        const T* pvar,
        T* prxvar) {
#ifdef USE_MPI
    // This routine gathers slave values for which MYPE owns the masters.
    const int tagmpi = 100;
    const int type_size = sizeof(T);
//    std::vector<T> slvvar(numslv);
    T* slvvar = AbstractedMemory::alloc<T>(num_slaves);

    // Post receives for incoming messages from slaves.
    // Store results in proxy buffer.
//    vector<MPI_Request> request(numslvpe);
    MPI_Request* request = AbstractedMemory::alloc<MPI_Request>(num_slave_pes);
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
    MPI_Status* status = AbstractedMemory::alloc<MPI_Status>(num_slave_pes);
    int ierr = MPI_Waitall(num_slave_pes, &request[0], &status[0]);
    if (ierr != 0) {
        cerr << "Error: parallelGather MPI error " << ierr <<
                " on PE " << my_PE << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    AbstractedMemory::free(slvvar);
    AbstractedMemory::free(request);
    AbstractedMemory::free(status);
#endif
}


template <typename T>
void LocalMesh::parallelSum(
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
void LocalMesh::parallelScatter(
        T* pvar,
        const T* prxvar) {
#ifdef USE_MPI
    // This routine scatters master values on MYPE to all slave copies
    // owned by other PEs.
    const int tagmpi = 200;
    const int type_size = sizeof(T);
//    std::vector<T> slvvar(numslv);
    T* slvvar = AbstractedMemory::alloc<T>(num_slaves);

    // Post receives for incoming messages from masters.
    // Store results in slave buffer.
//    vector<MPI_Request> request(nummstrpe);
    MPI_Request* request = AbstractedMemory::alloc<MPI_Request>(num_mesg_send2master);
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
    MPI_Status* status = AbstractedMemory::alloc<MPI_Status>(num_mesg_send2master);
    int ierr = MPI_Waitall(num_mesg_send2master, &request[0], &status[0]);
    if (ierr != 0) {
        cerr << "Error: parallelScatter MPI error " << ierr <<
                " on PE " << my_PE << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // Store slave data from buffer back to points.
    for (int slv = 0; slv < num_slaves; ++slv) {
        int p = map_slave2pt[slv];
        pvar[p] = slvvar[slv];
    }

    AbstractedMemory::free(slvvar);
    AbstractedMemory::free(request);
    AbstractedMemory::free(status);
#endif
}

/*
template <typename T>
void Mesh::sumAcrossProcs(T* pvar) {
    if (num_subregions_ == 1) return;
//    std::vector<T> prxvar(numprx);
    T* prxvar = AbstractedMemory::alloc<T>(num_proxies);
    parallelGather(pvar, &prxvar[0]);
    parallelSum(pvar, &prxvar[0]);
    parallelScatter(pvar, &prxvar[0]);
    AbstractedMemory::free(prxvar);
}
*/

template <typename T>
void LocalMesh::sumOnProc(
        const T* cvar,
	    RegionAccessor<AccessorType::Generic, T>& pvar) {

    for (int pch = 0; pch < num_pt_chunks; ++pch) {
        int pfirst = pt_chunks_first[pch];
        int plast = pt_chunks_last[pch];
        for (int p = pfirst; p < plast; ++p) {
        		ptr_t pt_ptr(p);
            T x = T();
            for (int c = map_pt2crn_first[p]; c >= 0; c = map_crn2crn_next[c]) {
                x += cvar[c];
            }
            pvar.write(pt_ptr, x);
        }  // for p
    }  // for pch

}


void LocalMesh::sumToPoints(
        const double* corner_mass,
        const double2* corner_force) {


    DoubleAccessor pt_weighted_mass_ = local_points_by_gid.getRegionAccessor<double>(FID_PMASWT);
    Double2Accessor pt_force_ = local_points_by_gid.getRegionAccessor<double2>(FID_PF);
    sumOnProc(corner_mass, pt_weighted_mass_);
    sumOnProc(corner_force, pt_force_);


    // TODO this should be in Parallel::

    // slaves send to masters
 /*   CopyLauncher copy_launcher;
    copy_launcher.add_copy_requirements(
    		RegionRequirement(lregion_all_pts_, READ_ONLY, EXCLUSIVE, lregion_all_pts_),
    		RegionRequirement(ghost_pts_.get_logical_region(), WRITE_DISCARD, EXCLUSIVE,
    				ghost_pts_.get_logical_region()));
    copy_launcher.add_src_field(0, FID_PF);
    copy_launcher.add_dst_field(0, FID_GHOST_PF);
    runtime_->issue_copy_operation(ctx_, copy_launcher);*/
}

