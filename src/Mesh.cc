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

#include "AddReductionOp.hh"
#include "Add2ReductionOp.hh"
#include "Vec2.hh"
#include "Memory.hh"
#include "GenerateMesh.hh"

using namespace std;


LocalMesh::LocalMesh(const InputParameters& params,
		LogicalUnstructured& points,
        std::vector<LogicalUnstructured>& halos_pts,
        std::vector<PhysicalRegion>& pregionshalos,
        PhaseBarrier as_master,
        std::vector<PhaseBarrier> masters,
		Context ctx, HighLevelRuntime* rt) :
			chunk_size(params.directs.chunk_size),
			subregion_xmin(params.directs.subregion_xmin),
			subregion_xmax(params.directs.subregion_xmax),
			subregion_ymin(params.directs.subregion_ymin),
			subregion_ymax(params.directs.subregion_ymax),
            local_points_by_gid(ctx, rt, points.getISpace()),
            pt_x_init_by_gid(points),
			generate_mesh(NULL),
	        pbarrier_as_master(as_master),
	        masters_pbarriers(masters),
			ctx(ctx),
			runtime(rt),
			halos_points(halos_pts),
			pregions_halos(pregionshalos),
			num_subregions(params.directs.ntasks),
			my_color(params.directs.task_id)
	{

    generate_mesh = new GenerateMesh(params);

    local_points_by_gid.addField<double>(FID_PMASWT);
    local_points_by_gid.addField<double2>(FID_PF);
    local_points_by_gid.allocate();

	init();
}


LocalMesh::~LocalMesh() {
    delete generate_mesh;
}


void LocalMesh::init() {
    std::vector<double2> point_position_initial;
    std::vector<int> zone_points_pointer_calc;
    std::vector<int> zone_points_values_calc;
    generate_mesh->generate(
            point_position_initial, zone_points_pointer_calc, zone_points_values_calc);

    num_pts = point_position_initial.size();
    num_sides = zone_points_values_calc.size();
    num_corners = num_sides;
    num_zones = zone_points_pointer_calc.size() - 1;

    zone_pts_ptr = AbstractedMemory::alloc<int>(num_zones+1);
    copy(zone_points_pointer_calc.begin(), zone_points_pointer_calc.end(), zone_pts_ptr);

    // use the cell* arrays to populate the side maps
    initSideMappingArrays(zone_points_pointer_calc, zone_points_values_calc);

    // release memory from cell* arrays
    zone_points_pointer_calc.resize(0);
    zone_points_values_calc.resize(0);

    initEdgeMappingArrays();

    populateChunks();

    populateInverseMap();   // for corner-to-point gathers

    writeMeshStats();

    edge_x = AbstractedMemory::alloc<double2>(num_edges);
    zone_x = AbstractedMemory::alloc<double2>(num_zones);
    pt_x0 = AbstractedMemory::alloc<double2>(num_pts);
    pt_x = AbstractedMemory::alloc<double2>(num_pts);
    pt_x_pred = AbstractedMemory::alloc<double2>(num_pts);
    edge_x_pred = AbstractedMemory::alloc<double2>(num_edges);
    zone_x_pred = AbstractedMemory::alloc<double2>(num_zones);
    side_area = AbstractedMemory::alloc<double>(num_sides);
    side_vol = AbstractedMemory::alloc<double>(num_sides);
    zone_area = AbstractedMemory::alloc<double>(num_zones);
    zone_vol = AbstractedMemory::alloc<double>(num_zones);
    side_area_pred = AbstractedMemory::alloc<double>(num_sides);
    side_vol_pred = AbstractedMemory::alloc<double>(num_sides);
    zone_area_pred = AbstractedMemory::alloc<double>(num_zones);
    zone_vol_pred = AbstractedMemory::alloc<double>(num_zones);
    zone_vol0 = AbstractedMemory::alloc<double>(num_zones);
    side_surfp = AbstractedMemory::alloc<double2>(num_sides);
    edge_len = AbstractedMemory::alloc<double>(num_edges);
    zone_dl = AbstractedMemory::alloc<double>(num_zones);
    side_mass_frac = AbstractedMemory::alloc<double>(num_sides);

    IndexIterator itr = pt_x_init_by_gid.getIterator();
    point_local_to_globalID = AbstractedMemory::alloc<ptr_t>(num_pts);
    int i = 0;
    while (itr.has_next()) {
    		ptr_t pt_ptr = itr.next();
        point_local_to_globalID[i] = pt_ptr;
        assert(point_local_to_globalID[i].value == generate_mesh->pointLocalToGlobalID(i));  // TODO find fastest and use that, no assert
        	i++;
    }
    assert(i == num_pts);

    initParallel();

    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < num_pt_chunks; ++pch) {
        int pfirst = pt_chunks_first[pch];
        int plast = pt_chunks_last[pch];
        for (int p = pfirst; p < plast; ++p)
            pt_x[p] = point_position_initial[p];
    }
    point_position_initial.resize(0);

    num_bad_sides = 0;
    for (int sch = 0; sch < num_side_chunks; ++sch) {
        calcCtrs(sch, false);
        calcVols(sch, false);
        calcSideMassFracs(sch);
    }
    checkBadSides();

}


void LocalMesh::initSideMappingArrays(
        const vector<int>& cellstart,
        const vector<int>& cellnodes) {

    map_side2pt1 = AbstractedMemory::alloc<int>(num_sides);
    zone_pts_val = map_side2pt1;
    map_side2zone  = AbstractedMemory::alloc<int>(num_sides);

    for (int z = 0; z < num_zones; ++z) {
        int sbase = cellstart[z];
        int size = cellstart[z+1] - sbase;
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            map_side2zone[s] = z;
            map_side2pt1[s] = cellnodes[s];
        } // for n
    } // for z
}


void LocalMesh::initEdgeMappingArrays() {

    vector<vector<int> > edgepp(num_pts), edgepe(num_pts);

    map_side2edge = AbstractedMemory::alloc<int>(num_sides);

    int e = 0;
    for (int s = 0; s < num_sides; ++s) {
        int p1 = min(map_side2pt1[s], mapSideToPt2(s));
        int p2 = max(map_side2pt1[s], mapSideToPt2(s));

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


void LocalMesh::populateChunks() {

    if (chunk_size == 0) chunk_size = max(num_pts, num_sides);

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s1, s2 = 0;
    while (s2 < num_sides) {
        s1 = s2;
        s2 = min(s2 + chunk_size, num_sides);
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
        p2 = min(p2 + chunk_size, num_pts);
        pt_chunks_first.push_back(p1);
        pt_chunks_last.push_back(p2);
    }
    num_pt_chunks = pt_chunks_first.size();

    // compute zone chunks
    int z1, z2 = 0;
    while (z2 < num_zones) {
        z1 = z2;
        z2 = min(z2 + chunk_size, num_zones);
        zone_chunk_first.push_back(z1);
        zone_chunk_last.push_back(z2);
    }
    num_zone_chunks = zone_chunk_first.size();

}


void LocalMesh::populateInverseMap() {
    map_pt2crn_first = AbstractedMemory::alloc<int>(num_pts);
    map_crn2crn_next = AbstractedMemory::alloc<int>(num_sides);

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


void LocalMesh::writeMeshStats() {

    int64_t gnump = num_pts;
    // make sure that boundary points aren't double-counted;
    // only count them if they are masters
    if (num_subregions > 1) gnump -= num_slaves;
    int64_t gnumz = num_zones;
    int64_t gnums = num_sides;
    int64_t gnume = num_edges;
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

    // TODO if (my_color > 0) return;

    cout << "--- Mesh Information ---" << endl; // TODO must be global sum
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

    for (int p = 0; p < num_pts; ++p) {
    		if (fabs(pt_x[p].x - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


vector<int> LocalMesh::getYPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts; ++p) {
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
        zx = zone_x;
    }
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1[s];
        int p2 = mapSideToPt2(s);
        int e = map_side2edge[s];
        int z = map_side2zone[s];
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
        zx = zone_x;
        sarea = side_area;
        svol = side_vol;
        zarea = zone_area;
        zvol = zone_vol;
    }
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    const double third = 1. / 3.;
    int count = 0;
    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1[s];
        int p2 = mapSideToPt2(s);
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
        int z = map_side2zone[s];
        side_mass_frac[s] = side_area[s] / zone_area[z];
    }
}


void LocalMesh::calcMedianMeshSurfVecs(const int side_chunk) {
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        int e = map_side2edge[s];

        side_surfp[s] = rotateCCW(edge_x_pred[e] - zone_x_pred[z]);

    }

}


void LocalMesh::calcEdgeLen(const int side_chunk) {
	int sfirst = side_chunks_first[side_chunk];
	int slast = side_chunks_last[side_chunk];

	for (int s = sfirst; s < slast; ++s) {
        const int p1 = map_side2pt1[s];
        const int p2 = mapSideToPt2(s);
        const int e = map_side2edge[s];

        edge_len[e] = length(pt_x_pred[p2] - pt_x_pred[p1]);

    }
}


void LocalMesh::calcCharacteristicLen(const int side_chunk) {
    int sfirst = side_chunks_first[side_chunk];
    int slast = side_chunks_last[side_chunk];

    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zone_dl[zfirst], &zone_dl[zlast], 1.e99);

    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        int e = map_side2edge[s];

        double area = side_area_pred[s];
        double base = edge_len[e];
        double fac = (zoneNPts(z) == 3 ? 3. : 4.);
        double sdl = fac * area / base;
        zone_dl[z] = min(zone_dl[z], sdl);
    }
}


template <typename T>
void LocalMesh::sumOnProc(
        const T* cvar,
	    RegionAccessor<AccessorType::Generic, T>& pvar) {

    for (int pch = 0; pch < num_pt_chunks; ++pch) {
        int pfirst = pt_chunks_first[pch];
        int plast = pt_chunks_last[pch];
        for (int p = pfirst; p < plast; ++p) {
        		ptr_t pt_ptr = point_local_to_globalID[p];
            T x = T();
            for (int c = map_pt2crn_first[p]; c >= 0; c = map_crn2crn_next[c]) {
                x += cvar[c];
            }
            pvar.write(pt_ptr, x);
        }  // for p
    }  // for pch

}


void LocalMesh::initParallel() {
    if (num_subregions == 1) return;

    vector<int> master_points_counts, master_points;
    std::vector<int> slaved_points_counts, slaved_points;
    generate_mesh->generateHaloPoints(
            master_colors, slaved_points_counts, slaved_points,
            slave_colors, master_points_counts, master_points);

    num_slaves = slaved_points.size();

    unsigned previous_master_pts_count = 0;
    for (unsigned  master = 0; master < slaved_points_counts.size(); master++) {
        Coloring my_slaved_pts_map;
        for (unsigned pt = 0; pt < slaved_points_counts[master]; pt++)
            my_slaved_pts_map[1+master].points.insert(
                    point_local_to_globalID[slaved_points[pt + previous_master_pts_count]].value);

        previous_master_pts_count += slaved_points_counts[master];
        halos_points[1+master].partition(my_slaved_pts_map, true);
        slaved_halo_points.push_back(LogicalUnstructured(ctx, runtime, halos_points[1+master].getLRegion(1+master)));
    }

    slaved_points_counts.resize(0);
    master_points_counts.resize(0);
    slaved_points.resize(0);
    master_points.resize(0);
}


void LocalMesh::sumToPoints(
        const double* corner_mass,
        const double2* corner_force)
{
    DoubleAccessor pt_weighted_mass_ = local_points_by_gid.getRegionAccessor<double>(FID_PMASWT);
    Double2Accessor pt_force_ = local_points_by_gid.getRegionAccessor<double2>(FID_PF);
    sumOnProc(corner_mass, pt_weighted_mass_);
    sumOnProc(corner_force, pt_force_);

    if (slave_colors.size() > 0) {
        // phase 1 as master: master copies partial result in; slaves may not access data
      {
        CopyLauncher copy_launcher;
        copy_launcher.add_copy_requirements(
                RegionRequirement(local_points_by_gid.getLRegion(), READ_ONLY, EXCLUSIVE, local_points_by_gid.getLRegion()),
                RegionRequirement(halos_points[0].getLRegion(), READ_WRITE, SIMULTANEOUS, halos_points[0].getLRegion()));
        copy_launcher.add_src_field(0, FID_PMASWT);
        copy_launcher.add_dst_field(0, FID_GHOST_PMASWT);
        copy_launcher.add_wait_barrier(pbarrier_as_master);                     // 3 * cycle
        copy_launcher.add_arrival_barrier(pbarrier_as_master);                  // 3 * cycle + 1
        runtime->issue_copy_operation(ctx, copy_launcher);
        pbarrier_as_master.arrive(slave_colors.size());                         // 3 * cycle + 1 (slaves never arrive here)
      }
      {
        CopyLauncher copy_launcher;
        copy_launcher.add_copy_requirements(
                RegionRequirement(local_points_by_gid.getLRegion(), READ_ONLY, EXCLUSIVE, local_points_by_gid.getLRegion()),
                RegionRequirement(halos_points[0].getLRegion(), READ_WRITE, SIMULTANEOUS, halos_points[0].getLRegion()));
        copy_launcher.add_src_field(0, FID_PF);
        copy_launcher.add_dst_field(0, FID_GHOST_PF);
        copy_launcher.add_wait_barrier(pbarrier_as_master);                     // 3 * cycle
        copy_launcher.add_arrival_barrier(pbarrier_as_master);                  // 3 * cycle + 1
        runtime->issue_copy_operation(ctx, copy_launcher);
        pbarrier_as_master.arrive(slave_colors.size());                         // 3 * cycle + 1 (slaves never arrive here)
      }
        pbarrier_as_master =
                runtime->advance_phase_barrier(ctx, pbarrier_as_master);        // 3 * cycle + 1
        // phase 2 as master: slaves reduce; no one can read data
        pbarrier_as_master.arrive(2);                                           // 3 * cycle + 2
        pbarrier_as_master =
                runtime->advance_phase_barrier(ctx, pbarrier_as_master);        // 3 * cycle + 2
    }

    for (int master=0; master < master_colors.size(); master++) {
        // phase 2 as slave: slaves reduce; no one can read data
        masters_pbarriers[master] =
                runtime->advance_phase_barrier(ctx, masters_pbarriers[master]); // 3 * cycle + 1
      {
        CopyLauncher copy_launcher;
        copy_launcher.add_copy_requirements(
                RegionRequirement(local_points_by_gid.getLRegion(), READ_ONLY, EXCLUSIVE,
                        local_points_by_gid.getLRegion()),
                RegionRequirement(slaved_halo_points[master].getLRegion(), AddReductionOp::redop_id,
                        SIMULTANEOUS, halos_points[1+master].getLRegion()));
        copy_launcher.add_src_field(0, FID_PMASWT);
        copy_launcher.add_dst_field(0, FID_GHOST_PMASWT);
        copy_launcher.add_wait_barrier(masters_pbarriers[master]);              // 3 * cycle + 1
        copy_launcher.add_arrival_barrier(masters_pbarriers[master]);           // 3 * cycle + 2
        runtime->issue_copy_operation(ctx, copy_launcher);
      }
      {
        CopyLauncher copy_launcher;
        copy_launcher.add_copy_requirements(
                RegionRequirement(local_points_by_gid.getLRegion(), READ_ONLY, EXCLUSIVE,
                        local_points_by_gid.getLRegion()),
                RegionRequirement(slaved_halo_points[master].getLRegion(), Add2ReductionOp::redop_id,
                        SIMULTANEOUS, halos_points[1+master].getLRegion()));
        copy_launcher.add_src_field(0, FID_PF);
        copy_launcher.add_dst_field(0, FID_GHOST_PF);
        copy_launcher.add_wait_barrier(masters_pbarriers[master]);              // 3 * cycle + 1
        copy_launcher.add_arrival_barrier(masters_pbarriers[master]);           // 3 * cycle + 2
        runtime->issue_copy_operation(ctx, copy_launcher);
      }
        masters_pbarriers[master] =
                runtime->advance_phase_barrier(ctx, masters_pbarriers[master]); // 3 * cycle + 2
    }

    if (slave_colors.size() > 0) {
        // phase 3 as master: everybody can read accumulation
        pbarrier_as_master.wait();                                              // 3 * cycle + 2

        RegionRequirement halo_req(halos_points[0].getLRegion(), READ_ONLY, EXCLUSIVE,
                halos_points[0].getLRegion());
        halo_req.add_field(FID_GHOST_PMASWT);
        halo_req.add_field(FID_GHOST_PF);
        // TODO use LogicUnstruct object
        InlineLauncher halo_launcher(halo_req);
        PhysicalRegion pregion_halo = runtime->map_region(ctx, halo_launcher);
        DoubleAccessor acc_halo =
                pregion_halo.get_field_accessor(FID_GHOST_PMASWT).typeify<double>();
        Double2Accessor acc2_halo =
                pregion_halo.get_field_accessor(FID_GHOST_PF).typeify<double2>();
        // TODO just copy launch it back
        {
            IndexIterator itr = halos_points[0].getIterator();
            while (itr.has_next()) {
                ptr_t pt_ptr = itr.next();
                pt_weighted_mass_.write(pt_ptr, acc_halo.read(pt_ptr));
                pt_force_.write(pt_ptr, acc2_halo.read(pt_ptr));
            }
        }
        runtime->unmap_region(ctx, pregion_halo);

        pbarrier_as_master.arrive(2);                                           // 3 * cycle + 3
        pbarrier_as_master =
                runtime->advance_phase_barrier(ctx, pbarrier_as_master);        // 3 * cycle + 3
    }

    for (int master=0; master < master_colors.size(); master++) {
        // phase 3 as slave: everybody can read accumulation
        AcquireLauncher acquire_launcher(slaved_halo_points[master].getLRegion(),
                halos_points[1+master].getLRegion(), pregions_halos[1+master]); // TODO tuck this funny pregion into LogicUnstruct
        acquire_launcher.add_field(FID_GHOST_PMASWT);
        acquire_launcher.add_field(FID_GHOST_PF);
        acquire_launcher.add_wait_barrier(masters_pbarriers[master]);           // 3 * cycle + 2
        runtime->issue_acquire(ctx, acquire_launcher);

        // TODO use LogicUnstruct
        RegionRequirement halo_req(slaved_halo_points[master].getLRegion(), READ_ONLY, EXCLUSIVE,
                halos_points[1+master].getLRegion());
        halo_req.add_field(FID_GHOST_PMASWT);
        halo_req.add_field(FID_GHOST_PF);
        InlineLauncher halo_launcher(halo_req);
        PhysicalRegion pregion_halo = runtime->map_region(ctx, halo_launcher);
        DoubleAccessor acc_halo =
                pregion_halo.get_field_accessor(FID_GHOST_PMASWT).typeify<double>();
        Double2Accessor acc2_halo =
                pregion_halo.get_field_accessor(FID_GHOST_PF).typeify<double2>();

        {
            IndexIterator itr = slaved_halo_points[master].getIterator();
            while (itr.has_next()) {
                ptr_t pt_ptr = itr.next();
                pt_weighted_mass_.write(pt_ptr, acc_halo.read(pt_ptr));
                pt_force_.write(pt_ptr, acc2_halo.read(pt_ptr));
            }
        }

        ReleaseLauncher release_launcher(slaved_halo_points[master].getLRegion(),
                halos_points[1+master].getLRegion(), pregions_halos[1+master]); // TODO tuck this funny pregion into LogicUnstruct
        release_launcher.add_field(FID_GHOST_PMASWT);
        release_launcher.add_field(FID_GHOST_PF);
        release_launcher.add_arrival_barrier(masters_pbarriers[master]);        // 3 * cycle + 3
        release_launcher.add_arrival_barrier(masters_pbarriers[master]);        // 3 * cycle + 3
        runtime->issue_release(ctx, release_launcher);
        masters_pbarriers[master] =
                runtime->advance_phase_barrier(ctx, masters_pbarriers[master]); // 3 * cycle + 3
    }
}

