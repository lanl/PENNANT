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

#include "LocalMesh.hh"

#include <stdint.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "AddReductionOp.hh"
#include "Add2ReductionOp.hh"
#include "GenerateMesh.hh"
#include "HaloTask.hh"
#include "Memory.hh"
#include "Vec2.hh"

using namespace std;


LocalMesh::LocalMesh(const InputParameters& params,
		IndexSpace points,
        std::vector<LogicalUnstructured>& halos_pts,
        std::vector<PhysicalRegion>& pregionshalos,
        PhaseBarrier as_master,
        std::vector<PhaseBarrier> masters,
        DynamicCollective add_reduction,
		Context ctx, HighLevelRuntime* rt) :
			subregion_xmin(params.directs.subregion_xmin),
			subregion_xmax(params.directs.subregion_xmax),
			subregion_ymin(params.directs.subregion_ymin),
			subregion_ymax(params.directs.subregion_ymax),
            local_points_by_gid(ctx, rt, points),
            zone_pts(ctx, rt),
            zones(ctx, rt),
            sides(ctx, rt),
            points(ctx, rt),
            edges(ctx, rt),
            chunk_size(params.directs.chunk_size),
            pt_x_init_by_gid(ctx, rt, points),
			generate_mesh(NULL),
	        pbarrier_as_master(as_master),
	        masters_pbarriers(masters),
	        add_reduction(add_reduction),
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

    allocateFields();

    zone_pts.allocate(num_zones+1);
    int* zone_pts_ptr = zone_pts.getRawPtr<int>(FID_ZONE_PTS_PTR);
    copy(zone_points_pointer_calc.begin(), zone_points_pointer_calc.end(), zone_pts_ptr);

    zones.allocate(num_zones);
    double2* zone_x = zones.getRawPtr<double2>(FID_ZX);
    double* zone_area = zones.getRawPtr<double>(FID_ZAREA);
    double* zone_vol = zones.getRawPtr<double>(FID_ZVOL);

    sides.allocate(num_sides);
    double* side_area = sides.getRawPtr<double>(FID_SAREA);
    double* side_vol = sides.getRawPtr<double>(FID_SVOL);
    double* side_mass_frac = sides.getRawPtr<double>(FID_SMF);
    int* map_side2pt1 = sides.getRawPtr<int>(FID_SMAP_SIDE_TO_PT1);
    int* map_side2zone = sides.getRawPtr<int>(FID_SMAP_SIDE_TO_ZONE);
    int* map_side2edge = sides.getRawPtr<int>(FID_SMAP_SIDE_TO_EDGE);
    int* map_crn2crn_next = sides.getRawPtr<int>(FID_MAP_CRN2CRN_NEXT);

    // use the cell* arrays to populate the side maps
    initSideMappingArrays(zone_points_pointer_calc, zone_points_values_calc,
            map_side2zone, map_side2pt1);

    // release memory from cell* arrays
    zone_points_pointer_calc.resize(0);
    zone_points_values_calc.resize(0);

    initEdgeMappingArrays(map_side2zone, zone_pts_ptr, map_side2pt1, map_side2edge);

    populateChunks(map_side2zone);

    points.allocate(num_pts);
    double2* pt_x = points.getRawPtr<double2>(FID_PX);
    int* map_pt2crn_first = points.getRawPtr<int>(FID_MAP_PT2CRN_FIRST);

    populateInverseMap(map_side2pt1, map_pt2crn_first,
            map_crn2crn_next);   // for corner-to-point gathers

    edges.allocate(num_edges);
    double2* edge_x = edges.getRawPtr<double2>(FID_EX);

    initParallel();

    writeMeshStats();

    #pragma omp parallel for schedule(static)
    for (int pch = 0; pch < num_pt_chunks(); ++pch) {
        int pfirst = pt_chunks_CRS[pch];
        int plast = pt_chunks_CRS[pch+1];
        for (int p = pfirst; p < plast; ++p)
            pt_x[p] = point_position_initial[p];
    }
    point_position_initial.resize(0);

    for (int sch = 0; sch < num_side_chunks(); ++sch) {
        int sfirst =  side_chunks_CRS[sch];
        int slast =  side_chunks_CRS[sch+1];
        calcCtrs(sfirst, slast,  pt_x,
                 map_side2zone,  num_sides,  num_zones,  map_side2pt1,  map_side2edge, zone_pts_ptr,
                 edge_x,  zone_x);
        calcVols(sfirst, slast,  pt_x,  zone_x,
                 map_side2zone,  num_sides,  num_zones,  map_side2pt1, zone_pts_ptr,
                 side_area,  side_vol,  zone_area,  zone_vol);
        calcSideMassFracs(sch, map_side2zone, side_area, zone_area, side_mass_frac);
    }

    edges.unMapPRegion();
    points.unMapPRegion();
    sides.unMapPRegion();
    zones.unMapPRegion();
    zone_pts.unMapPRegion();
}


void LocalMesh::allocateFields()
{
    zone_pts.addField<int>(FID_ZONE_PTS_PTR);
    zones.addField<double2>(FID_ZX);
    zones.addField<double>(FID_ZAREA);
    zones.addField<double>(FID_ZVOL);
    zones.addField<double>(FID_ZVOL0);
    zones.addField<double>(FID_ZDL);
    edges.addField<double2>(FID_EX);
    points.addField<double2>(FID_PX0);
    points.addField<double2>(FID_PX);
    points.addField<double2>(FID_PXP);
    points.addField<int>(FID_MAP_PT2CRN_FIRST);
    sides.addField<int>(FID_MAP_CRN2CRN_NEXT);
    sides.addField<double>(FID_SAREA);
    sides.addField<double>(FID_SVOL);
    sides.addField<double>(FID_SMF);
    sides.addField<int>(FID_SMAP_SIDE_TO_PT1);
    sides.addField<int>(FID_SMAP_SIDE_TO_ZONE);
    sides.addField<int>(FID_SMAP_SIDE_TO_EDGE);
}


void LocalMesh::initSideMappingArrays(
        const vector<int>& cellstart,
        const vector<int>& cellnodes,
        int* map_side2zone,
        int* map_side2pt1)
{
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


void LocalMesh::initEdgeMappingArrays(
        const int* map_side2zone,
        const int* zone_pts_ptr,
        int* map_side2pt1,
        int* map_side2edge)
{
    vector<vector<int> > edgepp(num_pts), edgepe(num_pts);

    int e = 0;
    for (int s = 0; s < num_sides; ++s) {
        int p1 = min(map_side2pt1[s], mapSideToPt2(s, map_side2pt1, map_side2zone, zone_pts_ptr));
        int p2 = max(map_side2pt1[s], mapSideToPt2(s, map_side2pt1, map_side2zone, zone_pts_ptr));

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


void LocalMesh::populateChunks(const int* map_side2zone)
{
    if (chunk_size == 0) chunk_size = max(num_pts, num_sides);

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s2 = 0;
    side_chunks_CRS.push_back(s2);
    while (s2 < num_sides) {
        s2 = min(s2 + chunk_size, num_sides);
        while (s2 < num_sides && map_side2zone[s2] == map_side2zone[s2-1])
            --s2;
        side_chunks_CRS.push_back(s2);
    }

    // compute point chunks
    int p2 = 0;
    pt_chunks_CRS.push_back(p2);
    while (p2 < num_pts) {
        p2 = min(p2 + chunk_size, num_pts);
        pt_chunks_CRS.push_back(p2);
    }

    // compute zone chunks
    int z2 = 0;
    zone_chunks_CRS.push_back(z2);
    while (z2 < num_zones) {
        z2 = min(z2 + chunk_size, num_zones);
        zone_chunks_CRS.push_back(z2);
    }
}


void LocalMesh::populateInverseMap(const int* map_side2pt1,
        int* map_pt2crn_first,
        int* map_crn2crn_next)
{
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
    int gnumpch = num_pt_chunks();
    int gnumzch = num_zone_chunks();
    int gnumsch = num_side_chunks();

    Future future_sum = Parallel::globalSumInt64(gnump, add_reduction, runtime, ctx);
    gnump = future_sum.get_result<int64_t>();

    future_sum = Parallel::globalSumInt64(gnumz, add_reduction, runtime, ctx);
    gnumz = future_sum.get_result<int64_t>();

    future_sum = Parallel::globalSumInt64(gnums, add_reduction, runtime, ctx);
    gnums = future_sum.get_result<int64_t>();

    future_sum = Parallel::globalSumInt64(gnume, add_reduction, runtime, ctx);
    gnume = future_sum.get_result<int64_t>();

    future_sum = Parallel::globalSumInt64(gnumpch, add_reduction, runtime, ctx);
    gnumpch = future_sum.get_result<int64_t>();

    future_sum = Parallel::globalSumInt64(gnumzch, add_reduction, runtime, ctx);
    gnumzch = future_sum.get_result<int64_t>();

    future_sum = Parallel::globalSumInt64(gnumsch, add_reduction, runtime, ctx);
    gnumsch = future_sum.get_result<int64_t>();

    if (my_color > 0) return;

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


/*static*/
vector<int> LocalMesh::getXPlane(const double c,
        const int num_pts,
        const double2 *pt_x)
{
    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts; ++p) {
    		if (fabs(pt_x[p].x - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


/*static*/
vector<int> LocalMesh::getYPlane(const double c,
        const int num_pts,
        const double2 *pt_x)
{
    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < num_pts; ++p) {
        if (fabs(pt_x[p].y - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


/*static*/
void LocalMesh::getPlaneChunks(
        const std::vector<int>& mapbp,
        const std::vector<int> pt_chunks_CRS,
        vector<int>& pchb_CRS)
{
    pchb_CRS.resize(0);

    // compute boundary point chunks
    // (boundary points contained in each point chunk)
    int bl = 0;
    pchb_CRS.push_back(bl);
    for (int pch = 0; pch < (pt_chunks_CRS.size()-1); ++pch) {
         int pl = pt_chunks_CRS[pch+1];
         bl = lower_bound(&mapbp[pchb_CRS.back()], &mapbp[mapbp.size()], pl) - &mapbp[0];
         pchb_CRS.push_back(bl);
    }
}


/*static*/
void LocalMesh::calcCtrs(
        const int sfirst,
        const int slast,
        const double2* px,
        const int* map_side2zone,
        const int num_sides,
        const int num_zones,
        const int* map_side2pt1,
        const int* map_side2edge,
        const int* zone_pts_ptr,
        double2* ex,
        double2* zx)
{
    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1[s];
        int p2 = mapSideToPt2(s, map_side2pt1, map_side2zone, zone_pts_ptr);
        int e = map_side2edge[s];
        int z = map_side2zone[s];
        ex[e] = 0.5 * (px[p1] + px[p2]);
        zx[z] += px[p1];
    }

    for (int z = zfirst; z < zlast; ++z) {
        zx[z] /= (double) zoneNPts(z, zone_pts_ptr);
    }

}


/*static*/
void LocalMesh::calcVols(
        const int sfirst,
        const int slast,
        const double2* px,
        const double2* zx,
        const int* map_side2zone,
        const int num_sides,
        const int num_zones,
        const int* map_side2pt1,
        const int* zone_pts_ptr,
        double* sarea,
        double* svol,
        double* zarea,
        double* zvol)
{
    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    const double third = 1. / 3.;
    int count = 0;
    for (int s = sfirst; s < slast; ++s) {
        int p1 = map_side2pt1[s];
        int p2 = mapSideToPt2(s, map_side2pt1, map_side2zone, zone_pts_ptr);
        int z = map_side2zone[s];

        // compute side volumes, sum to zone
        double sa = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
        double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
        sarea[s] = sa;
        if (svol != nullptr)
            svol[s] = sv;
        zarea[z] += sa;
        zvol[z] += sv;

        // check for negative side volumes
        if (sv <= 0.) count += 1;

    } // for s

    if (count > 0) {
        cerr << "Error: negative side volume" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


void LocalMesh::calcSideMassFracs(const int side_chunk,
        const int* map_side2zone,
        const double* side_area,
        const double* zone_area,
        double* side_mass_frac)
{
	int sfirst = side_chunks_CRS[side_chunk];
	int slast = side_chunks_CRS[side_chunk+1];

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        side_mass_frac[s] = side_area[s] / zone_area[z];
    }
}


/*static*/
void LocalMesh::calcMedianMeshSurfVecs(
        const int sfirst,
        const int slast,
        const int* map_side2zone,
        const int* map_side2edge,
        const double2* edge_x_pred,
        const double2* zone_x_pred,
        double2* side_surfp)
{
    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        int e = map_side2edge[s];
        side_surfp[s] = rotateCCW(edge_x_pred[e] - zone_x_pred[z]);
    }
}


/*static*/
void LocalMesh::calcEdgeLen(
        const int sfirst,
        const int slast,
        const int* map_side2pt1,
        const int* map_side2edge,
        const int* map_side2zone,
        const int* zone_pts_ptr,
        const double2* pt_x_pred,
        double* edge_len)
{
	for (int s = sfirst; s < slast; ++s) {
        const int p1 = map_side2pt1[s];
        const int p2 = mapSideToPt2(s, map_side2pt1, map_side2zone, zone_pts_ptr);
        const int e = map_side2edge[s];

        edge_len[e] = length(pt_x_pred[p2] - pt_x_pred[p1]);

    }
}


/*static*/
void LocalMesh::calcCharacteristicLen(
        const int sfirst,
        const int slast,
        const int* map_side2zone,
        const int* map_side2edge,
        const int* zone_pts_ptr,
        const double* side_area_pred,
        const double* edge_len,
        const int num_sides,
        const int num_zones,
        double* zone_dl)
{
    int zfirst = map_side2zone[sfirst];
    int zlast = (slast < num_sides ? map_side2zone[slast] : num_zones);
    fill(&zone_dl[zfirst], &zone_dl[zlast], 1.e99);

    for (int s = sfirst; s < slast; ++s) {
        int z = map_side2zone[s];
        int e = map_side2edge[s];

        double area = side_area_pred[s];
        double base = edge_len[e];
        double fac = (zoneNPts(z, zone_pts_ptr) == 3 ? 3. : 4.);
        double sdl = fac * area / base;
        zone_dl[z] = min(zone_dl[z], sdl);
    }
}


void LocalMesh::sumOnProc(
        const double* corner_mass,
        const double2* corner_force,
	    const std::vector<int> pt_chunks_CRS,
	    const int* map_pt2crn_first,
	    const int* map_crn2crn_next,
	    const GenerateMesh* generate_mesh,
	    DoubleAccessor pt_weighted_mass,
	    Double2Accessor pt_force)
{
    for (int point_chunk = 0; point_chunk < (pt_chunks_CRS.size()-1); ++point_chunk) {
        int pfirst = pt_chunks_CRS[point_chunk];
        int plast = pt_chunks_CRS[point_chunk+1];
        for (int point = pfirst; point < plast; ++point) {
        		ptr_t pt_ptr(generate_mesh->pointLocalToGlobalID(point));
            double mass = double();
            double2 force = double2();
            for (int corner = map_pt2crn_first[point]; corner >= 0; corner = map_crn2crn_next[corner]) {
                mass += corner_mass[corner];
                force += corner_force[corner];
            }
            pt_weighted_mass.write(pt_ptr, mass);
            pt_force.write(pt_ptr, force);
        }
    }

}


void LocalMesh::initParallel() {
    if (num_subregions == 1) return;

    vector<int> master_points_counts, master_points;
    std::vector<int> slaved_points_counts, slaved_points;
    generate_mesh->generateHaloPoints(
            master_colors, slaved_points_counts, slaved_points,
            slave_colors, master_points_counts, master_points);

    num_slaves = slaved_points.size();

    Coloring all_slaved_pts_map;
    all_slaved_pts_map[0].points = std::set<ptr_t>(); // empty set

    unsigned previous_slaved_pts_count = 0;
    for (unsigned  slave = 0; slave < master_points_counts.size(); slave++) {
        for (unsigned pt = 0; pt < master_points_counts[slave]; pt++) {
            all_slaved_pts_map[0].points.insert(
                    generate_mesh->pointLocalToGlobalID(master_points[pt + previous_slaved_pts_count]));
        }
        previous_slaved_pts_count += master_points_counts[slave];
    }

    unsigned previous_master_pts_count = 0;
    for (unsigned  master = 0; master < slaved_points_counts.size(); master++) {
        Coloring my_slaved_pts_map;
        for (unsigned pt = 0; pt < slaved_points_counts[master]; pt++) {
            my_slaved_pts_map[1+master].points.insert(
                    generate_mesh->pointLocalToGlobalID(slaved_points[pt + previous_master_pts_count]));
            all_slaved_pts_map[1+master].points.insert(
                    generate_mesh->pointLocalToGlobalID(slaved_points[pt + previous_master_pts_count]));
        }
        previous_master_pts_count += slaved_points_counts[master];
        halos_points[1+master].partition(my_slaved_pts_map, true);
        slaved_halo_points.push_back(LogicalUnstructured(ctx, runtime, halos_points[1+master].getLRegion(1+master)));
    }

    local_points_by_gid.partition(all_slaved_pts_map, true);
    local_halos_points.push_back(LogicalUnstructured(ctx, runtime, local_points_by_gid.getLRegion(0)));
    for (unsigned  master = 0; master < slaved_points_counts.size(); master++)
        local_halos_points.push_back(LogicalUnstructured(ctx, runtime, local_points_by_gid.getLRegion(1+master)));

    slaved_points_counts.resize(0);
    master_points_counts.resize(0);
    slaved_points.resize(0);
    master_points.resize(0);
}


void LocalMesh::sumCornersToPoints(LogicalStructured& sides_and_corners,
        DoCycleTasksArgsSerializer& serial)
{
    HaloTask halo_launcher(sides.getLRegion(),
            points.getLRegion(),
            local_points_by_gid.getLRegion(),
            sides_and_corners.getLRegion(),
            serial.getBitStream(), serial.getBitStreamSize());
    runtime->execute_task(ctx, halo_launcher);

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
        {
          CopyLauncher copy_launcher;
          copy_launcher.add_copy_requirements(
                  RegionRequirement(halos_points[0].getLRegion(), READ_ONLY,
                          EXCLUSIVE, halos_points[0].getLRegion()),
                  RegionRequirement(local_halos_points[0].getLRegion(), READ_WRITE, EXCLUSIVE,
                          local_points_by_gid.getLRegion()));
          copy_launcher.add_dst_field(0, FID_PMASWT);
          copy_launcher.add_src_field(0, FID_GHOST_PMASWT);
          copy_launcher.add_wait_barrier(pbarrier_as_master);
          runtime->issue_copy_operation(ctx, copy_launcher);
        }
        {
          CopyLauncher copy_launcher;
          copy_launcher.add_copy_requirements(
                  RegionRequirement(halos_points[0].getLRegion(), READ_ONLY,
                          EXCLUSIVE, halos_points[0].getLRegion()),
                  RegionRequirement(local_halos_points[0].getLRegion(), READ_WRITE, EXCLUSIVE,
                          local_points_by_gid.getLRegion()));
          copy_launcher.add_dst_field(0, FID_PF);
          copy_launcher.add_src_field(0, FID_GHOST_PF);
          copy_launcher.add_wait_barrier(pbarrier_as_master);
          runtime->issue_copy_operation(ctx, copy_launcher);
        }

        pbarrier_as_master.arrive(2);                                           // 3 * cycle + 3
        pbarrier_as_master =
                runtime->advance_phase_barrier(ctx, pbarrier_as_master);        // 3 * cycle + 3
    }

    for (int master=0; master < master_colors.size(); master++) {
        // phase 3 as slave: everybody can read accumulation
        {
          CopyLauncher copy_launcher;
          copy_launcher.add_copy_requirements(
                  RegionRequirement(slaved_halo_points[master].getLRegion(), READ_ONLY,
                          EXCLUSIVE, halos_points[1+master].getLRegion()),
                  RegionRequirement(local_halos_points[1+master].getLRegion(), READ_WRITE, EXCLUSIVE,
                          local_points_by_gid.getLRegion()));
          copy_launcher.add_dst_field(0, FID_PMASWT);
          copy_launcher.add_src_field(0, FID_GHOST_PMASWT);
          copy_launcher.add_wait_barrier(masters_pbarriers[master]);           // 3 * cycle + 2
          runtime->issue_copy_operation(ctx, copy_launcher);
        }
        {
          CopyLauncher copy_launcher;
          copy_launcher.add_copy_requirements(
                  RegionRequirement(slaved_halo_points[master].getLRegion(), READ_ONLY,
                          EXCLUSIVE, halos_points[1+master].getLRegion()),
                  RegionRequirement(local_halos_points[1+master].getLRegion(), READ_WRITE, EXCLUSIVE,
                          local_points_by_gid.getLRegion()));
          copy_launcher.add_dst_field(0, FID_PF);
          copy_launcher.add_src_field(0, FID_GHOST_PF);
          copy_launcher.add_wait_barrier(masters_pbarriers[master]);           // 3 * cycle + 2
          runtime->issue_copy_operation(ctx, copy_launcher);
        }
        masters_pbarriers[master].arrive(2);                                    // 3 * cycle + 3
        masters_pbarriers[master] =
                runtime->advance_phase_barrier(ctx, masters_pbarriers[master]); // 3 * cycle + 3
    }
}

