/*
 * LocalMesh.hh
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef LOCALMESH_HH_
#define LOCALMESH_HH_

#include <string>
#include <vector>

#include "GenerateMesh.hh"
#include "InputParameters.hh"
#include "LogicalStructured.hh"
#include "LogicalUnstructured.hh"
#include "Parallel.hh"
#include "Vec2.hh"

class LocalMesh {
public:

    LocalMesh(const InputParameters& params,
   		LogicalUnstructured& pts,
        std::vector<LogicalUnstructured>& halos_points,
        std::vector<PhysicalRegion>& pregions_halos,
        PhaseBarrier pbarrier_as_master,
        std::vector<PhaseBarrier> masters_pbarriers,
        Context ctx, HighLevelRuntime* rt);
    ~LocalMesh();

    // parameters
    int chunk_size;                 // max size for processing chunks
    double subregion_xmin; 		   // bounding box for a subregion
    double subregion_xmax; 		   // if xmin != std::numeric_limits<double>::max(),
    double subregion_ymin;         // should have 4 entries:
    double subregion_ymax; 		   // xmin, xmax, ymin, ymax

    // mesh variables
    // (See documentation for more details on the mesh
    //  data structures...)
    int num_pts, num_edges, num_zones, num_sides, num_corners;

    static inline int mapSideToSideNext(const int &s,
            const int* map_side2zone,
            const int* zone_pts_ptr)
    {
        const int z = map_side2zone[s];
        const int sbase = zone_pts_ptr[z];
        const int slast = zone_pts_ptr[z+1];
        const int snext = (s + 1 == slast ? sbase : s + 1);
        return snext;
    }
    static inline int mapSideToPt2(const int &s,
            const int* map_side2pt1,
            const int* map_side2zone,
            const int* zone_pts_ptr)
    {
        return map_side2pt1[LocalMesh::mapSideToSideNext(s, map_side2zone, zone_pts_ptr)];
    }

    inline int mapSideToSideNext(const int &s) const
    {
    	return mapSideToSideNext(s, map_side2zone, zone_pts_ptr);
    }
    inline int mapSideToSidePrev(const int &s) const
    {
        return mapSideToSidePrev(s, map_side2zone, zone_pts_ptr);
    }
    static inline int mapSideToSidePrev(const int &s,
            const int* map_side2zone,
            const int* zone_pts_ptr)
    {
        const int z = map_side2zone[s];
        const int sbase = zone_pts_ptr[z];
        const int slast = zone_pts_ptr[z+1];
        const int sprev = (s == sbase ? slast : s) - 1;
        return sprev;
    }
    inline int mapSideToPt2(const int &s) const
    {
        return mapSideToPt2(s, map_side2pt1, map_side2zone, zone_pts_ptr);
    }
    int* map_side2zone;        // map: side -> zone
    int* map_side2edge;        // map: side -> edge

    inline int zoneNPts(const int &i) const
    {return zoneNPts(i, zone_pts_ptr);}
    static inline int zoneNPts(const int &i, const int* zone_pts_ptr)
    {return zone_pts_ptr[i+1] - zone_pts_ptr[i];}
    int* map_side2pt1;     // maps: side -> points 1 and 2
    // Compressed Row Storage (CRS) of zone to points/sides mapping
    int* zone_pts_val;     // := map_side2pt1_
    int* zone_pts_ptr;

    double2* zone_x;       // zone center coordinates
    double2* edge_x_pred;      // edge ctr coords, middle of cycle
    double2* zone_x_pred;      // zone ctr coords, middle of cycle
    double2* pt_x0;      // point coords, start of cycle
    double2* pt_x;
    double2* pt_x_pred;

    double* zone_area;
    double* zone_vol;
    double* side_area_pred;    // side area, middle of cycle
    double* zone_area_pred;    // zone area, middle of cycle
    double* zone_vol_pred;     // zone volume, middle of cycle
    double* zone_vol0;     // zone volume, start of cycle

    double2* side_surfp;   // side surface vector
    double* edge_len;      // edge length
    double* side_mass_frac;       // side mass fraction
    double* zone_dl;       // zone characteristic length

    inline int num_side_chunks() { return side_chunks_CRS.size() - 1;}
    std::vector<int> side_chunks_CRS;    // start/stop index for side chunks, compressed row storage
    inline int side_zone_chunks_first(int s)
    { return side_zone_chunks_first(s, map_side2zone, side_chunks_CRS); }
    static inline int side_zone_chunks_first(int s,
            const int* map_side2zone,
            const std::vector<int> side_chunks_CRS)
    { return map_side2zone[side_chunks_CRS[s]]; }
    inline int side_zone_chunks_last(int s)
    { return side_zone_chunks_last(s, map_side2zone, side_chunks_CRS); }
    static inline int side_zone_chunks_last(int s,
            const int* map_side2zone,
            const std::vector<int> side_chunks_CRS)
    { return map_side2zone[side_chunks_CRS[s+1]-1] + 1; }
    inline int num_pt_chunks() { return pt_chunks_CRS.size() - 1;}
    std::vector<int> pt_chunks_CRS;    // start/stop index for point chunks, compressed row storage
    inline int num_zone_chunks() { return zone_chunks_CRS.size() - 1;}
    std::vector<int> zone_chunks_CRS;    // start/stop index for zone chunks, compressed row storage


    // find plane with constant x, y value
    static std::vector<int> getXPlane(
            const double c,
            const int num_pts,
            const double2 *pt_x);
    static std::vector<int> getYPlane(
            const double c,
            const int num_pts,
            const double2 *pt_x);

    // compute chunks for a given plane
    static void getPlaneChunks(
            const std::vector<int>& mapbp,
            const std::vector<int> pt_chunks_CRS,
            std::vector<int>& pchbfirst,
            std::vector<int>& pchblast);

    // compute edge, zone centers
    static void calcCtrs(
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
            double2* zx);

    // compute side, corner, zone volumes
    static void calcVols(
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
            double* zvol);

    static void calcMedianMeshSurfVecs(
            const int sfirst,
            const int slast,
            const int* map_side2zone,
            const int* map_side2edge,
            const double2* edge_x_pred,
            const double2* zone_x_pred,
            double2* side_surfp);

    static void calcEdgeLen(
            const int sfirst,
            const int slast,
            const int* map_side2pt1,
            const int* map_side2edge,
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const double2* pt_x_pred,
            double* edge_len);

    static void calcCharacteristicLen(
            const int sfirst,
            const int slast,
            const int* map_side2zone,
            const int* map_side2edge,
            const int* zone_pts_ptr,
            const double* side_area_pred,
            const double* edge_len,
            const int num_sides,
            const int num_zones,
            double* zone_dl);

    void sumCornersToPoints(LogicalStructured& sides_and_corners, DoCycleTasksArgsSerializer& serial);

    static void sumOnProc(
            const double* corner_mass,
            const double2* corner_force,
            const std::vector<int> pt_chunks_CRS,
            const int* map_pt2crn_first,
            const int* map_crn2crn_next,
            const GenerateMesh* generate_mesh,
            DoubleAccessor pt_weighted_mass,
            Double2Accessor pt_force);

    LogicalUnstructured local_points_by_gid;
    ptr_t* point_local_to_globalID;
    LogicalStructured zone_pts;
    LogicalStructured zones;
    LogicalStructured sides;
    LogicalStructured points;
    LogicalStructured edges;

    double2* edge_x;       // edge center coordinates
    double* side_area;
    double* side_vol;
    double* side_vol_pred;     // side volume, middle of cycle

private:

	LogicalUnstructured pt_x_init_by_gid;

	// children
    GenerateMesh* generate_mesh;

    // point-to-corner inverse map is stored as a linked list...
    int* map_pt2crn_first;   // map:  point -> first corner
    int* map_crn2crn_next;    // map:  corner -> next corner

    PhaseBarrier pbarrier_as_master;
    std::vector<PhaseBarrier> masters_pbarriers;
    int num_slaves;
    std::vector<int> master_colors;
    std::vector<int> slave_colors;

    Context ctx;
    HighLevelRuntime* runtime;

    std::vector<LogicalUnstructured> halos_points;
    std::vector<PhysicalRegion> pregions_halos;
    std::vector<LogicalUnstructured> slaved_halo_points;
    std::vector<LogicalUnstructured> local_halos_points;

    const int num_subregions;
    const int my_color; // TODO not used?

    void init();

    void initSideMappingArrays(const std::vector<int>& cellstart,
            const std::vector<int>& cellnodes);

    void initEdgeMappingArrays();

    void populateChunks();

    void populateInverseMap();

    void initParallel();

    void writeMeshStats();

    void calcSideMassFracs(const int side_chunk);

    void  allocateFields();

}; // class LocalMesh



#endif /* LOCALMESH_HH_ */
