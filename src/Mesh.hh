/*
 * Mesh.hh
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef MESH_HH_
#define MESH_HH_

#include <string>
#include <vector>

#include "InputParameters.hh"
#include "LogicalUnstructured.hh"
#include "Parallel.hh"
#include "Vec2.hh"

// forward declarations
class InputFile;
class GenerateMesh;

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
    int num_bad_sides;       // number of bad sides (negative volume)

    inline int mapSideToSideNext(const int &s) const
    {
    	const int z = map_side2zone[s];
    	const int sbase = zone_pts_ptr[z];
    	const int slast = zone_pts_ptr[z+1];
    	const int snext = (s + 1 == slast ? sbase : s + 1);
    	return snext;
    }
    inline int mapSideToSidePrev(const int &s) const
    {
    	const int z = map_side2zone[s];
    	const int sbase = zone_pts_ptr[z];
    	const int slast = zone_pts_ptr[z+1];
    	const int sprev = (s == sbase ? slast : s) - 1;
    	return sprev;
    }
    inline int mapSideToPt2(const int &s) const
    {
    	return map_side2pt1[mapSideToSideNext(s)];
    }
    int* map_side2zone;        // map: side -> zone
    int* map_side2edge;        // map: side -> edge

    inline int zoneNPts(const int &i) const
    {return zone_pts_ptr[i+1] - zone_pts_ptr[i];}        // number of points in zone
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

    int num_side_chunks;                    // number of side chunks
    std::vector<int> side_chunks_first;    // start/stop index for side chunks
    std::vector<int> side_chunks_last;
    std::vector<int> zone_chunks_first;    // start/stop index for zone chunks
    std::vector<int> zone_chunks_last;
    int num_pt_chunks;                    // number of point chunks
    std::vector<int> pt_chunks_first;    // start/stop index for point chunks
    std::vector<int> pt_chunks_last;
    int num_zone_chunks;                    // number of zone chunks
    std::vector<int> zone_chunk_first;    // start/stop index for zone chunks
    std::vector<int> zone_chunk_last;


    // find plane with constant x, y value
    std::vector<int> getXPlane(const double c);
    std::vector<int> getYPlane(const double c);

    // compute chunks for a given plane
    void getPlaneChunks(
            const int numb,
            const int* mapbp,
            std::vector<int>& pchbfirst,
            std::vector<int>& pchblast);

    // compute edge, zone centers
    void calcCtrs(const int side_chunk, const bool pred=true);

    // compute side, corner, zone volumes
    void calcVols(const int side_chunk, const bool pred=true);

    // check to see if previous volume computation had any
    // sides with negative volumes
    void checkBadSides();

    void calcMedianMeshSurfVecs(const int side_chunk);

    void calcEdgeLen(const int side_chunk);

    void calcCharacteristicLen(const int side_chunk);

    // sum corner variables to points (double or double2)
    void sumToPoints(
            const double* corner_mass,
            const double2* corner_force);

    LogicalUnstructured local_points_by_gid;
    ptr_t* point_local_to_globalID;

private:

	LogicalUnstructured pt_x_init_by_gid;

	// children
    GenerateMesh* generate_mesh;

    // point-to-corner inverse map is stored as a linked list...
    int* map_pt2crn_first;   // map:  point -> first corner
    int* map_crn2crn_next;    // map:  corner -> next corner

    double2* edge_x;       // edge center coordinates
    double* side_area;
    double* side_vol;
    double* side_vol_pred;     // side volume, middle of cycle

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

    const int num_subregions;
    const int my_color;

    void init();

    void initSideMappingArrays(const std::vector<int>& cellstart,
            const std::vector<int>& cellnodes);

    void initEdgeMappingArrays();

    void populateChunks();

    void populateInverseMap();

    void initParallel();

    void writeMeshStats();

    void calcSideMassFracs(const int side_chunk);

    // helper routines for sumToPoints
    template <typename T>
    void sumOnProc(
            const T* cvar,
			RegionAccessor<AccessorType::Generic, T>& pvar);

}; // class Mesh



#endif /* MESH_HH_ */
