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
   		IndexSpace pts,
        std::vector<LogicalUnstructured>& halos_points,
        std::vector<PhysicalRegion>& pregions_halos,
        PhaseBarrier pbarrier_as_master,
        std::vector<PhaseBarrier> masters_pbarriers,
        DynamicCollective add_reduction,
        Context ctx, HighLevelRuntime* rt);
    ~LocalMesh();

    // parameters
    const double subregion_xmin; 		   // bounding box for a subregion
    const double subregion_xmax; 		   // if xmin != std::numeric_limits<double>::max(),
    const double subregion_ymin;         // should have 4 entries:
    const double subregion_ymax; 		   // xmin, xmax, ymin, ymax

    // mesh variables
    // (See documentation for more details on the mesh
    //  data structures...)
    int num_pts, num_edges, num_zones, num_sides, num_corners;

    int num_pt_chunks;
    int num_zone_chunks;
    int num_side_chunks;

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

    static inline int zoneNPts(const int &i, const int* zone_pts_ptr)
    {return zone_pts_ptr[i+1] - zone_pts_ptr[i];}

    static inline int side_zone_chunks_first(int s,
            const int* map_side2zone,
            const int* side_chunks_CRS)
    { return map_side2zone[side_chunks_CRS[s]]; }

    static inline int side_zone_chunks_last(int s,
            const int* map_side2zone,
            const int* side_chunks_CRS)
    { return map_side2zone[side_chunks_CRS[s+1]-1] + 1; }

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
            const int* pt_chunks_CRS,
            const int num_pt_chunks,
            std::vector<int>& pchb_CRS);

    // compute edge, zone centers
    static void calcCtrs(
            const int sfirst,
            const int slast,
            const double2* px,
            const int* map_side2zone,
            const int num_sides,
            const int num_zones,
            const int* map_side2pt1,
            const int* map_side2pt2,
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
            const int* map_side2pt2,
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
            const int* map_side2pt2,
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
            const int* pt_chunks_CRS,
            const int num_pt_chunks,
            const int* map_pt2crn_first,
            const int* map_crn2crn_next,
            const ptr_t* pt_local2global,
            DoubleSOAAccessor pt_weighted_mass,
            Double2SOAAccessor pt_force);

    LogicalUnstructured local_points_by_gid;
    LogicalStructured zone_pts;
    LogicalStructured zones;
    LogicalStructured sides;
    LogicalStructured points;
    LogicalStructured edges;
    LogicalStructured zone_chunks;
    LogicalStructured side_chunks;
    LogicalStructured point_chunks;

private:
    int chunk_size;                 // max size for processing chunks

	LogicalUnstructured pt_x_init_by_gid;

    GenerateMesh* generate_mesh;

    PhaseBarrier pbarrier_as_master;
    std::vector<PhaseBarrier> masters_pbarriers;
    int num_slaves;
    std::vector<int> master_colors;
    std::vector<int> slave_colors;

    DynamicCollective add_reduction;
    Context ctx;
    HighLevelRuntime* runtime;

    std::vector<LogicalUnstructured> halos_points;
    std::vector<PhysicalRegion> pregions_halos;
    std::vector<LogicalUnstructured> slaved_halo_points;
    std::vector<LogicalUnstructured> local_halos_points;

    const int num_subregions;
    const int my_color;

    void init();

    void initSideMappingArrays(const std::vector<int>& cellstart,
            const std::vector<int>& cellnodes,
            int* map_side2zone,
            int* map_side2pt1,
            int* map_side2pt2);

    void initEdgeMappingArrays(
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const int* map_side2pt1,
            const int* map_side2pt2,
            int* map_side2edge);

    void populateChunks(const int* map_side2zone,
            int** point_chunks_CRS,
            int** side_chunks_CRS,
            int** zone_chunks_CRS);

    void populateInverseMap(const int* map_side2pt1,
            int* map_pt2crn_first,
            int* map_crn2crn_next);

    void initParallel(const ptr_t* pt_local2global);

    void writeMeshStats();

    void calcSideMassFracs(const int side_chunk,
            const int* map_side2zone,
            const double* side_area,
            const double* zone_area,
            const int* side_chunks_CRS,
            double* side_mass_frac);

    void  allocateFields();

}; // class LocalMesh



#endif /* LOCALMESH_HH_ */
