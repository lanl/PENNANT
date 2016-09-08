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

class Mesh {
public:

    Mesh(const InputParameters& params,
		LogicalUnstructured& ispace_zones,
    		LogicalUnstructured& sides,
   		LogicalUnstructured& pts,
    		LogicalUnstructured& zone_pts_crs,
    		const PhysicalRegion &ghost_pts,
        Context ctx, HighLevelRuntime* rt);
    ~Mesh();

    // parameters
    int chunk_size;                 // max size for processing chunks
    double subregion_xmin; 		   // bounding box for a subregion
    double subregion_xmax; 		   // if xmin != std::numeric_limits<double>::max(),
    double subregion_ymin;         // should have 4 entries:
    double subregion_ymax; 		   // xmin, xmax, ymin, ymax

    // mesh variables
    // (See documentation for more details on the mesh
    //  data structures...)
    int num_pts_, num_edges_, num_zones_, num_sides_, num_corners_;
                       // number of points, edges, zones,
                       // sides, corners, resp.
    int num_bad_sides;       // number of bad sides (negative volume)

    inline int mapSideToSideNext(const int &s) const
    {
    	const int z = map_side2zone_[s];
    	const int sbase = zone_pts_ptr_[z];
    	const int slast = zone_pts_ptr_[z+1];
    	const int snext = (s + 1 == slast ? sbase : s + 1);
    	return snext;
    }
    inline int mapSideToSidePrev(const int &s) const
    {
    	const int z = map_side2zone_[s];
    	const int sbase = zone_pts_ptr_[z];
    	const int slast = zone_pts_ptr_[z+1];
    	const int sprev = (s == sbase ? slast : s) - 1;
    	return sprev;
    }
    inline int mapSideToPt2(const int &s) const
    {
    	return map_side2pt1_[mapSideToSideNext(s)];
    }
    int* map_side2zone_;        // map: side -> zone
    int* map_side2edge_;        // map: side -> edge

    inline int zoneNPts(const int &i) const
    {return zone_pts_ptr_[i+1] - zone_pts_ptr_[i];}        // number of points in zone
    int* map_side2pt1_;  	// maps: side -> points 1 and 2
    // Compressed Row Storage (CRS) of zone to points/sides mapping
    int* zone_pts_val_;		// := map_side2pt1_
    int* zone_pts_ptr_;

    double2* zone_x_;       // zone center coordinates
    double2* edge_x_pred;      // edge ctr coords, middle of cycle
    double2* zone_x_pred;      // zone ctr coords, middle of cycle
    double2* pt_x0;      // point coords, start of cycle

    double* zone_area_;
    double* zone_vol_;
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

    LogicalUnstructured zone_points;
    LogicalUnstructured local_points;

private:

	LogicalUnstructured pt_x_init;
	LogicalUnstructured zone_pts_ptr_CRS;

	// children
    GenerateMesh* generate_mesh;

    // point-to-corner inverse map is stored as a linked list...
    int* map_pt2crn_first;   // map:  point -> first corner
    int* map_crn2crn_next;    // map:  corner -> next corner

    // mpi comm variables
    int num_mesg_send2master;     // number of messages mype sends to master pes
    int num_slave_pes;      // number of messages mype receives from slave pes
    int num_proxies;        // number of proxies on mype
    int num_slaves;        // number of slaves on mype
    int* map_slave_pe2global_pe;   // map: slave pe -> (global) pe
    int* map_slave_pe2prox1; // map: slave pe -> first proxy in proxy buffer
    int* map_prox2master_pt;      // map: proxy -> corresponding (master) point
    int* slave_pe_num_prox;  // number of proxies for each slave pe
    int* map_master_pe2globale_pe;  // map: master pe -> (global) pe
    int* master_pe_num_slaves; // number of slaves for each master pe
    int* map_master_pe2slave1;// map: master pe -> first slave in slave buffer
    int* map_slave2pt;      // map: slave -> corresponding (slave) point

    double2* edge_x;       // edge center coordinates
    double* side_area_;
    double* side_vol_;
    double* side_vol_pred;     // side volume, middle of cycle

    Context ctx;
    HighLevelRuntime* runtime;

    LogicalUnstructured zones;
    const PhysicalRegion& ghost_points;

    const int num_subregions;
    const int my_PE;

    void init();

    void initSideMappingArrays();

    void initEdgeMappingArrays();

    void populateChunks();

    void populateInverseMap();

    void initParallel(
            const std::vector<int>& slavemstrpes,
            const std::vector<int>& slavemstrcounts,
            const std::vector<int>& slavepoints,
            const std::vector<int>& masterslvpes,
            const std::vector<int>& masterslvcounts,
            const std::vector<int>& masterpoints);

    void writeMeshStats();

    void calcSideMassFracs(const int side_chunk);

    // helper routines for sumToPoints
    template <typename T>
    void sumOnProc(
            const T* cvar,
			RegionAccessor<AccessorType::Generic, T>& pvar);

    //template <typename T>
    //void sumAcrossProcs(T* pvar);

    template <typename T>
    void parallelGather(
            const T* pvar,
            T* prxvar);

    template <typename T>
    void parallelSum(
            T* pvar,
            T* prxvar);

    template <typename T>
    void parallelScatter(
            T* pvar,
            const T* prxvar);
}; // class Mesh



#endif /* MESH_HH_ */
