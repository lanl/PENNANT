/*
 * GenerateGlobalMesh.hh
 *
 *  Created on: Aug 16, 2016
 *      Author: jgraham
 */

#ifndef SRC_GENERATEGLOBALMESH_HH_
#define SRC_GENERATEGLOBALMESH_HH_

#include "Parallel.hh"

class GenerateGlobalMesh {
public:
    GenerateGlobalMesh(const InputParameters& params);
	~GenerateGlobalMesh();

    std::string meshtype_;                  // generated mesh type
    int global_nzones_x_, global_nzones_y_; // global number of zones, in x and y
                                            // directions
    double len_x_, len_y_;                  // length of mesh sides, in x and y
                                            // directions
    int num_proc_x_, num_proc_y_;           // number of PEs to use, in x and y
                                            // directions

    int numberOfZones() const;
	void colorPartitions(Coloring *local_zones_map,
			Coloring *local_pts_map);
    int numberOfPoints() const;

private:
    void calcPartitions();
    void colorPartitionsRect(Coloring *local_zones_map,
			Coloring *local_pts_map);
	void colorPartitionsPie(Coloring *local_zones_map,
			Coloring *local_pts_map);
	void colorPartitionsHex(Coloring *local_zones_map,
			Coloring *local_pts_map);
    int numberOfPointsRect() const;
    int numberOfPointsPie() const;
    int numberOfPointsHex() const;

};

#endif /* SRC_GENERATEGLOBALMESH_HH_ */
