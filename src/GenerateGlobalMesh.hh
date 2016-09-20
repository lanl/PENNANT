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
    int numberOfZones() const;
    int numberOfPoints() const;
    int numberOfSides() const;
	void colorPartitions(
			Coloring *local_zones_map,
			Coloring *local_pts_map) const;
	void sharePoints(
			int color,
			std::vector<int>* neighbors,
			Coloring *shared_pts) const;

private:
    std::string meshtype_;                  // generated mesh type
    int global_nzones_x_, global_nzones_y_; // global number of zones, in x and y
                                            // directions
    double len_x_, len_y_;                  // length of mesh sides, in x and y
                                            // directions
    int num_proc_x_, num_proc_y_;           // number of PEs to use, in x and y
                                            // directions
	const int num_subregions_;

    void calcPartitions();
    void colorZones(Coloring *local_zones_map) const;
    void colorPartitionsRect(
    			Coloring *local_zones_map,
			Coloring *local_pts_map) const;
	void colorPartitionsPie(
			Coloring *local_zones_map,
			Coloring *local_pts_map) const;
	void colorPartitionsHex(
			Coloring *local_zones_map,
			Coloring *local_pts_map) const;
	void sharePointsRect(
			int color,
			std::vector<int>* neighbors,
			Coloring *shared_pts) const;
	void sharePointsPie(
			int color,
			std::vector<int>* neighbors,
			Coloring *shared_pts) const;
	void sharePointsHex(
			int color,
			std::vector<int>* neighbors,
			Coloring *shared_pts) const;
    int numberOfPointsRect() const;
    int numberOfPointsPie() const;
    int numberOfPointsHex() const;
    int numberOfCornersRect() const;
    int numberOfCornersPie() const;
    int numberOfCornersHex() const;

    inline int y_start(int proc_index_y) const
    { return proc_index_y * global_nzones_y_ / num_proc_y_; }

    inline int x_start(int proc_index_x) const
    { return proc_index_x * global_nzones_x_ / num_proc_x_; }

};

#endif /* SRC_GENERATEGLOBALMESH_HH_ */
