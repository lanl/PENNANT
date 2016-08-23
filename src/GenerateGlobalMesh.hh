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
    void generate(std::vector<double2>& pointpos) const;

    int numberOfZones() const;
	void colorPartitions(Coloring *local_zones_map,
			Coloring *local_pts_map) const;
    int numberOfPoints() const;

private:
    void calcPartitions();
    void colorPartitionsRect(Coloring *local_zones_map,
			Coloring *local_pts_map) const;
	void colorPartitionsPie(Coloring *local_zones_map,
			Coloring *local_pts_map) const;
	void colorPartitionsHex(Coloring *local_zones_map,
			Coloring *local_pts_map) const;
    int numberOfPointsRect() const;
    int numberOfPointsPie() const;
    int numberOfPointsHex() const;
    void generateRect(
            std::vector<double2>& pointpos) const;
    void generatePie(
            std::vector<double2>& pointpos) const;
    void generateHex(
            std::vector<double2>& pointpos) const;

};

#endif /* SRC_GENERATEGLOBALMESH_HH_ */
