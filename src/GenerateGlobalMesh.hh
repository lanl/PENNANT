/*
 * GenerateGlobalMesh.hh
 *
 *  Created on: Aug 16, 2016
 *      Author: jgraham
 */

#ifndef SRC_GENERATEGLOBALMESH_HH_
#define SRC_GENERATEGLOBALMESH_HH_


#include "GenerateMesh.hh"
#include "Parallel.hh"


class GenerateGlobalMesh : public GenerateMesh {
public:
    GenerateGlobalMesh(const InputParameters& params);
    void setupHaloCommunication(
            int color,
            std::vector<int>* masters,
            std::vector<int>* slaves,
            Coloring* halo_pts_map);
    int numberOfZones() const;
    int numberOfPoints() const;
    int numberOfSides() const;
	void colorPartitions(
			Coloring* local_zones_map,
			Coloring* local_pts_map) const;

private:
    void colorZones(Coloring* local_zones_map) const;
    void colorPartitionsRect(
    			Coloring* local_zones_map,
			Coloring* local_pts_map) const;
	void colorPartitionsPie(
			Coloring* local_zones_map,
			Coloring* local_pts_map) const;
	void colorPartitionsHex(
			Coloring* local_zones_map,
			Coloring* local_pts_map) const;
    int numberOfPointsRect() const;
    int numberOfPointsPie() const;
    int numberOfPointsHex() const;
    int numberOfCornersRect() const;
    int numberOfCornersPie() const;
    int numberOfCornersHex() const;

};

#endif /* SRC_GENERATEGLOBALMESH_HH_ */
