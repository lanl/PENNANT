/*
 * GenerateGlobalMesh.cc
 *
 *  Created on: Aug 16, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#include "GenerateGlobalMesh.hh"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

GenerateGlobalMesh::GenerateGlobalMesh(const InputParameters& input_params) :
		GenerateMesh(input_params)
{
}


int GenerateGlobalMesh::numberOfPoints() const {
    if (meshtype == "pie")
        return numberOfPointsPie();
    else if (meshtype == "rect")
    	return numberOfPointsRect();
    else if (meshtype == "hex")
    	return numberOfPointsHex();
    else
    	return -1;
}


int GenerateGlobalMesh::numberOfPointsRect() const {
    return (global_nzones_x + 1) * (global_nzones_y + 1);
}


int GenerateGlobalMesh::numberOfPointsPie() const {
    return (global_nzones_x + 1) * global_nzones_y + 1;
}


int GenerateGlobalMesh::numberOfPointsHex() const {
    return 2 * global_nzones_x * global_nzones_y + 2;
}


int GenerateGlobalMesh::numberOfZones() const {
	return global_nzones_x * global_nzones_y;
}


void GenerateGlobalMesh::colorPartitions(
		Coloring *zone_map, Coloring *pt_map) const
{
    if (meshtype == "pie")
    		colorPartitionsPie(zone_map, pt_map);
    else if (meshtype == "rect")
    		colorPartitionsRect(zone_map, pt_map);
    else if (meshtype == "hex")
    		colorPartitionsHex(zone_map, pt_map);
}

void GenerateGlobalMesh::setupHaloCommunication(
        int color,
        std::vector<int>* master_colors,
        std::vector<int>* slave_colors,
		Coloring* halo_pts_map)
{
        calcLocalConstants(color);

        std::vector<int> slaved_points;
        std::vector<int> slaved_points_counts;
        std::vector<int> master_points;
        std::vector<int> master_points_counts;

        generateHaloPoints(*master_colors, slaved_points_counts, slaved_points,
                *slave_colors, master_points_counts, master_points);

        for (int i = 0; i < master_points.size(); i++) {
            (*halo_pts_map)[color].points.insert(pointLocalToGlobalID(master_points[i]));
        }
}


void GenerateGlobalMesh::colorPartitionsPie(
		Coloring *zone_map, Coloring *pt_map) const
{
	colorZones(zone_map);

	for (int proc_index_y = 0; proc_index_y < num_proc_y; proc_index_y++) {
		const int zone_y_start = yStart(proc_index_y);
		const int zone_y_stop = yStart(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x; proc_index_x++) {
			const int color = proc_index_y * num_proc_x + proc_index_x;
			const int zone_x_start = xStart(proc_index_x);
			const int zone_x_stop = xStart(proc_index_x + 1);
			for (int j = zone_y_start; j <= zone_y_stop; j++) {
				if (j == 0) {
					(*pt_map)[color].points.insert(0);
					continue;
				}
				for (int i = zone_x_start; i <= zone_x_stop; i++) {
					int pt = 1 + (j - 1) * (global_nzones_x + 1) + i;
					(*pt_map)[color].points.insert(pt);
				}
			}
		}
	}
}


void GenerateGlobalMesh::colorPartitionsHex(
		Coloring *zone_map, Coloring *pt_map) const
{
	colorZones(zone_map);

	for (int proc_index_y = 0; proc_index_y < num_proc_y; proc_index_y++) {
		const int zone_y_start = yStart(proc_index_y);
		const int zone_y_stop = yStart(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x; proc_index_x++) {
			const int color = proc_index_y * num_proc_x + proc_index_x;
			const int zone_x_start = xStart(proc_index_x);
			const int zone_x_stop = xStart(proc_index_x + 1);
			for (int gj = zone_y_start; gj <= zone_y_stop; gj++) {
				for (int gi = zone_x_start; gi <= zone_x_stop; gi++) {

					if (gj == 0) {
						(*pt_map)[color].points.insert(gi);
					} else {
						int pt = numPointsPreviousRowsNonZeroJHex(gj);
						if (gi == 0 || gj == global_nzones_y) {
							pt += gi;
							(*pt_map)[color].points.insert(pt);
						} else {
							pt += 2 * gi - 1;
							if (gi == global_nzones_x)
								(*pt_map)[color].points.insert(pt);
							else if (gi == zone_x_stop && gj == zone_y_start)
								(*pt_map)[color].points.insert(pt);
					        else if (gi == zone_x_start && gj == zone_y_stop)
								(*pt_map)[color].points.insert(pt+1);
				            else {
								(*pt_map)[color].points.insert(pt);
								(*pt_map)[color].points.insert(pt+1);
				            }
						}
					} // gj != 0
				} // gi
			} // gy
		} // proc_index_x
	} // proc_index_y
}


void GenerateGlobalMesh::colorPartitionsRect(
		Coloring *zone_map, Coloring *local_pt_map) const
{
	colorZones(zone_map);

	for (int proc_index_y = 0; proc_index_y < num_proc_y; proc_index_y++) {
		const int zone_y_start = yStart(proc_index_y);
		const int zone_y_stop = yStart(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x; proc_index_x++) {
			const int color = proc_index_y * num_proc_x + proc_index_x;
			const int zone_x_start = xStart(proc_index_x);
			const int zone_x_stop = xStart(proc_index_x + 1);
			for (int j = zone_y_start; j <= zone_y_stop; j++) {
				for (int i = zone_x_start; i <= zone_x_stop; i++) {
					int pt = j * (global_nzones_x + 1) + i;
					(*local_pt_map)[color].points.insert(pt);
				}
			}
		}
	}
}


void GenerateGlobalMesh::colorZones(Coloring *zone_map) const
{
	for (int proc_index_y = 0; proc_index_y < num_proc_y; proc_index_y++) {
		const int zone_y_start = yStart(proc_index_y);
		const int zone_y_stop = yStart(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x; proc_index_x++) {
			const int color = proc_index_y * num_proc_x + proc_index_x;
			const int zone_x_start = xStart(proc_index_x);
			const int zone_x_stop = xStart(proc_index_x + 1);
			for (int j = zone_y_start; j < zone_y_stop; j++) {
				for (int i = zone_x_start; i < zone_x_stop; i++) {
					int zone = j * (global_nzones_x) + i;
					(*zone_map)[color].points.insert(zone);
				}
			}
		}
	}
}


int GenerateGlobalMesh::numberOfSides() const {
    if (meshtype == "pie")
    	return numberOfCornersPie();
    else if (meshtype == "rect")
    	return numberOfCornersRect();
    else if (meshtype == "hex")
    	return numberOfCornersHex();
    else
    	return -1;
}


int GenerateGlobalMesh::numberOfCornersRect() const {
    return 4 * numberOfZones();
}


int GenerateGlobalMesh::numberOfCornersPie() const {
    return 4 * numberOfZones() - global_nzones_x;
}


int GenerateGlobalMesh::numberOfCornersHex() const {
    return 6 * numberOfZones() - 2 * global_nzones_x - 2 * global_nzones_y + 2;
}
