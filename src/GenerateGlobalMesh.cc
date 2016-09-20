/*
 * GenerateGlobalMesh.cc
 *
 *  Created on: Aug 16, 2016
 *      Author: jgraham
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
    if (meshtype_ == "pie")
        return numberOfPointsPie();
    else if (meshtype_ == "rect")
    	return numberOfPointsRect();
    else if (meshtype_ == "hex")
    	return numberOfPointsHex();
    else
    	return -1;
}


int GenerateGlobalMesh::numberOfPointsRect() const {
    return (global_nzones_x_ + 1) * (global_nzones_y_ + 1);
}


int GenerateGlobalMesh::numberOfPointsPie() const {
    return (global_nzones_x_ + 1) * global_nzones_y_ + 1;
}


int GenerateGlobalMesh::numberOfPointsHex() const {
    return 2 * global_nzones_x_ * global_nzones_y_ + 2;
}


int GenerateGlobalMesh::numberOfZones() const {
	return global_nzones_x_ * global_nzones_y_;
}


void GenerateGlobalMesh::colorPartitions(
		Coloring *zone_map, Coloring *pt_map) const
{
    if (meshtype_ == "pie")
    		colorPartitionsPie(zone_map, pt_map);
    else if (meshtype_ == "rect")
    		colorPartitionsRect(zone_map, pt_map);
    else if (meshtype_ == "hex")
    		colorPartitionsHex(zone_map, pt_map);
}

void GenerateGlobalMesh::sharePoints(int color,
		std::vector<int>* neighbors,
		Coloring *shared_pts) const
{
    if (meshtype_ == "pie")
    		sharePointsPie(color, neighbors, shared_pts);
    else if (meshtype_ == "rect")
		sharePointsRect(color, neighbors, shared_pts);
    else if (meshtype_ == "hex")
		sharePointsHex(color, neighbors, shared_pts);
}


void GenerateGlobalMesh::colorPartitionsPie(
		Coloring *zone_map, Coloring *pt_map) const
{
	colorZones(zone_map);

	for (int proc_index_y = 0; proc_index_y < num_proc_y_; proc_index_y++) {
		const int zone_y_start = y_start(proc_index_y);
		const int zone_y_stop = y_start(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x_; proc_index_x++) {
			const int color = proc_index_y * num_proc_x_ + proc_index_x;
			const int zone_x_start = x_start(proc_index_x);
			const int zone_x_stop = x_start(proc_index_x + 1);
			for (int j = zone_y_start; j <= zone_y_stop; j++) {
				if (j == 0) {
					(*pt_map)[color].points.insert(0);
					continue;
				}
				for (int i = zone_x_start; i <= zone_x_stop; i++) {
					int pt = 1 + (j - 1) * (global_nzones_x_ + 1) + i;
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

	for (int proc_index_y = 0; proc_index_y < num_proc_y_; proc_index_y++) {
		const int zone_y_start = y_start(proc_index_y);
		const int zone_y_stop = y_start(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x_; proc_index_x++) {
			const int color = proc_index_y * num_proc_x_ + proc_index_x;
			const int zone_x_start = x_start(proc_index_x);
			const int zone_x_stop = x_start(proc_index_x + 1);
			for (int gj = zone_y_start; gj <= zone_y_stop; gj++) {
				for (int gi = zone_x_start; gi <= zone_x_stop; gi++) {

					if (gj == 0) {
						(*pt_map)[color].points.insert(gi);
					} else {
						int pt = (2 * gj - 1) * global_nzones_x_ + 1;
						if (gi == 0 || gj == global_nzones_y_) {
							pt += gi;
							(*pt_map)[color].points.insert(pt);
						} else {
							pt += 2 * gi - 1;
							if (gi == global_nzones_x_)
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

	for (int proc_index_y = 0; proc_index_y < num_proc_y_; proc_index_y++) {
		const int zone_y_start = y_start(proc_index_y);
		const int zone_y_stop = y_start(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x_; proc_index_x++) {
			const int color = proc_index_y * num_proc_x_ + proc_index_x;
			const int zone_x_start = x_start(proc_index_x);
			const int zone_x_stop = x_start(proc_index_x + 1);
			for (int j = zone_y_start; j <= zone_y_stop; j++) {
				for (int i = zone_x_start; i <= zone_x_stop; i++) {
					int pt = j * (global_nzones_x_ + 1) + i;
					(*local_pt_map)[color].points.insert(pt);
				}
			}
		}
	}
}

void GenerateGlobalMesh::sharePointsPie(int color,
		std::vector<int>* neighbors,
		Coloring *shared_pts_map) const
{
	neighbors->push_back(color);   // need access to own ghost region
	const int proc_index_x = color % num_proc_x_;
	const int proc_index_y = color / num_proc_x_;
	const int zone_y_start = y_start(proc_index_y);
	const int zone_y_stop = y_start(proc_index_y + 1);
	const int zone_x_start = x_start(proc_index_x);
	const int zone_x_stop = x_start(proc_index_x + 1);
}

void GenerateGlobalMesh::sharePointsHex(int color,
		std::vector<int>* neighbors,
		Coloring *shared_pts_map) const
{
	neighbors->push_back(color);   // need access to own ghost region
	const int proc_index_x = color % num_proc_x_;
	const int proc_index_y = color / num_proc_x_;
	const int zone_y_start = y_start(proc_index_y);
	const int zone_y_stop = y_start(proc_index_y + 1);
	const int zone_x_start = x_start(proc_index_x);
	const int zone_x_stop = x_start(proc_index_x + 1);
}

void GenerateGlobalMesh::sharePointsRect(int color,
		std::vector<int>* neighbors,
		Coloring *shared_pts_map) const
{
	neighbors->push_back(color);   // need access to own ghost region
	const int proc_index_x = color % num_proc_x_;
	const int proc_index_y = color / num_proc_x_;
	const int zone_y_start = y_start(proc_index_y);
	const int zone_y_stop = y_start(proc_index_y + 1);
	const int zone_x_start = x_start(proc_index_x);
	const int zone_x_stop = x_start(proc_index_x + 1);

    const int local_origin = zone_y_start * (global_nzones_x_ + 1) + zone_x_start;

    // enumerate slave points
    // slave point with master at lower left
    if (proc_index_x != 0 && proc_index_y != 0) {
        int mstrpe = color - num_proc_x_ - 1;
        int pt = local_origin;
        (*shared_pts_map)[color].points.insert(pt);
        neighbors->push_back(mstrpe);
    }
    // slave points with master below
    if (proc_index_y != 0) {
        int mstrpe = color - num_proc_x_;
        int p = local_origin;
        for (int i = zone_x_start; i <= zone_x_stop; ++i) {
            if (i == zone_x_start && proc_index_x != 0) { p++; continue; }
            (*shared_pts_map)[color].points.insert(p);
            p++;
        }
        neighbors->push_back(mstrpe);
    }
    // slave points with master to left
    if (proc_index_x != 0) {
        int mstrpe = color - 1;
        int p = local_origin;
        for (int j = zone_y_start; j <= zone_y_stop; ++j) {
            if (j == zone_y_start && proc_index_y != 0) { p += global_nzones_x_ + 1; continue; }
            (*shared_pts_map)[color].points.insert(p);
            p += global_nzones_x_ + 1;
        }
        neighbors->push_back(mstrpe);
    }

    // enumerate master points
    // master points with slave to right
    if (proc_index_x != num_proc_x_ - 1) {
        int slvpe = color + 1;
        int p = zone_y_start * (global_nzones_x_ + 1) + zone_x_stop;
        for (int j = zone_y_start; j <= zone_y_stop; ++j) {
            if (j == zone_y_start && proc_index_y != 0) { p += global_nzones_x_ + 1; continue; }
            (*shared_pts_map)[color].points.insert(p);
            p += global_nzones_x_ + 1;
        }
        neighbors->push_back(slvpe);
    }
    // master points with slave above
    if (proc_index_y != num_proc_y_ - 1) {
        int slvpe = color + num_proc_x_;
        int p = zone_y_stop * (global_nzones_x_ + 1) + zone_x_start;
        for (int i = zone_x_start; i <= zone_x_stop; ++i) {
            if (i == zone_x_start && proc_index_x != 0) { p++; continue; }
            (*shared_pts_map)[color].points.insert(p);
            p++;
        }
        neighbors->push_back(slvpe);
    }
    // master point with slave at upper right
    if (proc_index_x != num_proc_x_ - 1 && proc_index_y != num_proc_y_ - 1) {
        int slvpe = color + num_proc_x_ + 1;
        int p = zone_y_stop * (global_nzones_x_ + 1) + zone_x_stop;
        (*shared_pts_map)[color].points.insert(p);
        neighbors->push_back(slvpe);
    }
}

void GenerateGlobalMesh::colorZones(Coloring *zone_map) const
{
	for (int proc_index_y = 0; proc_index_y < num_proc_y_; proc_index_y++) {
		const int zone_y_start = y_start(proc_index_y);
		const int zone_y_stop = y_start(proc_index_y + 1);
		for (int proc_index_x = 0; proc_index_x < num_proc_x_; proc_index_x++) {
			const int color = proc_index_y * num_proc_x_ + proc_index_x;
			const int zone_x_start = x_start(proc_index_x);
			const int zone_x_stop = x_start(proc_index_x + 1);
			for (int j = zone_y_start; j < zone_y_stop; j++) {
				for (int i = zone_x_start; i < zone_x_stop; i++) {
					int zone = j * (global_nzones_x_) + i;
					(*zone_map)[color].points.insert(zone);
				}
			}
		}
	}
}


int GenerateGlobalMesh::numberOfSides() const {
    if (meshtype_ == "pie")
    	return numberOfCornersPie();
    else if (meshtype_ == "rect")
    	return numberOfCornersRect();
    else if (meshtype_ == "hex")
    	return numberOfCornersHex();
    else
    	return -1;
}


int GenerateGlobalMesh::numberOfCornersRect() const {
    return 4 * numberOfZones();
}


int GenerateGlobalMesh::numberOfCornersPie() const {
    return 4 * numberOfZones() - global_nzones_x_;
}


int GenerateGlobalMesh::numberOfCornersHex() const {
    return 6 * numberOfZones() - 2 * global_nzones_x_ - 2 * global_nzones_y_ + 2;
}
