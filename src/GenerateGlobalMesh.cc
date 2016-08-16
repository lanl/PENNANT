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
	meshtype_(input_params.meshtype_),
	global_nzones_x_(input_params.directs_.nzones_x_),
	global_nzones_y_(input_params.directs_.nzones_y_),
	len_x_(input_params.directs_.len_x_),
	len_y_(input_params.directs_.len_y_)
{
    calcPartitions();
}

GenerateGlobalMesh::~GenerateGlobalMesh() {}

int GenerateGlobalMesh::numberOfZones() const {
	return global_nzones_x_ * global_nzones_y_;
}

void GenerateGlobalMesh::colorPartitions(Coloring *zone_map)
{
    if (meshtype_ == "pie")
    		colorPartitionsPie(zone_map);
    else if (meshtype_ == "rect")
    		colorPartitionsRect(zone_map);
    else if (meshtype_ == "hex")
    		colorPartitionsHex(zone_map);
}

void GenerateGlobalMesh::colorPartitionsPie(Coloring *zone_map)
{
	colorPartitionsRect(zone_map);
}

void GenerateGlobalMesh::colorPartitionsHex(Coloring *zone_map)
{
	colorPartitionsRect(zone_map);
}

void GenerateGlobalMesh::colorPartitionsRect(Coloring *zone_map)
{
	for (int proc_index_y = 0; proc_index_y < num_proc_y_; proc_index_y++) {
		for (int proc_index_x = 0; proc_index_x < num_proc_x_; proc_index_x++) {
			const int color = proc_index_y * num_proc_x_ + proc_index_x;
			const int zone_x_start = proc_index_x * global_nzones_x_ / num_proc_x_;
			const int zone_x_stop = (proc_index_x + 1) * global_nzones_x_ / num_proc_x_;
			const int zone_y_start = proc_index_y * global_nzones_y_ / num_proc_y_;
			const int zone_y_stop = (proc_index_y + 1) * global_nzones_y_ / num_proc_y_;
			for (int j = zone_y_start; j < zone_y_stop; j++) {
				for (int i = zone_x_start; i <= zone_x_stop; i++) {
					int zone = j * (global_nzones_x_) + i;
					(*zone_map)[color].points.insert(zone);
				}
			}
		}
	}
}


void GenerateGlobalMesh::calcPartitions() {

    // pick numpex, numpey such that PE blocks are as close to square
    // as possible
    // we would like:  gnzx / numpex == gnzy / numpey,
    // where numpex * numpey = numpe (total number of PEs available)
    // this solves to:  numpex = sqrt(numpe * gnzx / gnzy)
    // we compute this, assuming gnzx <= gnzy (swap if necessary)
    double nx = static_cast<double>(global_nzones_x_);
    double ny = static_cast<double>(global_nzones_y_);
    bool swapflag = (nx > ny);
    if (swapflag) swap(nx, ny);
    double n = sqrt(Parallel::num_subregions() * nx / ny);
    // need to constrain n to be an integer with numpe % n == 0
    // try rounding n both up and down
    int n1 = floor(n + 1.e-12);
    n1 = max(n1, 1);
    while (Parallel::num_subregions() % n1 != 0) --n1;
    int n2 = ceil(n - 1.e-12);
    while (Parallel::num_subregions() % n2 != 0) ++n2;
    // pick whichever of n1 and n2 gives blocks closest to square,
    // i.e. gives the shortest long side
    double longside1 = max(nx / n1, ny / (Parallel::num_subregions()/n1));
    double longside2 = max(nx / n2, ny / (Parallel::num_subregions()/n2));
    num_proc_x_ = (longside1 <= longside2 ? n1 : n2);
    num_proc_y_ = Parallel::num_subregions() / num_proc_x_;
    if (swapflag) swap(num_proc_x_, num_proc_y_);

}


