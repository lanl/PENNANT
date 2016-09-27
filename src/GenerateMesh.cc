/*
 * GenMesh.cc
 *
 *  Created on: Jun 4, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */


#include "GenerateMesh.hh"

#include <cassert>
#include <iostream>


using namespace std;


GenerateMesh::GenerateMesh(const InputParameters& input_params) :
	meshtype(input_params.meshtype),
	global_nzones_x(input_params.directs.nzones_x),
	global_nzones_y(input_params.directs.nzones_y),
	len_x(input_params.directs.len_x),
	len_y(input_params.directs.len_y),
	num_subregions(input_params.directs.ntasks),
	my_color(input_params.directs.task_id)
{
    calcPartitions();
	calcLocalConstants(my_color);

}


void GenerateMesh::generate(
        vector<double2>& pointpos,
        vector<int>& zonepoints_ptr_CRS,
        vector<int>& zonepoints) const {


    // mesh type-specific calculations
    vector<int> zonesize;
    string allowable_mesh_type = "!@#$&!*()@#$";
    if (meshtype == "pie")
        generatePie(pointpos, zonepoints_ptr_CRS, zonesize, zonepoints);
    else if (meshtype == "rect")
        generateRect(pointpos, zonepoints_ptr_CRS, zonesize, zonepoints);
    else if (meshtype == "hex")
        generateHex(pointpos, zonepoints_ptr_CRS, zonesize, zonepoints);
    else
        assert(meshtype != allowable_mesh_type);

    zonepoints_ptr_CRS.push_back(zonepoints_ptr_CRS.back()+zonesize.back());
}


void GenerateMesh::generateHaloPoints(
        vector<int>& master_colors,
        vector<int>& slaved_points_counts,
        vector<int>& slaved_points,
        vector<int>& slave_colors,
        vector<int>& master_points_counts,
        vector<int>& master_points) const {


    // mesh type-specific calculations
    vector<int> zonesize;
    string allowable_mesh_type = "!@#$&!*()@#$";
    if (meshtype == "pie")
        generateHaloPointsPie(
                master_colors, slaved_points_counts, slaved_points,
                slave_colors, master_points_counts, master_points);
    else if (meshtype == "rect")
        generateHaloPointsRect(
                master_colors, slaved_points_counts, slaved_points,
                slave_colors, master_points_counts, master_points);
    else if (meshtype == "hex")
        generateHaloPointsHex(
                master_colors, slaved_points_counts, slaved_points,
                slave_colors, master_points_counts, master_points);
    else
        assert(meshtype != allowable_mesh_type);
}


void GenerateMesh::generateRect(
        vector<double2>& pointpos,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints) const {

    const int np = num_points_x * num_points_y;

    // generate point coordinates
    pointpos.reserve(np);
    double dx = len_x / (double) global_nzones_x;
    double dy = len_y / (double) global_nzones_y;
    for (int j = 0; j < num_points_y; ++j) {
        double y = dy * (double) (j + zone_y_offset);
        for (int i = 0; i < num_points_x; ++i) {
            double x = dx * (double) (i + zone_x_offset);
            pointpos.push_back(make_double2(x, y));
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(num_zones);
    zonesize.reserve(num_zones);
    zonepoints.reserve(4 * num_zones);
    for (int j = 0; j < nzones_y; ++j) {
        for (int i = 0; i < nzones_x; ++i) {
            zonestart.push_back(zonepoints.size());
            zonesize.push_back(4);
            int p0 = j * num_points_x + i;
            zonepoints.push_back(p0);
            zonepoints.push_back(p0 + 1);
            zonepoints.push_back(p0 + num_points_x + 1);
            zonepoints.push_back(p0 + num_points_x);
       }
    }
}


void GenerateMesh::generatePie(
        vector<double2>& pointpos,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints) const {

    const int np = (proc_index_y == 0 ? num_points_x * (num_points_y - 1) + 1 : num_points_x * num_points_y);

    // generate point coordinates
    pointpos.reserve(np);
    double dth = len_x / (double) global_nzones_x;
    double dr  = len_y / (double) global_nzones_y;
    for (int j = 0; j < num_points_y; ++j) {
        if (j + zone_y_offset == 0) {
            pointpos.push_back(make_double2(0., 0.));
            continue;
        }
        double r = dr * (double) (j + zone_y_offset);
        for (int i = 0; i < num_points_x; ++i) {
            double th = dth * (double) (global_nzones_x - (i + zone_x_offset));
            double x = r * cos(th);
            double y = r * sin(th);
            pointpos.push_back(make_double2(x, y));
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(num_zones);
    zonesize.reserve(num_zones);
    zonepoints.reserve(4 * num_zones);
    for (int j = 0; j < nzones_y; ++j) {
        for (int i = 0; i < nzones_x; ++i) {
            zonestart.push_back(zonepoints.size());
            int p0 = j * num_points_x + i;
            if (proc_index_y == 0) p0 -= num_points_x - 1;
            if (j + zone_y_offset == 0) {
                zonesize.push_back(3);
                zonepoints.push_back(0);
            }
            else {
                zonesize.push_back(4);
                zonepoints.push_back(p0);
                zonepoints.push_back(p0 + 1);
            }
            zonepoints.push_back(p0 + num_points_x + 1);
            zonepoints.push_back(p0 + num_points_x);
        }
    }
}


void GenerateMesh::generateHex(
        vector<double2>& pointpos,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints) const {

    // generate point coordinates
    pointpos.reserve(2 * num_points_x * num_points_y);  // upper bound
    double dx = len_x / (double) (global_nzones_x - 1);
    double dy = len_y / (double) (global_nzones_y - 1);

    vector<int> pbase(num_points_y);
    for (int j = 0; j < num_points_y; ++j) {
        pbase[j] = pointpos.size();
        int gj = j + zone_y_offset;
        double y = dy * ((double) gj - 0.5);
        y = max(0., min(len_y, y));
        for (int i = 0; i < num_points_x; ++i) {
            int gi = i + zone_x_offset;
            double x = dx * ((double) gi - 0.5);
            x = max(0., min(len_x, x));
            if (gi == 0 || gi == global_nzones_x || gj == 0 || gj == global_nzones_y)
                pointpos.push_back(make_double2(x, y));
            else if (i == nzones_x && j == 0)
                pointpos.push_back(
                        make_double2(x - dx / 6., y + dy / 6.));
            else if (i == 0 && j == nzones_y)
                pointpos.push_back(
                        make_double2(x + dx / 6., y - dy / 6.));
            else {
                pointpos.push_back(
                        make_double2(x - dx / 6., y + dy / 6.));
                pointpos.push_back(
                        make_double2(x + dx / 6., y - dy / 6.));
            }
        } // for i
    } // for j
    int np = pointpos.size();

    // generate zone adjacency lists
    zonestart.reserve(num_zones);
    zonesize.reserve(num_zones);
    zonepoints.reserve(6 * num_zones);  // upper bound
    for (int j = 0; j < nzones_y; ++j) {
        int gj = j + zone_y_offset;
        int pbasel = pbase[j];
        int pbaseh = pbase[j+1];
        if (proc_index_x > 0) {
            if (gj > 0) pbasel += 1;
            if (j < nzones_y - 1) pbaseh += 1;
        }
        for (int i = 0; i < nzones_x; ++i) {
            int gi = i + zone_x_offset;
            vector<int> v(6);
            v[1] = pbasel + 2 * i;
            v[0] = v[1] - 1;
            v[2] = v[1] + 1;
            v[5] = pbaseh + 2 * i;
            v[4] = v[5] + 1;
            v[3] = v[4] + 1;
            if (gj == 0) {
                v[0] = pbasel + i;
                v[2] = v[0] + 1;
                if (gi == global_nzones_x - 1) v.erase(v.begin()+3);
                v.erase(v.begin()+1);
            } // if j
            else if (gj == global_nzones_y - 1) {
                v[5] = pbaseh + i;
                v[3] = v[5] + 1;
                v.erase(v.begin()+4);
                if (gi == 0) v.erase(v.begin()+0);
            } // else if j
            else if (gi == 0)
                v.erase(v.begin()+0);
            else if (gi == global_nzones_x - 1)
                v.erase(v.begin()+3);
            zonestart.push_back(zonepoints.size());
            zonesize.push_back(v.size());
            zonepoints.insert(zonepoints.end(), v.begin(), v.end());
        } // for i
    } // for j
}


void GenerateMesh::generateHaloPointsRect(
        vector<int>& master_colors,
        vector<int>& slaved_points_counts,
        vector<int>& slaved_points,
        vector<int>& slave_colors,
        vector<int>& master_points_counts,
        vector<int>& master_points) const {

    const int np = num_points_x * num_points_y;

    if (num_subregions == 1) return;

    // estimate sizes of slave/master arrays
    slaved_points.reserve((proc_index_y != 0) * num_points_x + (proc_index_x != 0) * num_points_y);
    master_points.reserve((proc_index_y != num_proc_y - 1) * num_points_x +
            (proc_index_x != num_proc_x - 1) * num_points_y + 1);

    // enumerate slave points
    // slave point with master at lower left
    if (proc_index_x != 0 && proc_index_y != 0) {
        int mstrpe = my_color - num_proc_x - 1;
        slaved_points.push_back(0);
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(1);
    }
    // slave points with master below
    if (proc_index_y != 0) {
        int mstrpe = my_color - num_proc_x;
        int oldsize = slaved_points.size();
        int p = 0;
        for (int i = 0; i < num_points_x; ++i) {
            if (i == 0 && proc_index_x != 0) { p++; continue; }
            slaved_points.push_back(p);
            p++;
        }
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(slaved_points.size() - oldsize);
    }
    // slave points with master to left
    if (proc_index_x != 0) {
        int mstrpe = my_color - 1;
        int oldsize = slaved_points.size();
        int p = 0;
        for (int j = 0; j < num_points_y; ++j) {
            if (j == 0 && proc_index_y != 0) { p += num_points_x; continue; }
            slaved_points.push_back(p);
            p += num_points_x;
        }
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(slaved_points.size() - oldsize);
    }

    // enumerate master points
    // master points with slave to right
    if (proc_index_x != num_proc_x - 1) {
        int slvpe = my_color + 1;
        int oldsize = master_points.size();
        int p = num_points_x - 1;
        for (int j = 0; j < num_points_y; ++j) {
            if (j == 0 && proc_index_y != 0) { p += num_points_x; continue; }
            master_points.push_back(p);
            p += num_points_x;
        }
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(master_points.size() - oldsize);
    }
    // master points with slave above
    if (proc_index_y != num_proc_y - 1) {
        int slvpe = my_color + num_proc_x;
        int oldsize = master_points.size();
        int p = (num_points_y - 1) * num_points_x;
        for (int i = 0; i < num_points_x; ++i) {
            if (i == 0 && proc_index_x != 0) { p++; continue; }
            master_points.push_back(p);
            p++;
        }
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(master_points.size() - oldsize);
    }
    // master point with slave at upper right
    if (proc_index_x != num_proc_x - 1 && proc_index_y != num_proc_y - 1) {
        int slvpe = my_color + num_proc_x + 1;
        int p = num_points_x * num_points_y - 1;
        master_points.push_back(p);
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(1);
    }

}


void GenerateMesh::generateHaloPointsPie(
        vector<int>& master_colors,
        vector<int>& slaved_points_counts,
        vector<int>& slaved_points,
        vector<int>& slave_colors,
        vector<int>& master_points_counts,
        vector<int>& master_points) const {

    const int np = (proc_index_y == 0 ? num_points_x * (num_points_y - 1) + 1 : num_points_x * num_points_y);

    if (num_subregions == 1) return;

    // estimate sizes of slave/master arrays
    slaved_points.reserve((proc_index_y != 0) * num_points_x + (proc_index_x != 0) * num_points_y);
    master_points.reserve((proc_index_y != num_proc_y - 1) * num_points_x +
            (proc_index_x != num_proc_x - 1) * num_points_y + 1);

    // enumerate slave points
    // slave point with master at lower left
    if (proc_index_x != 0 && proc_index_y != 0) {
        int mstrpe = my_color - num_proc_x - 1;
        slaved_points.push_back(0);
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(1);
    }
    // slave points with master below
    if (proc_index_y != 0) {
        int mstrpe = my_color - num_proc_x;
        int oldsize = slaved_points.size();
        int p = 0;
        for (int i = 0; i < num_points_x; ++i) {
            if (i == 0 && proc_index_x != 0) { p++; continue; }
            slaved_points.push_back(p);
            p++;
        }
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(slaved_points.size() - oldsize);
    }
    // slave points with master to left
    if (proc_index_x != 0) {
        int mstrpe = my_color - 1;
        int oldsize = slaved_points.size();
        if (proc_index_y == 0) {
            slaved_points.push_back(0);
            // special case:
            // slave point at origin, master not to immediate left
            if (proc_index_x > 1) {
                master_colors.push_back(0);
                slaved_points_counts.push_back(1);
                oldsize += 1;
            }
        }
        int p = (proc_index_y > 0 ? num_points_x : 1);
        for (int j = 1; j < num_points_y; ++j) {
            slaved_points.push_back(p);
            p += num_points_x;
        }
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(slaved_points.size() - oldsize);
    }

    // enumerate master points
    // master points with slave to right
    if (proc_index_x != num_proc_x - 1) {
        int slvpe = my_color + 1;
        int oldsize = master_points.size();
        // special case:  origin as master for slave on PE 1
        if (proc_index_x == 0 && proc_index_y == 0) {
            master_points.push_back(0);
        }
        int p = (proc_index_y > 0 ? 2 * num_points_x - 1 : num_points_x);
        for (int j = 1; j < num_points_y; ++j) {
            master_points.push_back(p);
            p += num_points_x;
        }
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(master_points.size() - oldsize);
        // special case:  origin as master for slaves on PEs > 1
        if (proc_index_x == 0 && proc_index_y == 0) {
            for (int slvpe = 2; slvpe < num_proc_x; ++slvpe) {
                master_points.push_back(0);
                slave_colors.push_back(slvpe);
                master_points_counts.push_back(1);
            }
        }
    }
    // master points with slave above
    if (proc_index_y != num_proc_y - 1) {
        int slvpe = my_color + num_proc_x;
        int oldsize = master_points.size();
        int p = (num_points_y - 1) * num_points_x;
        if (proc_index_y == 0) p -= num_points_x - 1;
        for (int i = 0; i < num_points_x; ++i) {
            if (i == 0 && proc_index_x != 0) { p++; continue; }
            master_points.push_back(p);
            p++;
        }
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(master_points.size() - oldsize);
    }
    // master point with slave at upper right
    if (proc_index_x != num_proc_x - 1 && proc_index_y != num_proc_y - 1) {
        int slvpe = my_color + num_proc_x + 1;
        int p = num_points_x * num_points_y - 1;
        if (proc_index_y == 0) p -= num_points_x - 1;
        master_points.push_back(p);
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(1);
    }

}


void GenerateMesh::generateHaloPointsHex(
        vector<int>& master_colors,
        vector<int>& slaved_points_counts,
        vector<int>& slaved_points,
        vector<int>& slave_colors,
        vector<int>& master_points_counts,
        vector<int>& master_points) const
{

    if (num_subregions == 1) return;

    int np = 0;
    vector<int> pbase(num_points_y);
    for (int j = 0; j < num_points_y; ++j) {
        pbase[j] = np;
        int gj = j + zone_y_offset;
        for (int i = 0; i < num_points_x; ++i) {
            int gi = i + zone_x_offset;
            if (gi == 0 || gi == global_nzones_x || gj == 0 || gj == global_nzones_y)
                np++;
            else if (i == nzones_x && j == 0)
                np++;
            else if (i == 0 && j == nzones_y)
                np++;
            else {
                np += 2;
            }
        } // for i
    } // for j

    // estimate upper bounds for sizes of slave/master arrays
    slaved_points.reserve((proc_index_y != 0) * 2 * num_points_x +
            (proc_index_x != 0) * 2 * num_points_y);
    master_points.reserve((proc_index_y != num_proc_y - 1) * 2 * num_points_x +
            (proc_index_x != num_proc_x - 1) * 2 * num_points_y + 2);

    // enumerate slave points
    // slave points with master at lower left
    if (proc_index_x != 0 && proc_index_y != 0) {
        int mstrpe = my_color - num_proc_x - 1;
        slaved_points.push_back(0);
        slaved_points.push_back(1);
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(2);
    }
    // slave points with master below
    if (proc_index_y != 0) {
        int p = 0;
        int mstrpe = my_color - num_proc_x;
        int oldsize = slaved_points.size();
        for (int i = 0; i < num_points_x; ++i) {
            if (i == 0 && proc_index_x != 0) {
                p += 2;
                continue;
            }
            if (i == 0 || i == nzones_x)
                slaved_points.push_back(p++);
            else {
                slaved_points.push_back(p++);
                slaved_points.push_back(p++);
            }
        }  // for i
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(slaved_points.size() - oldsize);
    }  // if mypey != 0
    // slave points with master to left
    if (proc_index_x != 0) {
        int mstrpe = my_color - 1;
        int oldsize = slaved_points.size();
        for (int j = 0; j < num_points_y; ++j) {
            if (j == 0 && proc_index_y != 0) continue;
            int p = pbase[j];
            if (j == 0 || j == nzones_y)
                slaved_points.push_back(p++);
            else {
                slaved_points.push_back(p++);
                slaved_points.push_back(p++);
           }
        }  // for j
        master_colors.push_back(mstrpe);
        slaved_points_counts.push_back(slaved_points.size() - oldsize);
    }  // if mypex != 0

    // enumerate master points
    // master points with slave to right
    if (proc_index_x != num_proc_x - 1) {
        int slvpe = my_color + 1;
        int oldsize = master_points.size();
        for (int j = 0; j < num_points_y; ++j) {
            if (j == 0 && proc_index_y != 0) continue;
            int p = (j == nzones_y ? np : pbase[j+1]);
            if (j == 0 || j == nzones_y)
                master_points.push_back(p-1);
            else {
                master_points.push_back(p-2);
                master_points.push_back(p-1);
           }
        }
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(master_points.size() - oldsize);
    }  // if mypex != numpex - 1
    // master points with slave above
    if (proc_index_y != num_proc_y - 1) {
        int p = pbase[nzones_y];
        int slvpe = my_color + num_proc_x;
        int oldsize = master_points.size();
        for (int i = 0; i < num_points_x; ++i) {
            if (i == 0 && proc_index_x != 0) {
                p++;
                continue;
            }
            if (i == 0 || i == nzones_x)
                master_points.push_back(p++);
            else {
                master_points.push_back(p++);
                master_points.push_back(p++);
            }
        }  // for i
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(master_points.size() - oldsize);
    }  // if mypey != numpey - 1
    // master points with slave at upper right
    if (proc_index_x != num_proc_x - 1 && proc_index_y != num_proc_y - 1) {
        int slvpe = my_color + num_proc_x + 1;
        master_points.push_back(np-2);
        master_points.push_back(np-1);
        slave_colors.push_back(slvpe);
        master_points_counts.push_back(2);
    }

}


void GenerateMesh::calcPartitions() {

    // pick numpex, numpey such that PE blocks are as close to square
    // as possible
    // we would like:  gnzx / numpex == gnzy / numpey,
    // where numpex * numpey = numpe (total number of PEs available)
    // this solves to:  numpex = sqrt(numpe * gnzx / gnzy)
    // we compute this, assuming gnzx <= gnzy (swap if necessary)
    double nx = static_cast<double>(global_nzones_x);
    double ny = static_cast<double>(global_nzones_y);
    bool swapflag = (nx > ny);
    if (swapflag) swap(nx, ny);
    double n = sqrt(num_subregions * nx / ny);
    // need to constrain n to be an integer with numpe % n == 0
    // try rounding n both up and down
    int n1 = floor(n + 1.e-12);
    n1 = max(n1, 1);
    while (num_subregions % n1 != 0) --n1;
    int n2 = ceil(n - 1.e-12);
    while (num_subregions % n2 != 0) ++n2;
    // pick whichever of n1 and n2 gives blocks closest to square,
    // i.e. gives the shortest long side
    double longside1 = max(nx / n1, ny / (num_subregions/n1));
    double longside2 = max(nx / n2, ny / (num_subregions/n2));
    num_proc_x = (longside1 <= longside2 ? n1 : n2);
    num_proc_y = num_subregions / num_proc_x;
    if (swapflag) swap(num_proc_x, num_proc_y);
}


void GenerateMesh::calcLocalConstants(int color)
{
    my_color = color;
    proc_index_x = my_color % num_proc_x;
    proc_index_y = my_color / num_proc_x;

    zone_x_offset = proc_index_x * global_nzones_x / num_proc_x;
    const int zxstop = (proc_index_x + 1) * global_nzones_x / num_proc_x;
    nzones_x = zxstop - zone_x_offset;
    zone_y_offset = proc_index_y * global_nzones_y / num_proc_y;
    const int zystop = (proc_index_y + 1) * global_nzones_y / num_proc_y;
    nzones_y = zystop - zone_y_offset;

    num_zones = nzones_x * nzones_y;
    num_points_x = nzones_x + 1;
    num_points_y = nzones_y + 1;
}


long long int GenerateMesh::pointLocalToGlobalID(int p) const
{
	long long int globalID = -1;
	if (meshtype == "pie")
		globalID = pointLocalToGlobalIDPie(p);
	else if (meshtype == "rect")
		globalID = pointLocalToGlobalIDRect(p);
	else if (meshtype == "hex")
		globalID = pointLocalToGlobalIDHex(p);
	return globalID;
}


long long int GenerateMesh::pointLocalToGlobalIDPie(int p) const
{
	long long int globalID = -2;
	int px, py;

	if ( (zone_y_offset == 0) && (p == 0) )
	    globalID = 0;
	else {
	    if (zone_y_offset == 0) {
			py = (p - 1) / num_points_x + 1;
			px = p - (py - 1) * num_points_x - 1;
	    } else {
	        py = p / num_points_x;
	        px = p - py * num_points_x;
	    }
	    globalID = (global_nzones_x + 1) * (py + zone_y_offset - 1) + 1
	            + px + zone_x_offset;
	}
	return globalID;
}


long long int GenerateMesh::pointLocalToGlobalIDRect(int p) const
{
	const int py = p / num_points_x;
	const int px = p - py * num_points_x;

	long long int globalID = (global_nzones_x + 1) * (py + zone_y_offset)
			+ px + zone_x_offset;
	return globalID;
}


long long int GenerateMesh::pointLocalToGlobalIDHex(int p) const
{
	const int zone_y_start = yStart(proc_index_y);
	const int zone_y_stop = yStart(proc_index_y + 1);
	const int zone_x_start = xStart(proc_index_x);
	const int zone_x_stop = xStart(proc_index_x + 1);

	long long int globalID = -3;
	int i,j;

	int first_row_npts = 2 * num_points_x;
	int mid_rows_npts = 2 * num_points_x;

	if (zone_y_start == 0)
		first_row_npts = num_points_x;
	else {
		if (zone_x_start == 0)
			first_row_npts -= 1;
        // lower right corner
        first_row_npts -= 1;
	}
	if (zone_x_start == 0)
		mid_rows_npts -= 1;
	if (zone_x_stop == global_nzones_x)
		mid_rows_npts -= 1;

	if (p < first_row_npts) {
		j = 0;
		i = p;
	} else {
		j =  (p - first_row_npts) / mid_rows_npts + 1;
		i = p - first_row_npts - (j - 1) * mid_rows_npts;
	}

	int gj = j + zone_y_offset;

	if (gj == 0)
		globalID = 0;
	else
		globalID = numPointsPreviousRowsNonZeroJHex(gj);

	if ( (gj == 0) || (gj == global_nzones_y))
		globalID += zone_x_offset;
	else if (zone_x_offset != 0)
		globalID += 2 * zone_x_offset - 1;
	globalID += i;

	// upper left corner skips a point
	if ( (gj == zone_y_stop) &&
			(zone_x_start != 0) && (gj != 0) && (gj != global_nzones_y) )
		globalID++;
	return globalID;
}
