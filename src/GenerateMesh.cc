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

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "Vec2.hh"
#include "Parallel.hh"

using namespace std;


GenerateMesh::GenerateMesh(const InputParameters& input_params) :
	meshtype_(input_params.meshtype_),
	global_nzones_x_(input_params.directs_.nzones_x_),
	global_nzones_y_(input_params.directs_.nzones_y_),
	len_x_(input_params.directs_.len_x_),
	len_y_(input_params.directs_.len_y_)
{
    calcPartitions();
}


GenerateMesh::~GenerateMesh() {}


void GenerateMesh::generate(
        std::vector<int>& slavemstrpes,
        std::vector<int>& slavemstrcounts,
        std::vector<int>& slavepoints,
        std::vector<int>& masterslvpes,
        std::vector<int>& masterslvcounts,
        std::vector<int>& masterpoints){

    std::vector<double2> pointpos;
    std::vector<int> zonestart;
    std::vector<int> zonepoints;

	// do calculations common to all mesh types
    zone_x_offset_ = proc_index_x_ * global_nzones_x_ / num_proc_x_;
    const int zxstop = (proc_index_x_ + 1) * global_nzones_x_ / num_proc_x_;
    nzones_x_ = zxstop - zone_x_offset_;
    zone_y_offset_ = proc_index_y_ * global_nzones_y_ / num_proc_y_;
    const int zystop = (proc_index_y_ + 1) * global_nzones_y_ / num_proc_y_;
    nzones_y_ = zystop - zone_y_offset_;

    // mesh type-specific calculations
    std::vector<int> zonesize;
    if (meshtype_ == "pie")
        generatePie(pointpos, zonestart, zonesize, zonepoints,
                slavemstrpes, slavemstrcounts, slavepoints,
                masterslvpes, masterslvcounts, masterpoints);
    else if (meshtype_ == "rect")
        generateRect(pointpos, zonestart, zonesize, zonepoints,
                slavemstrpes, slavemstrcounts, slavepoints,
                masterslvpes, masterslvcounts, masterpoints);
    else if (meshtype_ == "hex")
        generateHex(pointpos, zonestart, zonesize, zonepoints,
                slavemstrpes, slavemstrcounts, slavepoints,
                masterslvpes, masterslvcounts, masterpoints);

    zonestart.push_back(zonestart.back()+zonesize.back());  // compressed row storage

    //assert(point_position.size() == numberOfPoints());
	//assert((zone_points_ptr.size() - 1) == numberOfZones());
	//assert(zone_points_val.size() == numberOfSides());
}


void GenerateMesh::generateRect(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& slavemstrpes,
        std::vector<int>& slavemstrcounts,
        std::vector<int>& slavepoints,
        std::vector<int>& masterslvpes,
        std::vector<int>& masterslvcounts,
        std::vector<int>& masterpoints) {

    const int nz = nzones_x_ * nzones_y_;
    const int npx = nzones_x_ + 1;
    const int npy = nzones_y_ + 1;
    const int np = npx * npy;

    // generate point coordinates
    pointpos.reserve(np);
    double dx = len_x_ / (double) global_nzones_x_;
    double dy = len_y_ / (double) global_nzones_y_;
    for (int j = 0; j < npy; ++j) {
        double y = dy * (double) (j + zone_y_offset_);
        for (int i = 0; i < npx; ++i) {
            double x = dx * (double) (i + zone_x_offset_);
            pointpos.push_back(make_double2(x, y));
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(4 * nz);
    for (int j = 0; j < nzones_y_; ++j) {
        for (int i = 0; i < nzones_x_; ++i) {
            zonestart.push_back(zonepoints.size());
            zonesize.push_back(4);
            int p0 = j * npx + i;
            zonepoints.push_back(p0);
            zonepoints.push_back(p0 + 1);
            zonepoints.push_back(p0 + npx + 1);
            zonepoints.push_back(p0 + npx);
       }
    }

    if (Parallel::num_subregions() == 1) return;

    // estimate sizes of slave/master arrays
    slavepoints.reserve((proc_index_y_ != 0) * npx + (proc_index_x_ != 0) * npy);
    masterpoints.reserve((proc_index_y_ != num_proc_y_ - 1) * npx +
            (proc_index_x_ != num_proc_x_ - 1) * npy + 1);

    // enumerate slave points
    // slave point with master at lower left
    if (proc_index_x_ != 0 && proc_index_y_ != 0) {
        int mstrpe = Parallel::mype() - num_proc_x_ - 1;
        slavepoints.push_back(0);
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(1);
    }
    // slave points with master below
    if (proc_index_y_ != 0) {
        int mstrpe = Parallel::mype() - num_proc_x_;
        int oldsize = slavepoints.size();
        int p = 0;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && proc_index_x_ != 0) { p++; continue; }
            slavepoints.push_back(p);
            p++;
        }
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(slavepoints.size() - oldsize);
    }
    // slave points with master to left
    if (proc_index_x_ != 0) {
        int mstrpe = Parallel::mype() - 1;
        int oldsize = slavepoints.size();
        int p = 0;
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && proc_index_y_ != 0) { p += npx; continue; }
            slavepoints.push_back(p);
            p += npx;
        }
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(slavepoints.size() - oldsize);
    }

    // enumerate master points
    // master points with slave to right
    if (proc_index_x_ != num_proc_x_ - 1) {
        int slvpe = Parallel::mype() + 1;
        int oldsize = masterpoints.size();
        int p = npx - 1;
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && proc_index_y_ != 0) { p += npx; continue; }
            masterpoints.push_back(p);
            p += npx;
        }
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(masterpoints.size() - oldsize);
    }
    // master points with slave above
    if (proc_index_y_ != num_proc_y_ - 1) {
        int slvpe = Parallel::mype() + num_proc_x_;
        int oldsize = masterpoints.size();
        int p = (npy - 1) * npx;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && proc_index_x_ != 0) { p++; continue; }
            masterpoints.push_back(p);
            p++;
        }
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(masterpoints.size() - oldsize);
    }
    // master point with slave at upper right
    if (proc_index_x_ != num_proc_x_ - 1 && proc_index_y_ != num_proc_y_ - 1) {
        int slvpe = Parallel::mype() + num_proc_x_ + 1;
        int p = npx * npy - 1;
        masterpoints.push_back(p);
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(1);
    }

}


void GenerateMesh::generatePie(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& slavemstrpes,
        std::vector<int>& slavemstrcounts,
        std::vector<int>& slavepoints,
        std::vector<int>& masterslvpes,
        std::vector<int>& masterslvcounts,
        std::vector<int>& masterpoints) {

    const int nz = nzones_x_ * nzones_y_;
    const int npx = nzones_x_ + 1;
    const int npy = nzones_y_ + 1;
    const int np = (proc_index_y_ == 0 ? npx * (npy - 1) + 1 : npx * npy);

    // generate point coordinates
    pointpos.reserve(np);
    double dth = len_x_ / (double) global_nzones_x_;
    double dr  = len_y_ / (double) global_nzones_y_;
    for (int j = 0; j < npy; ++j) {
        if (j + zone_y_offset_ == 0) {
            pointpos.push_back(make_double2(0., 0.));
            continue;
        }
        double r = dr * (double) (j + zone_y_offset_);
        for (int i = 0; i < npx; ++i) {
            double th = dth * (double) (global_nzones_x_ - (i + zone_x_offset_));
            double x = r * cos(th);
            double y = r * sin(th);
            pointpos.push_back(make_double2(x, y));
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(4 * nz);
    for (int j = 0; j < nzones_y_; ++j) {
        for (int i = 0; i < nzones_x_; ++i) {
            zonestart.push_back(zonepoints.size());
            int p0 = j * npx + i;
            if (proc_index_y_ == 0) p0 -= npx - 1;
            if (j + zone_y_offset_ == 0) {
                zonesize.push_back(3);
                zonepoints.push_back(0);
            }
            else {
                zonesize.push_back(4);
                zonepoints.push_back(p0);
                zonepoints.push_back(p0 + 1);
            }
            zonepoints.push_back(p0 + npx + 1);
            zonepoints.push_back(p0 + npx);
        }
    }

    if (Parallel::num_subregions() == 1) return;

    // estimate sizes of slave/master arrays
    slavepoints.reserve((proc_index_y_ != 0) * npx + (proc_index_x_ != 0) * npy);
    masterpoints.reserve((proc_index_y_ != num_proc_y_ - 1) * npx +
            (proc_index_x_ != num_proc_x_ - 1) * npy + 1);

    // enumerate slave points
    // slave point with master at lower left
    if (proc_index_x_ != 0 && proc_index_y_ != 0) {
        int mstrpe = Parallel::mype() - num_proc_x_ - 1;
        slavepoints.push_back(0);
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(1);
    }
    // slave points with master below
    if (proc_index_y_ != 0) {
        int mstrpe = Parallel::mype() - num_proc_x_;
        int oldsize = slavepoints.size();
        int p = 0;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && proc_index_x_ != 0) { p++; continue; }
            slavepoints.push_back(p);
            p++;
        }
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(slavepoints.size() - oldsize);
    }
    // slave points with master to left
    if (proc_index_x_ != 0) {
        int mstrpe = Parallel::mype() - 1;
        int oldsize = slavepoints.size();
        if (proc_index_y_ == 0) {
            slavepoints.push_back(0);
            // special case:
            // slave point at origin, master not to immediate left
            if (proc_index_x_ > 1) {
                slavemstrpes.push_back(0);
                slavemstrcounts.push_back(1);
                oldsize += 1;
            }
        }
        int p = (proc_index_y_ > 0 ? npx : 1);
        for (int j = 1; j < npy; ++j) {
            slavepoints.push_back(p);
            p += npx;
        }
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(slavepoints.size() - oldsize);
    }

    // enumerate master points
    // master points with slave to right
    if (proc_index_x_ != num_proc_x_ - 1) {
        int slvpe = Parallel::mype() + 1;
        int oldsize = masterpoints.size();
        // special case:  origin as master for slave on PE 1
        if (proc_index_x_ == 0 && proc_index_y_ == 0) {
            masterpoints.push_back(0);
        }
        int p = (proc_index_y_ > 0 ? 2 * npx - 1 : npx);
        for (int j = 1; j < npy; ++j) {
            masterpoints.push_back(p);
            p += npx;
        }
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(masterpoints.size() - oldsize);
        // special case:  origin as master for slaves on PEs > 1
        if (proc_index_x_ == 0 && proc_index_y_ == 0) {
            for (int slvpe = 2; slvpe < num_proc_x_; ++slvpe) {
                masterpoints.push_back(0);
                masterslvpes.push_back(slvpe);
                masterslvcounts.push_back(1);
            }
        }
    }
    // master points with slave above
    if (proc_index_y_ != num_proc_y_ - 1) {
        int slvpe = Parallel::mype() + num_proc_x_;
        int oldsize = masterpoints.size();
        int p = (npy - 1) * npx;
        if (proc_index_y_ == 0) p -= npx - 1;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && proc_index_x_ != 0) { p++; continue; }
            masterpoints.push_back(p);
            p++;
        }
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(masterpoints.size() - oldsize);
    }
    // master point with slave at upper right
    if (proc_index_x_ != num_proc_x_ - 1 && proc_index_y_ != num_proc_y_ - 1) {
        int slvpe = Parallel::mype() + num_proc_x_ + 1;
        int p = npx * npy - 1;
        if (proc_index_y_ == 0) p -= npx - 1;
        masterpoints.push_back(p);
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(1);
    }

}


void GenerateMesh::generateHex(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& slavemstrpes,
        std::vector<int>& slavemstrcounts,
        std::vector<int>& slavepoints,
        std::vector<int>& masterslvpes,
        std::vector<int>& masterslvcounts,
        std::vector<int>& masterpoints) {

    const int nz = nzones_x_ * nzones_y_;
    const int npx = nzones_x_ + 1;
    const int npy = nzones_y_ + 1;

    // generate point coordinates
    pointpos.reserve(2 * npx * npy);  // upper bound
    double dx = len_x_ / (double) (global_nzones_x_ - 1);
    double dy = len_y_ / (double) (global_nzones_y_ - 1);

    vector<int> pbase(npy);
    for (int j = 0; j < npy; ++j) {
        pbase[j] = pointpos.size();
        int gj = j + zone_y_offset_;
        double y = dy * ((double) gj - 0.5);
        y = max(0., min(len_y_, y));
        for (int i = 0; i < npx; ++i) {
            int gi = i + zone_x_offset_;
            double x = dx * ((double) gi - 0.5);
            x = max(0., min(len_x_, x));
            if (gi == 0 || gi == global_nzones_x_ || gj == 0 || gj == global_nzones_y_)
                pointpos.push_back(make_double2(x, y));
            else if (i == nzones_x_ && j == 0)
                pointpos.push_back(
                        make_double2(x - dx / 6., y + dy / 6.));
            else if (i == 0 && j == nzones_y_)
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
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(6 * nz);  // upper bound
    for (int j = 0; j < nzones_y_; ++j) {
        int gj = j + zone_y_offset_;
        int pbasel = pbase[j];
        int pbaseh = pbase[j+1];
        if (proc_index_x_ > 0) {
            if (gj > 0) pbasel += 1;
            if (j < nzones_y_ - 1) pbaseh += 1;
        }
        for (int i = 0; i < nzones_x_; ++i) {
            int gi = i + zone_x_offset_;
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
                if (gi == global_nzones_x_ - 1) v.erase(v.begin()+3);
                v.erase(v.begin()+1);
            } // if j
            else if (gj == global_nzones_y_ - 1) {
                v[5] = pbaseh + i;
                v[3] = v[5] + 1;
                v.erase(v.begin()+4);
                if (gi == 0) v.erase(v.begin()+0);
            } // else if j
            else if (gi == 0)
                v.erase(v.begin()+0);
            else if (gi == global_nzones_x_ - 1)
                v.erase(v.begin()+3);
            zonestart.push_back(zonepoints.size());
            zonesize.push_back(v.size());
            zonepoints.insert(zonepoints.end(), v.begin(), v.end());
        } // for i
    } // for j

    if (Parallel::num_subregions() == 1) return;

    // estimate upper bounds for sizes of slave/master arrays
    slavepoints.reserve((proc_index_y_ != 0) * 2 * npx +
            (proc_index_x_ != 0) * 2 * npy);
    masterpoints.reserve((proc_index_y_ != num_proc_y_ - 1) * 2 * npx +
            (proc_index_x_ != num_proc_x_ - 1) * 2 * npy + 2);

    // enumerate slave points
    // slave points with master at lower left
    if (proc_index_x_ != 0 && proc_index_y_ != 0) {
        int mstrpe = Parallel::mype() - num_proc_x_ - 1;
        slavepoints.push_back(0);
        slavepoints.push_back(1);
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(2);
    }
    // slave points with master below
    if (proc_index_y_ != 0) {
        int p = 0;
        int mstrpe = Parallel::mype() - num_proc_x_;
        int oldsize = slavepoints.size();
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && proc_index_x_ != 0) {
                p += 2;
                continue;
            }
            if (i == 0 || i == nzones_x_)
                slavepoints.push_back(p++);
            else {
                slavepoints.push_back(p++);
                slavepoints.push_back(p++);
            }
        }  // for i
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(slavepoints.size() - oldsize);
    }  // if mypey != 0
    // slave points with master to left
    if (proc_index_x_ != 0) {
        int mstrpe = Parallel::mype() - 1;
        int oldsize = slavepoints.size();
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && proc_index_y_ != 0) continue;
            int p = pbase[j];
            if (j == 0 || j == nzones_y_)
                slavepoints.push_back(p++);
            else {
                slavepoints.push_back(p++);
                slavepoints.push_back(p++);
           }
        }  // for j
        slavemstrpes.push_back(mstrpe);
        slavemstrcounts.push_back(slavepoints.size() - oldsize);
    }  // if mypex != 0

    // enumerate master points
    // master points with slave to right
    if (proc_index_x_ != num_proc_x_ - 1) {
        int slvpe = Parallel::mype() + 1;
        int oldsize = masterpoints.size();
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && proc_index_y_ != 0) continue;
            int p = (j == nzones_y_ ? np : pbase[j+1]);
            if (j == 0 || j == nzones_y_)
                masterpoints.push_back(p-1);
            else {
                masterpoints.push_back(p-2);
                masterpoints.push_back(p-1);
           }
        }
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(masterpoints.size() - oldsize);
    }  // if mypex != numpex - 1
    // master points with slave above
    if (proc_index_y_ != num_proc_y_ - 1) {
        int p = pbase[nzones_y_];
        int slvpe = Parallel::mype() + num_proc_x_;
        int oldsize = masterpoints.size();
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && proc_index_x_ != 0) {
                p++;
                continue;
            }
            if (i == 0 || i == nzones_x_)
                masterpoints.push_back(p++);
            else {
                masterpoints.push_back(p++);
                masterpoints.push_back(p++);
            }
        }  // for i
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(masterpoints.size() - oldsize);
    }  // if mypey != numpey - 1
    // master points with slave at upper right
    if (proc_index_x_ != num_proc_x_ - 1 && proc_index_y_ != num_proc_y_ - 1) {
        int slvpe = Parallel::mype() + num_proc_x_ + 1;
        masterpoints.push_back(np-2);
        masterpoints.push_back(np-1);
        masterslvpes.push_back(slvpe);
        masterslvcounts.push_back(2);
    }

}


void GenerateMesh::calcPartitions() {

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
    proc_index_x_ = Parallel::mype() % num_proc_x_;
    proc_index_y_ = Parallel::mype() / num_proc_x_;

}

