/*
 * GenMesh.hh
 *
 *  Created on: Jun 4, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef GENMESH_HH_
#define GENMESH_HH_


#include <string>
#include <vector>


#include "InputParameters.hh"
#include "Vec2.hh"


class GenerateMesh {
public:
    GenerateMesh(const InputParameters& params);

    void generate(
            std::vector<double2>& pointpos,
            std::vector<int>& zonepoints_ptr_CRS,
            std::vector<int>& zonepoints,
            std::vector<int>& slavemstrpes,
            std::vector<int>& slavemstrcounts,
            std::vector<int>& slavepoints,
            std::vector<int>& masterslvpes,
            std::vector<int>& masterslvcounts,
            std::vector<int>& masterpoints) const;

    long long int pointLocalToGlobalID(int p) const;

protected:
    std::string meshtype;                   // generated mesh type
    int global_nzones_x, global_nzones_y;   // global number of zones, in x and y
                                            // directions
    double len_x, len_y;                    // length of mesh sides, in x and y
                                            // directions
    int num_proc_x, num_proc_y;             // number of PEs to use, in x and y
                                            // directions
    int proc_index_x, proc_index_y;         // my PE index, in x and y directions
    int nzones_x, nzones_y;                 // (local) number of zones, in x and y
                                            // directions
    int zone_x_offset, zone_y_offset;       // offsets of local zone array into
                                            // global, in x and y directions
    // local grid info
    int num_zones;
    int num_points_x;
    int num_points_y;

    const int num_subregions;
    const int my_color;

    void calcLocalConstants(int color);

    void generateRect(
            std::vector<double2>& pointpos,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& slavemstrpes,
            std::vector<int>& slavemstrcounts,
            std::vector<int>& slavepoints,
            std::vector<int>& masterslvpes,
            std::vector<int>& masterslvcounts,
            std::vector<int>& masterpoints) const;

    void generatePie(
            std::vector<double2>& pointpos,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& slavemstrpes,
            std::vector<int>& slavemstrcounts,
            std::vector<int>& slavepoints,
            std::vector<int>& masterslvpes,
            std::vector<int>& masterslvcounts,
            std::vector<int>& masterpoints) const;

    void generateHex(
            std::vector<double2>& pointpos,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& slavemstrpes,
            std::vector<int>& slavemstrcounts,
            std::vector<int>& slavepoints,
            std::vector<int>& masterslvpes,
            std::vector<int>& masterslvcounts,
            std::vector<int>& masterpoints) const;

    long long int pointLocalToGlobalIDPie(int p) const;

    long long int pointLocalToGlobalIDRect(int p) const;

    long long int pointLocalToGlobalIDHex(int p) const;

    void calcPartitions();

    inline int yStart(int proc_index_y) const
    { return proc_index_y * global_nzones_y / num_proc_y; }

    inline int xStart(int proc_index_x) const
    { return proc_index_x * global_nzones_x / num_proc_x; }

    inline int numPointsPreviousRowsNonZeroJHex(int j) const
    { return (2 * j - 1) * global_nzones_x + 1; }

}; // class GenerateMesh


#endif /* GENMESH_HH_ */
