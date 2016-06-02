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
#include "Vec2.hh"

// forward declarations
class InputFile;


class GenMesh {
public:

    std::string meshtype_;       // generated mesh type
    int global_nzones_x_, global_nzones_y_;             // global number of zones, in x and y
                                // directions
    double len_x_, len_y_;          // length of mesh sides, in x and y
                                // directions
    int num_proc_x_, num_proc_y_;         // number of PEs to use, in x and y
                                // directions
    int proc_index_x_, proc_index_y_;           // my PE index, in x and y directions
    int nzones_x_, nzones_y_;               // (local) number of zones, in x and y
                                // directions
    int zone_x_offset_, zone_y_offset_;     // offsets of local zone array into
                                // global, in x and y directions

    GenMesh(const InputFile* inp);
    ~GenMesh();

    void generate(
            std::vector<double2>& pointpos,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& slavemstrpes,
            std::vector<int>& slavemstrcounts,
            std::vector<int>& slavepoints,
            std::vector<int>& masterslvpes,
            std::vector<int>& masterslvcounts,
            std::vector<int>& masterpoints);

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
            std::vector<int>& masterpoints);

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
            std::vector<int>& masterpoints);

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
            std::vector<int>& masterpoints);

    void calcPartitions();

}; // class GenMesh


#endif /* GENMESH_HH_ */
