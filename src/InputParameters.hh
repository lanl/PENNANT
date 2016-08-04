/*
 * InputParameters.hh
 *
 *  Created on: Jun 9, 2016
 *      Author: jgraham
 */

#ifndef INPUTPARAMETERS_HH_
#define INPUTPARAMETERS_HH_

#include <string>
#include <vector>

struct InputParameters {
	int ntasks_;
	int task_id_;
    double tstop_;                  // simulation stop time
    int cstop_;                     // simulation stop cycle
    double dtmax_;                  // maximum timestep size
    double dtinit_;                 // initial timestep size
    double dtfac_;                  // factor limiting timestep growth
    int dtreport_;                  // frequency for timestep reports
    int chunk_size_;                 // max size for processing chunks
    std::vector<double> subregion_; // bounding box for a subregion
                                   // if nonempty, should have 4 entries:
                                   // xmin, xmax, ymin, ymax
    bool write_xy_file_;                  // flag:  write .xy file?
    bool write_gold_file_;                // flag:  write Ensight file?
    std::string meshtype_;       // generated mesh type
    int nzones_x_, nzones_y_;             // global number of zones, in x and y
                                // directions
    double len_x_, len_y_;          // length of mesh sides, in x and y
                                // directions
    std::string probname;
};

#endif /* INPUTPARAMETERS_HH_ */
