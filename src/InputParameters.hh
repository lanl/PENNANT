/*
 * InputParameters.hh
 *
 *  Created on: Jun 9, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#ifndef INPUTPARAMETERS_HH_
#define INPUTPARAMETERS_HH_

#include <string>
#include <vector>

struct DirectInputParameters {
	int ntasks;
	int task_id;
    double tstop;                  // simulation stop time
    int cstop;                     // simulation stop cycle
    double dtmax;                  // maximum timestep size
    double dtinit;                 // initial timestep size
    double dtfac;                  // factor limiting timestep growth
    int dtreport;                  // frequency for timestep reports
    int chunk_size;                // max size for processing chunks
    bool write_xy_file;            // flag:  write .xy file?
    bool write_gold_file;          // flag:  write Ensight file?
    int nzones_x, nzones_y;       // global number of zones, in x and y
                                    // directions
    double len_x, len_y;          // length of mesh sides, in x and y
                                    // directions
    double cfl;                    // Courant number, limits timestep
    double cflv;                   // volume change limit for timestep
    double rho_init;               // initial density for main mesh
    double energy_init;            // initial energy for main mesh
    double rho_init_sub;           // initial density in subregion
    double energy_init_sub;        // initial energy in subregion
    double vel_init_radial;        // initial velocity in radial direction
    double gamma;                  // coeff. for ideal gas equation
    double ssmin;                  // minimum sound speed for gas
    double alpha;                   // alpha coefficient for TTS model
    double qgamma;                 // gamma coefficient for Q model
    double q1, q2;                // linear and quadratic coefficients
                                    // for Q model
    double subregion_xmin; 		   // bounding box for a subregion
    double subregion_xmax; 		   // if xmin != std::numeric_limits<double>::max(),
    double subregion_ymin;         // should have 4 entries:
    double subregion_ymax; 		   // xmin, xmax, ymin, ymax
};

struct InputParameters {
	DirectInputParameters directs; // for serialization
    std::string meshtype;          // generated mesh type
    std::vector<double> bcx;       // x values of x-plane fixed boundaries
    std::vector<double> bcy;       // y values of y-plane fixed boundaries
    std::string probname;
};

#endif /* INPUTPARAMETERS_HH_ */
