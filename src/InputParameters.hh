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
    int chunk_size_;                // max size for processing chunks
    std::vector<double> subregion_; // bounding box for a subregion
                                    // if nonempty, should have 4 entries:
                                    // xmin, xmax, ymin, ymax
    bool write_xy_file_;            // flag:  write .xy file?
    bool write_gold_file_;          // flag:  write Ensight file?
    std::string meshtype_;          // generated mesh type
    int nzones_x_, nzones_y_;       // global number of zones, in x and y
                                    // directions
    double len_x_, len_y_;          // length of mesh sides, in x and y
                                    // directions
    double cfl_;                    // Courant number, limits timestep
    double cflv_;                   // volume change limit for timestep
    double rho_init_;               // initial density for main mesh
    double energy_init_;            // initial energy for main mesh
    double rho_init_sub_;           // initial density in subregion
    double energy_init_sub_;        // initial energy in subregion
    double vel_init_radial_;        // initial velocity in radial direction
    std::vector<double> bcx_;       // x values of x-plane fixed boundaries
    std::vector<double> bcy_;       // y values of y-plane fixed boundaries
    double gamma_;                  // coeff. for ideal gas equation
    double ssmin_;                  // minimum sound speed for gas
    double alfa_;                   // alpha coefficient for TTS model
    double qgamma_;                 // gamma coefficient for Q model
    double q1_, q2_;                // linear and quadratic coefficients
                                    // for Q model

    std::string probname;
};

#endif /* INPUTPARAMETERS_HH_ */
