/*
 * TTS.hh
 *
 *  Created on: Feb 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef TTS_HH_
#define TTS_HH_

#include "InputParameters.hh"
#include "Parallel.hh"
#include "Vec2.hh"

// forward declarations
class InputFile;
class Hydro;


class TTS {
public:

    // parent hydro object
    Hydro* hydro;

    double alfa;                   // alpha coefficient for TTS model
    double ssmin;                  // minimum sound speed

    TTS(const InputParameters& params, Hydro* h);
    ~TTS();

void calcForce(
        const double* zarea,
        const DoubleAccessor& zr,
        const double* zss,
        const double* sarea,
        const double* smf,
        const double2* ssurfp,
        double2* sf,
        const int sfirst,
        const int slast);

}; // class TTS


#endif /* TTS_HH_ */
