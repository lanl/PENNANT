/*
 * Hydro.hh
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Triad National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDRO_HH_
#define HYDRO_HH_

#include <string>
#include <vector>

#include "Vec2.hh"

// forward declarations
class InputFile;
class Mesh;
class PolyGas;
class TTS;
class QCS;
class HydroBC;


class Hydro {
public:

    // associated mesh object
    Mesh* mesh;

    // children of this object
    PolyGas* pgas;
    TTS* tts;
    QCS* qcs;
    std::vector<HydroBC*> bcs;

    double cfl;                 // Courant number, limits timestep
    double cflv;                // volume change limit for timestep
    double rinit;               // initial density for main mesh
    double einit;               // initial energy for main mesh
    double rinitsub;            // initial density in subregion
    double einitsub;            // initial energy in subregion
    double uinitradial;         // initial velocity in radial direction
    std::vector<double> bcx;    // x values of x-plane fixed boundaries
    std::vector<double> bcy;    // y values of y-plane fixed boundaries

    double dtrec;               // maximum timestep for hydro
    std::string msgdtrec;       // message:  reason for dtrec

    double2* pu;       // point velocity
    double2* pu0;      // point velocity, start of cycle
    double2* pap;      // point acceleration
    double2* pf;       // point force
    double* pmaswt;    // point mass, weighted by 1/r
    double* cmaswt;    // side contribution to pmaswt // TODO: original name was smaswt. Check comment; check usage in rest of code
    double2* cftot;    

    double* zm;        // zone mass
    double* zr;        // zone density
    double* zrp;       // zone density, middle of cycle
    double* ze;        // zone specific internal energy
                       // (energy per unit mass)
    double* zetot;     // zone total internal energy
    double* zetot0;    // zetot at start of cycle
    double* zw;        // zone work done in cycle
    double* zwrate;    // zone work rate
    double* zp;        // zone pressure
    double* zss;       // zone sound speed
    double* zdu;       // zone velocity difference

    double2* sf;       // side force (from pressure)
    double2* sfq;      // side force from artificial visc.
    double2* sft;      // side force from tts

#ifdef USE_MPI
    // mpi comm variables
    int nummstrpe;     // number of messages mype sends to master pes
    int numslvpe;      // number of messages mype receives from slave pes
    int numprx;        // number of proxies on mype
    int numslv;        // number of slaves on mype
    int* mapslvpepe;   // map: slave pe -> (global) pe
    int* mapslvpeprx1; // map: slave pe -> first proxy in proxy buffer
    int* mapprxp;      // map: proxy -> corresponding (master) point
    int* slvpenumprx;  // number of proxies for each slave pe
    int* mapmstrpepe;  // map: master pe -> (global) pe
    int* mstrpenumslv; // number of slaves for each master pe
    int* mapmstrpeslv1;// map: master pe -> first slave in slave buffer
    int* mapslvp;     
#endif

    Hydro(const InputFile* inp, Mesh* m);
    ~Hydro();

    void init();

    void getData();

    void initRadialVel(const double vel);

    void doCycle(const double dt);

    void getDtHydro(
            double& dtnew,
            std::string& msgdtnew);

   void sumEnergy(
        const double* zetot,
        const double* zarea,
        const double* zvol,
        const double* zm,
        const double* smf,
        const double2* px,
        const double2* pu,
        double& ei,
        double& ek,
        const int zfirst,
        const int zlast,
        const int sfirst,
        const int slast);

   void writeEnergyCheck();
}; // class Hydro



#endif /* HYDRO_HH_ */
