/*
 * Hydro.hh
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDRO_HH_
#define HYDRO_HH_

#include <string>
#include <vector>

#include "InputParameters.hh"
#include "LogicalStructured.hh"
#include "Parallel.hh"
#include "Vec2.hh"

// forward declarations
class InputFile;
class LocalMesh;
class PolyGas;
class TTS;
class QCS;
class HydroBC;

// TODO making all member variables public is not encapsulation
class Hydro {
public:

    Hydro(const InputParameters& params, LocalMesh* m,
    		DynamicCollective add_reduction,
        Context ctx, HighLevelRuntime* rt);
    ~Hydro();

    // associated mesh object
    LocalMesh* mesh;

    // children of this object
    PolyGas* pgas;
    TTS* tts;
    QCS* qcs;
    std::vector<HydroBC*> bcs;

    double cfl;                 // Courant number, limits timestep
    double cflv;                // volume change limit for timestep
    double rho_init;               // initial density for main mesh
    double energy_init;               // initial energy for main mesh
    double rho_init_sub;            // initial density in subregion
    double energy_init_sub;            // initial energy in subregion
    double vel_init_radial;         // initial velocity in radial direction
    std::vector<double> bcx;    // x values of x-plane fixed boundaries
    std::vector<double> bcy;    // y values of y-plane fixed boundaries

    double2* pt_vel;       // point velocity
    double2* pt_vel0;      // point velocity, start of cycle
    double2* pt_accel;      // point acceleration
    double* crnr_weighted_mass;    // corner contribution to pmaswt

    double* zone_mass;        // zone mass
    double* zone_energy_tot;     // zone total internal energy
    double* zone_work;        // zone work done in cycle
    double* zone_work_rate;    // zone work rate
    double* zone_sound_speed;       // zone sound speed
    double* zone_dvel;       // zone velocity difference

    double2* side_force_pres;      // side force from pressure
    double2* side_force_visc;      // side force from artificial visc.
    double2* side_force_tts;      // side force from tts
    double2* crnr_force_tot;    // corner force, total from all sources

    void init();

    void initRadialVel(
            const double vel,
            const int pfirst,
            const int plast);

    TimeStep doCycle(const double dt);

    void advPosHalf(
            const double dt,
            const int pfirst,
            const int plast);

    void advPosFull(
            const double dt,
            const int pfirst,
            const int plast);

    void calcCrnrMass(
            const int sfirst,
            const int slast);

    void sumCrnrForce(
            const int sfirst,
            const int slast);

    void calcAccel(
            const int pfirst,
            const int plast);

    void calcRho(
            const double* zvol,
            double* zr,
            const int zfirst,
            const int zlast);

    void calcWork(
            const double dt,
            const int sfirst,
            const int slast);

    void calcWorkRate(
            const double dt,
            const int zfirst,
            const int zlast);

    void calcEnergy(
            const int zfirst,
            const int zlast);

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

    void copyZonesToLegion(
            DoubleAccessor* zone_rho,
            DoubleAccessor*  zone_energy_density,
            DoubleAccessor*  zone_pressure,
            IndexSpace ispace_zones);

	double* zone_rho;             // zone density // TODO make private
    double* zone_rho_pred;        // zone density, middle of cycle
	double* zone_energy_density;  // zone specific internal energy
    // (energy per unit mass)  // TODO make private
	double* zone_pressure_;        // zone pressure  // TODO make private
private:
	void  allocateFields();

    DynamicCollective add_reduction;
    Context ctx;
    HighLevelRuntime* runtime;
    LogicalStructured zones;
    LogicalStructured sides_and_corners;
    LogicalStructured edges;
    LogicalStructured points;
    const int mype;

}; // class Hydro



#endif /* HYDRO_HH_ */
