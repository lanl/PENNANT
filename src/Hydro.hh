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
#include "Parallel.hh"
#include "Vec2.hh"

// forward declarations
class InputFile;
class Mesh;
class PolyGas;
class TTS;
class QCS;
class HydroBC;

// TODO making all member variables public is not encapsulation
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
    double rho_init;               // initial density for main mesh
    double energy_init;               // initial energy for main mesh
    double rho_init_sub;            // initial density in subregion
    double energy_init_sub;            // initial energy in subregion
    double vel_init_radial;         // initial velocity in radial direction
    std::vector<double> bcx;    // x values of x-plane fixed boundaries
    std::vector<double> bcy;    // y values of y-plane fixed boundaries

    double dt_recommend;               // maximum timestep for hydro
    char dt_recommend_mesg[80];          // message:  reason for dtrec

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

    Hydro(const InputParameters& params, Mesh* m,
    		DynamicCollective add_reduction,
		const PhysicalRegion &zones,
        Context ctx, HighLevelRuntime* rt);
    ~Hydro();

    void init();

    void initRadialVel(
            const double vel,
            const int pfirst,
            const int plast);

    void doCycle(const double dt);

    void advPosHalf(
            const double2* px0,
            const double2* pu0,
            const double dt,
            double2* pxp,
            const int pfirst,
            const int plast);

    void advPosFull(
            const double2* px0,
            const double2* pu0,
            const double2* pa,
            const double dt,
            double2* px,
            double2* pu,
            const int pfirst,
            const int plast);

    void calcCrnrMass(
            const DoubleAccessor& zr,
            const double* zarea,
            const double* smf,
            double* cmaswt,
            const int sfirst,
            const int slast);

    void sumCrnrForce(
            const double2* sf,
            const double2* sf2,
            const double2* sf3,
            double2* cftot,
            const int sfirst,
            const int slast);

    void calcAccel(
            const Double2Accessor& pf,
            const DoubleAccessor& pmass,
            double2* pa,
            const int pfirst,
            const int plast);

    void calcRho(
            const double* zm,
            const double* zvol,
            DoubleAccessor& zr,
            const int zfirst,
            const int zlast);

    void calcWork(
            const double2* sf,
            const double2* sf2,
            const double2* pu0,
            const double2* pu,
            const double2* px0,
            const double dt,
            double* zw,
            double* zetot,
            const int sfirst,
            const int slast);

    void calcWorkRate(
            const double* zvol0,
            const double* zvol,
            const double* zw,
            const DoubleAccessor& zp,
            const double dt,
            double* zwrate,
            const int zfirst,
            const int zlast);

    void calcEnergy(
            const double* zetot,
            const double* zm,
			const DoubleAccessor& ze,
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

    void calcDtCourant(
            const double* zdl,
            double& dtrec,
            char* msgdtrec,
            const int zfirst,
            const int zlast);

    void calcDtVolume(
            const double* zvol,
            const double* zvol0,
            const double dtlast,
            double& dtrec,
            char* msgdtrec,
            const int zfirst,
            const int zlast);

    void calcDtHydro(
            const double* zvol,
            const double* zvol0,
            const double dtlast,
            const int zfirst,
            const int zlast);

    void getDtHydro(
            double& dtnew,
            std::string& msgdtnew);

    void resetDtHydro();

    void writeEnergyCheck();

	DoubleAccessor zone_rho_;             // zone density // TODO make private
    DoubleAccessor zone_rho_pred_;        // zone density, middle of cycle
	DoubleAccessor zone_energy_density_;  // zone specific internal energy
    // (energy per unit mass)  // TODO make private
	DoubleAccessor zone_pressure_;        // zone pressure  // TODO make private
private:
	void allocateZoneFields();
	void allocatePtFields();
    void fillZoneAccessor(DoubleAccessor *acc, double value);

    Double2Accessor pt_force_;            // point force
    DoubleAccessor pt_weighted_mass_;     // point mass, weighted by 1/r

    FieldSpace fspace_local_pts_;
	LogicalRegion lregion_local_pts_;

    FieldSpace fspace_zones_;
	LogicalRegion lregion_local_zones_;

	IndexSpace ispace_zones_;
    DynamicCollective add_reduction_;
    Context ctx_;
    HighLevelRuntime* runtime_;

}; // class Hydro



#endif /* HYDRO_HH_ */
