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

#include "GenerateMesh.hh"
#include "InputParameters.hh"
#include "LogicalStructured.hh"
#include "Parallel.hh"
#include "Vec2.hh"

// forward declarations
class LocalMesh;


class Hydro {
public:

    Hydro(const InputParameters& params, LocalMesh* m,
    		DynamicCollective add_reduction,
        Context ctx, HighLevelRuntime* rt);

    // associated mesh object
    LocalMesh* mesh;

    double cfl;                 // Courant number, limits timestep
    double cflv;                // volume change limit for timestep
    double rho_init;               // initial density for main mesh
    double energy_init;               // initial energy for main mesh
    double rho_init_sub;            // initial density in subregion
    double energy_init_sub;            // initial energy in subregion
    double vel_init_radial;         // initial velocity in radial direction
    std::vector<double> bcx;    // x values of x-plane fixed boundaries
    std::vector<double> bcy;    // y values of y-plane fixed boundaries

    void init();

    void initRadialVel(
            const double vel,
            const int pfirst,
            const int plast,
            double2* pt_vel);

    TimeStep doCycle(const double dt);

    static void advPosHalf(
            const double dt,
            const int pfirst,
            const int plast,
            const double2* pt_x0,
            const double2* pt_vel0,
            double2* pt_x_pred);

    static void calcCrnrMass(
            const int sfirst,
            const int slast,
            const double* zone_area_pred,
            const double* side_mass_frac,
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const double* zone_rho_pred,
            double* crnr_weighted_mass);

    static void sumCrnrForce(
            const double2* side_force_pres,
            const double2* side_force_visc,
            const double2* side_force_tts,
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const int sfirst,
            const int slast,
            double2* crnr_force_tot);

    static void calcAccelAndAdvPosFull(
            const GenerateMesh* generate_mesh,
            const Double2Accessor pf,
            const DoubleAccessor pmass,
            const double dt,
            const double2* pt_vel0,
            const double2* pt_x0,
            double2* pt_vel,
            double2* pt_x,
            const int pfirst,
            const int plast);

    static void calcRho(
            const double* zvol,
            const double* zm,
            double* zr,
            const int zfirst,
            const int zlast);

    static void calcWork(
            const double dt,
            const int* map_side2pt1,
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const double2* side_force_pres,
            const double2* side_force_visc,
            const double2* pt_vel,
            const double2* pt_vel0,
            const double2* pt_x_pred,
            double* zone_energy_tot,
            double* zone_work,
            const int side_first,
            const int side_last);

    static void calcWorkRate(
            const double dt,
            const double* zone_vol,
            const double* zone_vol0,
            const double* zone_work,
            const double* zone_pressure,
            double* zone_work_rate,
            const int zfirst,
            const int zlast);

    static void calcEnergy(
            const double* zone_energy_tot,
            const double* zone_mass,
            double* zone_energy_density,
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

    static void calcDtCourant(
            double& dtrec,
            char* msgdtrec,
            const int zfirst,
            const int zlast,
            const double* zdl,
            const double* zone_dvel,
            const double* zone_sound_speed,
            const double cfl);

    static void calcDtVolume(
            const double dtlast,
            double& dtrec,
            char* msgdtrec,
            const int zfirst,
            const int zlast,
            const double* zvol,
            const double* zvol0,
            const double cflv);

    static void calcDtHydro(
            const double dtlast,
            const int zfirst,
            const int zlast,
            const double* zone_dl,
            const double* zone_dvel,
            const double* zone_sound_speed,
            const double cfl,
            const double* zone_vol,
            const double* zone_vol0,
            const double cflv,
            TimeStep& recommend);

    void writeEnergyCheck();

    void copyZonesToLegion(LogicalUnstructured& global_zones);

private:
	void  allocateFields();

    DynamicCollective add_reduction;
    Context ctx;
    HighLevelRuntime* runtime;
    LogicalStructured zones;
    LogicalStructured sides_and_corners;
    LogicalStructured edges;
    LogicalStructured points;
    const InputParameters params;
    const int my_color;

}; // class Hydro



#endif /* HYDRO_HH_ */
