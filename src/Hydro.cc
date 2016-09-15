/*
 * Hydro.cc
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Hydro.hh"

#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

#include "Memory.hh"
#include "Mesh.hh"
#include "PolyGas.hh"
#include "TTS.hh"
#include "QCS.hh"
#include "HydroBC.hh"

using namespace std;

// JPG TODO: declare const initialized in all constructors as const
Hydro::Hydro(const InputParameters& params, LocalMesh* m,
		DynamicCollective add_reduction,
		IndexSpace* ispace_zones,
		DoubleAccessor* zone_rho,
		DoubleAccessor* zone_energy_density,
		DoubleAccessor* zone_pressure,
		DoubleAccessor* zone_rho_pred,
        Context ctx, HighLevelRuntime* rt) :
		mesh(m),
		cfl(params.directs_.cfl_),
		cflv(params.directs_.cflv_),
		rho_init(params.directs_.rho_init_),
		energy_init(params.directs_.energy_init_),
		rho_init_sub(params.directs_.rho_init_sub_),
		energy_init_sub(params.directs_.energy_init_sub_),
		vel_init_radial(params.directs_.vel_init_radial_),
		bcx(params.bcx_),
		bcy(params.bcy_),
		zone_rho_(zone_rho),
		zone_rho_pred_(zone_rho_pred),
		zone_energy_density_(zone_energy_density),
		zone_pressure_(zone_pressure),
		ispace_zones_(ispace_zones),
		add_reduction_(add_reduction),
		ctx_(ctx),
		runtime_(rt),
		mype_(params.directs_.task_id_)
{
    pgas = new PolyGas(params, this);
    tts = new TTS(params, this);
    qcs = new QCS(params, this);

    const double2 vfixx = double2(1., 0.);
    const double2 vfixy = double2(0., 1.);
    for (int i = 0; i < bcx.size(); ++i)
        bcs.push_back(new HydroBC(mesh, vfixx, mesh->getXPlane(bcx[i])));
    for (int i = 0; i < bcy.size(); ++i)
        bcs.push_back(new HydroBC(mesh, vfixy, mesh->getYPlane(bcy[i])));

    init();
}


Hydro::~Hydro() {

    delete tts;
    delete qcs;
    for (int i = 0; i < bcs.size(); ++i) {
        delete bcs[i];
    }
}


void Hydro::init() {

    const int numpch = mesh->num_pt_chunks;
    const int numzch = mesh->num_zone_chunks;
    const int nump = mesh->num_pts_;
    const int numz = mesh->num_zones_;
    const int nums = mesh->num_sides_;

    const double2* zx = mesh->zone_x_;
    const double* zvol = mesh->zone_vol_;

    // allocate arrays
    pt_vel = AbstractedMemory::alloc<double2>(nump);
    pt_vel0 = AbstractedMemory::alloc<double2>(nump);
    pt_accel = AbstractedMemory::alloc<double2>(nump);
    crnr_weighted_mass = AbstractedMemory::alloc<double>(nums);
    zone_mass = AbstractedMemory::alloc<double>(numz);
    zone_energy_tot = AbstractedMemory::alloc<double>(numz);
    zone_work = AbstractedMemory::alloc<double>(numz);
    zone_work_rate = AbstractedMemory::alloc<double>(numz);
    zone_sound_speed = AbstractedMemory::alloc<double>(numz);
    zone_dvel = AbstractedMemory::alloc<double>(numz);
    side_force_pres = AbstractedMemory::alloc<double2>(nums);
    side_force_visc = AbstractedMemory::alloc<double2>(nums);
    side_force_tts = AbstractedMemory::alloc<double2>(nums);
    crnr_force_tot = AbstractedMemory::alloc<double2>(nums);

    // initialize hydro vars
    fillZoneAccessor(zone_rho_, rho_init);
    fillZoneAccessor(zone_energy_density_, energy_init);
    for (int zch = 0; zch < numzch; ++zch) {
        int zfirst = mesh->zone_chunk_first[zch];
        int zlast = mesh->zone_chunk_last[zch];

        fill(&zone_work_rate[zfirst], &zone_work_rate[zlast], 0.);

        const double& subrgn_xmin = mesh->subregion_xmin;
        const double& subrgn_xmax = mesh->subregion_xmax;
        const double& subrgn_ymin = mesh->subregion_ymin;
        const double& subrgn_ymax = mesh->subregion_ymax;
        if (subrgn_xmin != std::numeric_limits<double>::max()) {
            const double eps = 1.e-12;
            #pragma ivdep
            for (int z = zfirst; z < zlast; ++z) {
            		ptr_t zone_ptr(z); // TODO local to global ID
                if (zx[z].x > (subrgn_xmin - eps) &&
                    zx[z].x < (subrgn_xmax + eps) &&
                    zx[z].y > (subrgn_ymin - eps) &&
                    zx[z].y < (subrgn_ymax + eps)) {
                    zone_rho_->write(zone_ptr,rho_init_sub);
                    zone_energy_density_->write(zone_ptr,energy_init_sub);
                }
            }
        }

        #pragma ivdep
        for (int z = zfirst; z < zlast; ++z) {
        		ptr_t zone_ptr(z); // TODO local to global ID
        		zone_mass[z] = zone_rho_->read(zone_ptr) * zvol[z];
        		zone_energy_tot[z] = zone_energy_density_->read(zone_ptr) * zone_mass[z];
        }
    }  // for sch

    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = mesh->pt_chunks_first[pch];
        int plast = mesh->pt_chunks_last[pch];
        if (vel_init_radial != 0.)
            initRadialVel(vel_init_radial, pfirst, plast);
        else
            fill(&pt_vel[pfirst], &pt_vel[plast], double2(0., 0.));
    }  // for pch

    resetDtHydro();

}

void Hydro::fillZoneAccessor(DoubleAccessor* acc, double value)
{
	IndexIterator itr(runtime_, ctx_, *ispace_zones_);
	while (itr.has_next()) {
		ptr_t zone_ptr = itr.next();
		acc->write(zone_ptr, value);
	}
}

void Hydro::initRadialVel(
        const double vel,
        const int pfirst,
        const int plast) {
    const double eps = 1.e-12;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        double pmag = length(mesh->pt_x[p]);
        if (pmag > eps)
            pt_vel[p] = vel * mesh->pt_x[p] / pmag;
        else
            pt_vel[p] = double2(0., 0.);
    }
}


void Hydro::doCycle(
            const double dt) {

    const int num_pt_chunks = mesh->num_pt_chunks;
    const int num_side_chunks = mesh->num_side_chunks;

    // Begin hydro cycle
    // TODO use pointer and copy back later
    Double2Accessor point_force = mesh->local_points_by_gid.getRegionAccessor<double2>(FID_PF);
    for (int pt_chunk = 0; pt_chunk < num_pt_chunks; ++pt_chunk) {
        int pt_first = mesh->pt_chunks_first[pt_chunk];
        int pt_last = mesh->pt_chunks_last[pt_chunk];

        // save off point variable values from previous cycle
        copy(&mesh->pt_x[pt_first], &mesh->pt_x[pt_last], &mesh->pt_x0[pt_first]);
        copy(&pt_vel[pt_first], &pt_vel[pt_last], &pt_vel0[pt_first]);

        // ===== Predictor step =====
        // 1. advance mesh to center of time step
        advPosHalf(dt, pt_first, pt_last);
    } // for pch

    for (int sch = 0; sch < num_side_chunks; ++sch) {
        int sfirst = mesh->side_chunks_first[sch];
        int slast = mesh->side_chunks_last[sch];
        int zfirst = mesh->zone_chunks_first[sch];
        int zlast = mesh->zone_chunks_last[sch];

        // save off zone variable values from previous cycle
        copy(&mesh->zone_vol_[zfirst], &mesh->zone_vol_[zlast], &mesh->zone_vol0[zfirst]);

        // 1a. compute new mesh geometry
        mesh->calcCtrs(sch);
        mesh->calcVols(sch);
        mesh->calcMedianMeshSurfVecs(sch);
        mesh->calcEdgeLen(sch);
        mesh->calcCharacteristicLen(sch);

        // 2. compute point masses
        calcRho(mesh->zone_vol_pred, zone_rho_pred_, zfirst, zlast);
        calcCrnrMass(sfirst, slast);

        // 3. compute material state (half-advanced)
        pgas->calcStateAtHalf(zone_rho_, mesh->zone_vol_pred, mesh->zone_vol0, zone_energy_density_, zone_work_rate, zone_mass, dt,
                zone_pressure_, zone_sound_speed, zfirst, zlast);

        // 4. compute forces
        pgas->calcForce(zone_pressure_, mesh->side_surfp, side_force_pres, sfirst, slast);
        tts->calcForce(mesh->zone_area_pred, zone_rho_pred_, zone_sound_speed, mesh->side_area_pred, mesh->side_mass_frac, mesh->side_surfp, side_force_tts,
                sfirst, slast);
        qcs->calcForce(side_force_visc, sfirst, slast);
        sumCrnrForce(sfirst, slast);
    }  // for sch
    mesh->checkBadSides();

    // sum corner masses, forces to points
    mesh->sumToPoints(crnr_weighted_mass, crnr_force_tot);

    for (int pch = 0; pch < num_pt_chunks; ++pch) {
        int pfirst = mesh->pt_chunks_first[pch];
        int plast = mesh->pt_chunks_last[pch];

        // 4a. apply boundary conditions
        for (int i = 0; i < bcs.size(); ++i) {
            int bfirst = bcs[i]->pchbfirst[pch];
            int blast = bcs[i]->pchblast[pch];
            bcs[i]->applyFixedBC(pt_vel0, point_force, bfirst, blast); // TODO double2*
        }

        // 5. compute accelerations
        calcAccel(pfirst, plast);

        // ===== Corrector step =====
        // 6. advance mesh to end of time step
        advPosFull(dt, pfirst, plast);
    }  // for pch

    resetDtHydro();

    for (int sch = 0; sch < num_side_chunks; ++sch) {
        int sfirst = mesh->side_chunks_first[sch];
        int slast = mesh->side_chunks_last[sch];
        int zfirst = mesh->zone_chunks_first[sch];
        int zlast = mesh->zone_chunks_last[sch];

        // 6a. compute new mesh geometry
        mesh->calcCtrs(sch, false);
        mesh->calcVols(sch, false);

        // 7. compute work
        fill(&zone_work[zfirst], &zone_work[zlast], 0.);
        calcWork(dt, sfirst, slast);
    }  // for sch
    mesh->checkBadSides();

    for (int zch = 0; zch < mesh->num_zone_chunks; ++zch) {
        int zfirst = mesh->zone_chunk_first[zch];
        int zlast = mesh->zone_chunk_last[zch];

        // 7a. compute work rate
        calcWorkRate(dt, zfirst, zlast);

        // 8. update state variables
        calcEnergy(zfirst, zlast);
        calcRho(mesh->zone_vol_, zone_rho_, zfirst, zlast);

        // 9.  compute timestep for next cycle
        calcDtHydro(dt, zfirst, zlast);
    }  // for zch

}


void Hydro::advPosHalf(
        const double dt,
        const int pfirst,
        const int plast) {

    double dth = 0.5 * dt;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        mesh->pt_x_pred[p] = mesh->pt_x0[p] + pt_vel0[p] * dth;
    }
}

void Hydro::advPosFull(
        const double dt,
        const int pfirst,
        const int plast) {

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        pt_vel[p] = pt_vel0[p] + pt_accel[p] * dt;
        mesh->pt_x[p] = mesh->pt_x0[p] + 0.5 * (pt_vel[p] + pt_vel0[p]) * dt;
    }

}


void Hydro::calcCrnrMass(
        const int sfirst,
        const int slast) {
    const DoubleAccessor* zr = zone_rho_pred_; // TODO go back to pointers and copy at end
    const double* zarea = mesh->zone_area_pred;
    const double* side_mass_frac = mesh->side_mass_frac;

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int s3 = mesh->mapSideToSidePrev(s);
        int z = mesh->map_side2zone_[s];

        	ptr_t zone_ptr(z);
        double m = zr->read(zone_ptr) * zarea[z] * 0.5 * (side_mass_frac[s] + side_mass_frac[s3]);
        crnr_weighted_mass[s] = m;
    }
}


void Hydro::sumCrnrForce(
        const int sfirst,
        const int slast) {

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int s3 = mesh->mapSideToSidePrev(s);

        double2 f = (side_force_pres[s] + side_force_visc[s] + side_force_tts[s]) -
                    (side_force_pres[s3] + side_force_visc[s3] + side_force_tts[s3]);
        crnr_force_tot[s] = f;
    }
}


void Hydro::calcAccel(
        const int pfirst,
        const int plast) {
    const Double2Accessor pf = mesh->local_points_by_gid.getRegionAccessor<double2>(FID_PF);
    const DoubleAccessor pmass = mesh->local_points_by_gid.getRegionAccessor<double>(FID_PMASWT);
    double2* pa = pt_accel;

    const double fuzz = 1.e-99;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {  // TODO pf and pmass use gid
    		ptr_t pt_ptr(p);
        pa[p] = pf.read(pt_ptr) / max(pmass.read(pt_ptr), fuzz);
    }

}


void Hydro::calcRho(
        const double* zvol,
        DoubleAccessor* zr,
        const int zfirst,
        const int zlast) {
    const double* zm = zone_mass;

    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
    		ptr_t zone_ptr(z);
        zr->write(zone_ptr, zm[z] / zvol[z]);
    }

}

void Hydro::calcWork(
        const double dt,
        const int sfirst,
        const int slast) {
    // Compute the work done by finding, for each element/node pair,
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const double dth = 0.5 * dt;

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mesh->map_side2pt1_[s];
        int p2 = mesh->mapSideToPt2(s);
        int z = mesh->map_side2zone_[s];

        double2 sftot = side_force_pres[s] + side_force_visc[s];
        double sd1 = dot( sftot, (pt_vel0[p1] + pt_vel[p1]));
        double sd2 = dot(-sftot, (pt_vel0[p2] + pt_vel[p2]));
        double dwork = -dth * (sd1 * mesh->pt_x_pred[p1].x + sd2 * mesh->pt_x_pred[p2].x);

        zone_energy_tot[z] += dwork;
        zone_work[z] += dwork;

    }

}


void Hydro::calcWorkRate(
        const double dt,
        const int zfirst,
        const int zlast) {
    double dtinv = 1. / dt;
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
    		ptr_t zone_ptr(z);
        double dvol = mesh->zone_vol_[z] - mesh->zone_vol0[z];
        zone_work_rate[z] = (zone_work[z] + zone_pressure_->read(zone_ptr) * dvol) * dtinv;
    }

}


void Hydro::calcEnergy(
        const int zfirst,
        const int zlast) {

    const double fuzz = 1.e-99;
    #pragma ivdep
    for (int z = zfirst; z < zlast; ++z) {
    		ptr_t zone_ptr(z); // TODO gid problems deferred to end of cycle
        zone_energy_density_->write(zone_ptr, zone_energy_tot[z] / (zone_mass[z] + fuzz));
    }

}


void Hydro::sumEnergy(
        const double* zetot,
        const double* zarea,
        const double* zvol,
        const double* zm,
        const double* side_mass_frac,
       const double2* px,
        const double2* pu,
        double& ei,
        double& ek,
        const int zfirst,
        const int zlast,
        const int sfirst,
        const int slast) {

    // compute internal energy
    double sumi = 0.; 
    for (int z = zfirst; z < zlast; ++z) {
        sumi += zetot[z];
    }
    // multiply by 2\pi for cylindrical geometry
    ei += sumi * 2 * M_PI;

    // compute kinetic energy
    // in each individual zone:
    // zone ke = zone mass * (volume-weighted average of .5 * u ^ 2)
    //         = zm sum(c in z) [cvol / zvol * .5 * u ^ 2]
    //         = sum(c in z) [zm * cvol / zvol * .5 * u ^ 2]
    double sumk = 0.; 
    for (int s = sfirst; s < slast; ++s) {
        int s3 = mesh->mapSideToSidePrev(s);
        int p1 = mesh->map_side2pt1_[s];
        int z = mesh->map_side2zone_[s];

        double cvol = zarea[z] * px[p1].x * 0.5 * (side_mass_frac[s] + side_mass_frac[s3]);
        double cke = zm[z] * cvol / zvol[z] * 0.5 * length2(pu[p1]);
        sumk += cke;
    }
    // multiply by 2\pi for cylindrical geometry
    ek += sumk * 2 * M_PI;

}


void Hydro::calcDtCourant(
        double& dtrec,
        char* msgdtrec,
        const int zfirst,
        const int zlast) {
    const double* zdl = mesh->zone_dl;

    const double fuzz = 1.e-99;
    double dtnew = 1.e99;
    int zmin = -1;
    for (int z = zfirst; z < zlast; ++z) {
        double cdu = max(zone_dvel[z], max(zone_sound_speed[z], fuzz));
        double zdthyd = zdl[z] * cfl / cdu;
        zmin = (zdthyd < dtnew ? z : zmin);
        dtnew = (zdthyd < dtnew ? zdthyd : dtnew);
    }

    if (dtnew < dtrec) {
        dtrec = dtnew;
        snprintf(msgdtrec, 80, "Hydro Courant limit for z = %d", zmin);
    }

}


void Hydro::calcDtVolume(
        const double dtlast,
        double& dtrec,
        char* msgdtrec,
        const int zfirst,
        const int zlast) {
    const double* zvol = mesh->zone_vol_;
    const double* zvol0 = mesh->zone_vol0;

    double dvovmax = 1.e-99;
    int zmax = -1;
    for (int z = zfirst; z < zlast; ++z) {
        double zdvov = abs((zvol[z] - zvol0[z]) / zvol0[z]);
        zmax = (zdvov > dvovmax ? z : zmax);
        dvovmax = (zdvov > dvovmax ? zdvov : dvovmax);
    }
    double dtnew = dtlast * cflv / dvovmax;
    if (dtnew < dtrec) {
        dtrec = dtnew;
        snprintf(msgdtrec, 80, "Hydro dV/V limit for z = %d", zmax);
    }

}


void Hydro::calcDtHydro(
        const double dtlast,
        const int zfirst,
        const int zlast) {

    double dtchunk = 1.e99;
    char msgdtchunk[80];

    calcDtCourant(dtchunk, msgdtchunk, zfirst, zlast);
    calcDtVolume(dtlast, dtchunk, msgdtchunk, zfirst, zlast);
    if (dtchunk < dt_recommend) {
        {
            // redundant test needed to avoid race condition
            if (dtchunk < dt_recommend) {
                dt_recommend = dtchunk;
                strncpy(dt_recommend_mesg, msgdtchunk, 80);
            }
        }
    }

}


void Hydro::getDtHydro(
        double& dtnew,
        string& msgdtnew) {

    if (dt_recommend < dtnew) {
        dtnew = dt_recommend;
        msgdtnew = string(dt_recommend_mesg);
    }

}


void Hydro::resetDtHydro() {

    dt_recommend = 1.e99;
    strcpy(dt_recommend_mesg, "Hydro default");

}


void Hydro::writeEnergyCheck() {

    double ei = 0.;
    double ek = 0.;
    for (int sch = 0; sch < mesh->num_side_chunks; ++sch) {
        int sfirst = mesh->side_chunks_first[sch];
        int slast = mesh->side_chunks_last[sch];
        int zfirst = mesh->zone_chunks_first[sch];
        int zlast = mesh->zone_chunks_last[sch];

        double eichunk = 0.;
        double ekchunk = 0.;
        sumEnergy(zone_energy_tot, mesh->zone_area_, mesh->zone_vol_, zone_mass, mesh->side_mass_frac,
                mesh->pt_x, pt_vel, eichunk, ekchunk,
                zfirst, zlast, sfirst, slast);
        {
            ei += eichunk;
            ek += ekchunk;
        }
    }


	Future future_sum = Parallel::globalSum(ei, add_reduction_, runtime_, ctx_);
	ei = future_sum.get_result<double>();

	future_sum = Parallel::globalSum(ek, add_reduction_, runtime_, ctx_);
	ek = future_sum.get_result<double>();

    if (mype_ == 0) {
        cout << scientific << setprecision(6);
        cout << "Energy check:  "
             << "total energy  = " << setw(14) << ei + ek << endl;
        cout << "(internal = " << setw(14) << ei
             << ", kinetic = " << setw(14) << ek << ")" << endl;
    }

 }
