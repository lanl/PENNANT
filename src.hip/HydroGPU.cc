/*
 * HydroGPU.cu
 *
 *  Created on: Aug 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Triad National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "HydroGPU.hh"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <hip/hip_runtime.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#ifdef USE_MPI
#include "Parallel.hh"
#include "HydroMPI.hh"
#endif

#include "Memory.hh"
#include "Vec2.hh"

#ifdef USE_JIT
#include "pajama.h"
#endif

// stuff used while developing zone-based version of gpuMain2. TODO: remove when done.

double *zvol_zbD, *zvol_cpD;
__constant__ double *zvol_zb, *zvol_cp;

double *zvol0_zbD, *zvol0_cpD;
__constant__ double *zvol0_zb, *zvol0_cp;

double *zdl_zbD, *zdl_cpD;
__constant__ double *zdl_zb, *zdl_cp;

double *zp_zbD, *zp_cpD;
__constant__ double *zp_zb, *zp_cp;

double *zss_zbD, *zss_cpD;
__constant__ double *zss_zb, *zss_cp;

double *cmaswt_zbD, *cmaswt_cpD;
__constant__ double *cmaswt_zb, *cmaswt_cp;

double *zrp_zbD, *zrp_cpD;
__constant__ double *zrp_zb, *zrp_cp;

double2 *zxp_zbD, *zxp_cpD;
__constant__ double2 *zxp_zb, *zxp_cp;

double *sareap_zbD, *sareap_cpD;
__constant__ double *sareap_zb, *sareap_cp;

double *svolp_zbD, *svolp_cpD;
__constant__ double *svolp_zb, *svolp_cp;

double *zareap_zbD, *zareap_cpD;
__constant__ double *zareap_zb, *zareap_cp;

double *zvolp_zbD, *zvolp_cpD;
__constant__ double *zvolp_zb, *zvolp_cp;

double2 *ssurf_zbD, *ssurf_cpD;
__constant__ double2 *ssurf_zb, *ssurf_cp;

// end of stuff used while developing zone-based version of gpuMain2. TODO: remove when done.

using namespace std;


const int CHUNK_SIZE = 64;
int numz_zb; // temporary global used for developing zone-base version of gpuMain2. TODO: remove.
int nums_zb; // temporary global used for developing zone-base version of gpuMain2. TODO: remove.

#ifdef USE_MPI
__constant__ int numslv;
__constant__ int numprx;
__constant__ int* mapprxp;
__constant__ int* mapslvp;
#endif

__constant__ int nump;
__constant__ int numz;
__constant__ int nums;
__constant__ double dt;
__constant__ double pgamma, pssmin;
__constant__ double talfa, tssmin;
__constant__ double qgamma, q1, q2;
__constant__ double hcfl, hcflv;
__constant__ double2 vfixx, vfixy;
__constant__ int numbcx, numbcy;
__constant__ double bcx[2], bcy[2];

__constant__ int numsbad;
__constant__ double dtnext;
__constant__ int idtnext;

__constant__ const int* schsfirst;
__constant__ const int* schslast;
__constant__ const int* mapsp1;
__constant__ const int* mapsp2;
__constant__ const int* mapsz;
__constant__ const int* mapzs;
__constant__ const int* mapss4;
__constant__ const int *mappsfirst, *mapssnext;
__constant__ const int* znump;

__constant__ double2 *px, *pxp;
__constant__ double2 *zx, *zxp;
__constant__ double2 *pu, *pu0;
__constant__ double2* pap;
__constant__ double2* ssurf;
__constant__ const double* zm;
__constant__ double *zr, *zrp;
__constant__ double *ze, *zetot;
__constant__ double *zw, *zwrate;
__constant__ double *zp, *zss;
__constant__ const double* smf;
__constant__ double *careap, *sareap, *svolp, *zareap, *zvolp;
__constant__ double *sarea, *svol, *zarea, *zvol, *zvol0;
__constant__ double *zdl, *zdu;
__constant__ double *cmaswt, *pmaswt;
__constant__ double2 *sfp, *sft, *sfq, *cftot, *pf;
__constant__ double* cevol;
__constant__ double* cdu;
__constant__ double* cdiv;
__constant__ double2* zuc;
__constant__ double2* cqe;
__constant__ double* ccos;
__constant__ double* cw;

int numschH, numpchH, numzchH;
int* numsbadD;
int *schsfirstH, *schslastH, *schzfirstH, *schzlastH;
int *schsfirstD, *schslastD, *schzfirstD, *schzlastD;
int *mapsp1D, *mapsp2D, *mapszD, *mapzsD, *mapss4D, *znumpD;
int *mapspkeyD, *mapspvalD;
int *mappsfirstD, *mapssnextD;
double2 *pxD, *pxpD, *zxD, *zxpD, *puD, *pu0D, *papD,
    *ssurfD, *sfpD, *sftD, *sfqD, *cftotD, *pfD, *zucD, *cqeD;
double *zmD, *zrD, *zrpD,
    *sareaD, *svolD, *zareaD, *zvolD, *zvol0D, *zdlD, *zduD,
    *zeD, *zetot0D, *zetotD, *zwD, *zwrateD,
    *zpD, *zssD, *smfD, *careapD, *sareapD, *svolpD, *zareapD, *zvolpD;
double *cmaswtD, *pmaswtD;
double *cevolD, *cduD, *cdivD, *crmuD, *ccosD, *cwD;

#ifdef USE_MPI
int nummstrpeD, numslvpeD;
int *mapslvpepeD, *mapslvpeprx1D, *mapprxpD, *slvpenumprxD, *mapmstrpepeD,
  *mstrpenumslvD, *mapmstrpeslv1D, *mapslvpD, *mapslvpD1;
int numslvH, numprxH;

// We need to communnicate data between slave points on our rank to proxy points on other ranks,
// and between proxy points on our ranks and slave points on other ranks.
// Since point data is not consecutive in memory (which is desirable for MPI comms, to minimize the
// number of Send/Recv ops), we allocate array buffers as a staging area for the data to be exchanged.
// Properties that need to be send/received: pmaswt (doubles) and pf (double2s).
// for both of these properties, we'll have arrays on the device (*_D).
// If we cannot use GPU-aware MPI, we'll have to copy data between host and device before and after
// MPI comms. In that case, we use additional arrays on the host (*_H).
// To simplify the MPI code, we'll use one additional pointer for each *_D and (optional) *_H buffer:
// this pointer equals the device pointer if we can use GPU-aware MPI, and it equals the host pointer otherwise.
// The name of the latter pointer equals the device/host pointer without the _D/_H suffix.

double *pmaswt_proxy_buffer, *pmaswt_slave_buffer;      // copies of either *_D or *_H pointers
double2 *pf_proxy_buffer, *pf_slave_buffer;
double *pmaswt_proxy_buffer_D, *pmaswt_slave_buffer_D;  // pointers used to allocate device memory
double2 *pf_proxy_buffer_D, *pf_slave_buffer_D;
#ifndef USE_GPU_AWARE_MPI
double *pmaswt_proxy_buffer_H, *pmaswt_slave_buffer_H;  // pointers used to allocat host memory in case
double2 *pf_proxy_buffer_H, *pf_slave_buffer_H;         // we can't do MPI transfers from/to device memory
#endif // USE_GPU_AWARE_MPI
#endif // USE_MPI

#ifdef USE_JIT
std::unique_ptr<Pajama> jit;
#endif

int checkCudaError(const hipError_t err, const char* cmd)
{
    if(err) {
        printf("CUDA error in command '%s'\n", cmd); \
        printf("Error message: %s\n", hipGetErrorString(err)); \
    }
    return err;
}

#define CHKERR(cmd) checkCudaError(cmd, #cmd)


__device__ void advPosHalf(
        const int p,
        const double2* __restrict__ px,
        const double2* __restrict__ pu0,
        const double dt,
        double2* __restrict__ pxp) {

    pxp[p] = px[p] + pu0[p] * dt;

}


inline __device__ void calcZoneCtrs_zb(int z,
				       int local_tid,
				       const double2* __restrict__ sh_p1x,
				       double2*__restrict__  zx){
  constexpr int sides_per_zone = 4;
  double2 zxtot = {0., 0.};
  auto s_sh_idx = local_tid * sides_per_zone;
  for(int s = 0; s != sides_per_zone; ++s, ++s_sh_idx){
    zxtot += sh_p1x[s_sh_idx];
  }
  constexpr double recip = 1. / sides_per_zone;
  zx[z] = recip * zxtot;
}


__device__ void calcZoneCtrs(
        const int s,
        const int s0,
        const int z,
        const int p1,
        const double2* __restrict__ px,
        double2* __restrict__ zx,
	int dss4[CHUNK_SIZE],
	double2 ctemp2[CHUNK_SIZE]) {

    ctemp2[s0] = px[p1];
    __syncthreads();

    double2 zxtot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zxtot += ctemp2[sn];
        zct += 1.;
    }
    zx[z] = zxtot / zct;
}


__device__ void calcSideVols(
    const int s,
    const int z,
    const int p1,
    const int p2,
    const double2* __restrict__ px,
    const double2* __restrict__ zx,
    double* __restrict__ sarea,
    double* __restrict__ svol)
{
    const double third = 1. / 3.;
    double sa = 0.5 * cross(px[p2] - px[p1],  zx[z] - px[p1]);
    double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
    sarea[s] = sa;
    svol[s] = sv;
    
    if (sv <= 0.) atomicAdd(&numsbad, 1);
}


__device__ void calcZoneVols(
    const int s,
    const int s0,
    const int z,
    const double* __restrict__ sarea,
    const double* __restrict__ svol,
    double* __restrict__ zarea,
    double* __restrict__ zvol)
{
    // make sure all side volumes have been stored
    __syncthreads();

    double zatot = sarea[s];
    double zvtot = svol[s];
    for (int sn = mapss4[s]; sn != s; sn = mapss4[sn]) {
        zatot += sarea[sn];
        zvtot += svol[sn];
    }
    zarea[z] = zatot;
    zvol[z] = zvtot;
}


// This function combines the calculations of the folling device
// functions:
// - meshCalcCharLen
// - computation of ssurf in gpuMain2
// - calcSideVols
// - calcZoneVols
inline void __device__ fusedZoneSideUpdates_zb(int z,
					       int local_tid,
					       int first_side_in_block,
					       const double2* __restrict__ sh_p1x,
					       const double2* __restrict__ sh_workspace,
					       const int* __restrict__ znump,
					       const double2* __restrict__ px,
					       const double2* __restrict__ zx,
					       const double* __restrict__ zm,
					       const double* __restrict__ smf,
					       double* __restrict__ zdl,
					       double2* __restrict__ ssurf, 
					       double* __restrict__ sarea,
					       double* __restrict__ svol,
					       double* __restrict__ zarea,
					       double* __restrict__ zvol,
					       double* __restrict__ zr,
					       double* __restrict__ cmaswt){
  constexpr int sides_per_zone = 4;
  auto s_global = first_side_in_block + local_tid * sides_per_zone;
  auto s_sh_idx = local_tid * sides_per_zone;

  auto zxz = zx[z];
  double sdlmin = std::numeric_limits<double>::max();
  constexpr double third = 1. / 3.;
  double summed_side_area = 0.;
  double summed_side_volume = 0.;
  // TODO: consider the optimization from commit 20654d
  for(int s = 0; s != sides_per_zone; ++s, ++s_global, ++s_sh_idx){
    // computation of zdl values -----
    auto pxp1 = sh_p1x[s_sh_idx];
    auto pxp2 = sh_p1x[s != sides_per_zone - 1 ? s_sh_idx + 1 : s_sh_idx - sides_per_zone + 1];
    double side_area = 0.5 * cross(pxp2 - pxp1, zxz - pxp1);
    double base = length(pxp2 - pxp1);
    double sdl = side_area / base; // TODO: can we simplify this computation?
    sdlmin = min(sdlmin, sdl);
    // fused computation of ssurf values ----
    ssurf[s_global] = rotateCCW(0.5 * (pxp1 + pxp2) - zxz);
  }

  // reset s_global and s_sh_index
  s_global = first_side_in_block + local_tid * sides_per_zone;
  s_sh_idx = local_tid * sides_per_zone;
  for(int s = 0; s != sides_per_zone; ++s, ++s_global, ++s_sh_idx){
    // computation of zdl values -----
    auto pxp1 = sh_p1x[s_sh_idx];
    auto pxp2 = sh_p1x[s != sides_per_zone - 1 ? s_sh_idx + 1 : s_sh_idx - sides_per_zone + 1];
    double side_area = 0.5 * cross(pxp2 - pxp1, zxz - pxp1);
    sarea[s_global] = side_area;
    // fused: update summed_side_area for compuation of zone area
    summed_side_area += side_area;
    // fused: compute and store side volume
    double side_volume = third * side_area * (pxp1.x + pxp2.x + zxz.x);
    svol[s_global] = side_volume;
    // fused: update summed_side_volume for computation of zone volume
    summed_side_volume += side_volume;
  }

  // finalization of computation of zdl values
  constexpr double fac = sides_per_zone == 3 ? 3.0 : 4.0;
  sdlmin *= fac;
  zdl[z] = sdlmin;
  // fused: store zone area
  zarea[z] = summed_side_area;
  // fused: store zone volume
  zvol[z] = summed_side_volume;
  // fused: calculate and store zone rho values
  zr[z] = zm[z] / zvol[z];
  // fused: calculate cmaswt values

  // reset s_global
  s_global = first_side_in_block + local_tid * sides_per_zone;
  auto s3 = s_global + sides_per_zone - 1;
  auto smf_s3 = smf[s3];
  auto partial_m = 0.5 * zr[z] * zarea[z];
  for(int s = 0; s != sides_per_zone; ++s, ++s_global){
    auto smf_s = smf[s_global];
    cmaswt[s_global] = partial_m * (smf_s + smf_s3);
    smf_s3 = smf_s;
  }
}


__device__ void meshCalcCharLen(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const int* __restrict__ znump,
        const double2* __restrict__ px,
        const double2* __restrict__ zx,
        double* __restrict__ zdl,
	int dss4[CHUNK_SIZE],
	double ctemp[CHUNK_SIZE] ) {

    double area = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
    double base = length(px[p2] - px[p1]);
    double fac = (znump[z] == 3 ? 3. : 4.);
    double sdl = fac * area / base;

    ctemp[s0] = sdl;
    __syncthreads();
    double sdlmin = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        sdlmin = min(sdlmin, ctemp[sn]);
    }
    zdl[z] = sdlmin;
}

__device__ void hydroCalcRho(const int z,
        const double* __restrict__ zm,
        const double* __restrict__ zvol,
        double* __restrict__ zr)
{
    zr[z] = zm[z] / zvol[z];
}


__device__ void pgasCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zp,
        const double2* __restrict__ ssurf,
        double2* __restrict__ sf) {
    sf[s] = -zp[z] * ssurf[s];
}


__device__ void ttsCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zarea,
        const double* __restrict__ zr,
        const double* __restrict__ zss,
        const double* __restrict__ sarea,
        const double* __restrict__ smf,
        const double2* __restrict__ ssurf,
        double2* __restrict__ sf) {
    double svfacinv = zarea[z] / sarea[s];
    double srho = zr[z] * smf[s] * svfacinv;
    double sstmp = max(zss[z], tssmin);
    sstmp = talfa * sstmp * sstmp;
    double sdp = sstmp * (srho - zr[z]);
    sf[s] = -sdp * ssurf[s];
}


// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
__device__ void qcsSetCornerDiv(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
	int dss4[CHUNK_SIZE],
	double2 ctemp2[CHUNK_SIZE]) {

    // [1] Compute a zone-centered velocity
    ctemp2[s0] = pu[p1];
    __syncthreads();

    double2 zutot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zutot += ctemp2[sn];
        zct += 1.;
    }
    zuc[z] = zutot / zct;

    // [2] Divergence at the corner
    // Associated zone, corner, point
    const int p0 = mapsp1[s3];
    double2 up0 = pu[p1];
    double2 xp0 = pxp[p1];
    double2 up1 = 0.5 * (pu[p1] + pu[p2]);
    double2 xp1 = 0.5 * (pxp[p1] + pxp[p2]);
    double2 up2 = zuc[z];
    double2 xp2 = zxp[z];
    double2 up3 = 0.5 * (pu[p0] + pu[p1]);
    double2 xp3 = 0.5 * (pxp[p0] + pxp[p1]);

    // position, velocity diffs along diagonals
    double2 up2m0 = up2 - up0;
    double2 xp2m0 = xp2 - xp0;
    double2 up3m1 = up3 - up1;
    double2 xp3m1 = xp3 - xp1;

    // average corner-centered velocity
    double2 duav = 0.25 * (up0 + up1 + up2 + up3);

    // compute cosine angle
    double2 v1 = xp1 - xp0;
    double2 v2 = xp3 - xp0;
    double de1 = length(v1);
    double de2 = length(v2);
    double minelen = 2.0 * min(de1, de2);
    ccos[s] = (minelen < 1.e-12 ? 0. : dot(v1, v2) / (de1 * de2));

        // compute 2d cartesian volume of corner
    double cvolume = 0.5 * cross(xp2m0, xp3m1);
    careap[s] = cvolume;

    // compute velocity divergence of corner
    cdiv[s] = (cross(up2m0, xp3m1) - cross(up3m1, xp2m0)) /
            (2.0 * cvolume);

    // compute delta velocity
    double dv1 = length2(up2m0 - up3m1);
    double dv2 = length2(up2m0 + up3m1);
    double du = sqrt(max(dv1, dv2));
    cdu[s]   = (cdiv[s] < 0.0 ? du   : 0.);

    // compute evolution factor
    double2 dxx1 = 0.5 * (xp2m0 - xp3m1);
    double2 dxx2 = 0.5 * (xp2m0 + xp3m1);
    double dx1 = length(dxx1);
    double dx2 = length(dxx2);

    double test1 = abs(dot(dxx1, duav) * dx2);
    double test2 = abs(dot(dxx2, duav) * dx1);
    double num = (test1 > test2 ? dx1 : dx2);
    double den = (test1 > test2 ? dx2 : dx1);
    double r = num / den;
    double evol = sqrt(4.0 * cvolume * r);
    evol = min(evol, 2.0 * minelen);
    cevol[s] = (cdiv[s] < 0.0 ? evol : 0.);

}


// Routine number [4]  in the full algorithm CS2DQforce(...)
__device__ void qcsSetQCnForce(
        const int s,
        const int s3,
        const int z,
        const int p1,
        const int p2) {

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the rmu (real Kurapatenko viscous scalar)
    // Kurapatenko form of the viscosity
    double ztmp2 = q2 * 0.25 * gammap1 * cdu[s];
    double ztmp1 = q1 * zss[z];
    double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
    // Compute rmu for each corner
    double rmu = zkur * zrp[z] * cevol[s];
    rmu = (cdiv[s] > 0. ? 0. : rmu);

    // [4.2] Compute the cqe for each corner
    const int p0 = mapsp1[s3];
    const double elen1 = length(pxp[p1] - pxp[p0]);
    const double elen2 = length(pxp[p2] - pxp[p1]);
    // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
    //          cqe(2,1,3)=edge 2, x component (1st)
    cqe[2 * s]     = rmu * (pu[p1] - pu[p0]) / elen1;
    cqe[2 * s + 1] = rmu * (pu[p2] - pu[p1]) / elen2;
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
__device__ void qcsSetForce(
        const int s,
        const int s4,
        const int p1,
        const int p2) {

    // [5.1] Preparation of extra variables
    double csin2 = 1. - ccos[s] * ccos[s];
    cw[s]   = ((csin2 < 1.e-4) ? 0. : careap[s] / csin2);
    ccos[s] = ((csin2 < 1.e-4) ? 0. : ccos[s]);
    __syncthreads();

    // [5.2] Set-Up the forces on corners
    const double2 x1 = pxp[p1];
    const double2 x2 = pxp[p2];
    // Edge length for c1, c2 contribution to s
    double elen = length(x1 - x2);
    sfq[s] = (cw[s] * (cqe[2*s+1] + ccos[s] * cqe[2*s]) +
             cw[s4] * (cqe[2*s4] + ccos[s4] * cqe[2*s4+1]))
            / elen;
}


// Routine number [6]  in the full algorithm
__device__ void qcsSetVelDiff(
        const int s,
        const int s0,
        const int p1,
        const int p2,
        const int z,
	int dss4[CHUNK_SIZE],
	double ctemp[CHUNK_SIZE] ) {

    double2 dx = pxp[p2] - pxp[p1];
    double2 du = pu[p2] - pu[p1];
    double lenx = length(dx);
    double dux = dot(du, dx);
    dux = (lenx > 0. ? abs(dux) / lenx : 0.);

    ctemp[s0] = dux;
    __syncthreads();

    double ztmp = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        ztmp = max(ztmp, ctemp[sn]);
    }
    __syncthreads();

    zdu[z] = q1 * zss[z] + 2. * q2 * ztmp;
}


__device__ void qcsCalcForce(
        const int s,
        const int s0,
        const int s3,
        const int s4,
        const int z,
        const int p1,
        const int p2,
	int dss3[CHUNK_SIZE],
	int dss4[CHUNK_SIZE],
	double ctemp[CHUNK_SIZE],
	double2 ctemp2[CHUNK_SIZE]) {
    // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv(s, s0, s3, z, p1, p2,dss4, ctemp2);

    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce(s, s3, z, p1, p2);

    // [5] Compute the Q forces
    qcsSetForce(s, s4, p1, p2);

    ctemp2[s0] = sfp[s] + sft[s] + sfq[s];
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff(s, s0, p1, p2, z, dss4, ctemp);

}


__device__ void calcCrnrMass(
    const int s,
    const int s3,
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ zarea,
    const double* __restrict__ smf,
    double* __restrict__ cmaswt)
{
    double m = zr[z] * zarea[z] * 0.5 * (smf[s] + smf[s3]);
    cmaswt[s] = m;
}


__device__ void pgasCalcEOS(
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ ze,
    double* __restrict__ zp,
    double& zper,
    double* __restrict__ zss)
{
    const double gm1 = pgamma - 1.;
    const double ss2 = fmax(pssmin * pssmin, 1.e-99);

    double rx = zr[z];
    double ex = max(ze[z], 0.0);
    double px = gm1 * rx * ex;
    double prex = gm1 * ex;
    double perx = gm1 * rx;
    double csqd = max(ss2, prex + perx * px / (rx * rx));
    zp[z] = px;
    zper = perx;
    zss[z] = sqrt(csqd);
}


__device__ void pgasCalcStateAtHalf(
    const int z,
    const double* __restrict__ zr0,
    const double* __restrict__ zvolp,
    const double* __restrict__ zvol0,
    const double* __restrict__ ze,
    const double* __restrict__ zwrate,
    const double* __restrict__ zm,
    const double dt,
    double* __restrict__ zp,
    double* __restrict__ zss)
{
    double zper;
    pgasCalcEOS(z, zr0, ze, zp, zper, zss);

    const double dth = 0.5 * dt;
    const double zminv = 1. / zm[z];
    double dv = (zvolp[z] - zvol0[z]) * zminv;
    double bulk = zr0[z] * zss[z] * zss[z];
    double denom = 1. + 0.5 * zper * dv;
    double src = zwrate[z] * dth * zminv;
    zp[z] += (zper * src - zr0[z] * bulk * dv) / denom;
}


__global__ void gpuInvMap(
        const int* mapspkey,
        const int* mapspval,
        int* mappsfirst,
        int* mapssnext)
{
    const int i = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (i >= nums) return;

    int p = mapspkey[i];
    int pp = mapspkey[i+1];
    int pm = i == 0 ? -1 : mapspkey[i-1];
    int s = mapspval[i];
    int sp = mapspval[i+1];

    if (i == 0 || p != pm)  mappsfirst[p] = s;
    if (i+1 == nums || p != pp)
        mapssnext[s] = -1;
    else
        mapssnext[s] = sp;

}


__device__ void applyFixedBC(
        const int p,
        const double2* __restrict__ px,
        double2* __restrict__ pu,
        double2* __restrict__ pf,
        const double2 vfix,
        const double bcconst) {

    const double eps = 1.e-12;
    double dp = dot(px[p], vfix);

    if (fabs(dp - bcconst) < eps) {
        pu[p] = project(pu[p], vfix);
        pf[p] = project(pf[p], vfix);
    }

}


__device__ void calcAccel(
        const int p,
        const double2* __restrict__ pf,
        const double* __restrict__ pmass,
        double2* __restrict__ pa) {

    const double fuzz = 1.e-99;
    pa[p] = pf[p] / max(pmass[p], fuzz);

}


__device__ void advPosFull(
        const int p,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pa,
        const double dt,
        double2* __restrict__ px,
        double2* __restrict__ pu) {

    pu[p] = pu0[p] + pa[p] * dt;
    px[p] = px[p] + 0.5 * (pu[p] + pu0[p]) * dt;

}


__device__ void hydroCalcWork(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const double2* __restrict__ sf,
        const double2* __restrict__ sf2,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pu,
        const double2* __restrict__ px,
        const double dt,
        double* __restrict__ zw,
        double* __restrict__ zetot,
	int dss4[CHUNK_SIZE],
	double ctemp[CHUNK_SIZE]) {

    // Compute the work done by finding, for each element/node pair
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    double sd1 = dot( (sf[s] + sf2[s]), (pu0[p1] + pu[p1]));
    double sd2 = dot(-(sf[s] + sf2[s]), (pu0[p2] + pu[p2]));
    double dwork = -0.5 * dt * (sd1 * px[p1].x + sd2 * px[p2].x);

    ctemp[s0] = dwork;
    double etot = zetot[z];
    __syncthreads();

    double dwtot = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        dwtot += ctemp[sn];
    }
    zetot[z] = etot + dwtot;
    zw[z] = dwtot;

}


__device__ void hydroCalcWorkRate(
        const int z,
        const double* __restrict__ zvol0,
        const double* __restrict__ zvol,
        const double* __restrict__ zw,
        const double* __restrict__ zp,
        const double dt,
        double* __restrict__ zwrate) {

    double dvol = zvol[z] - zvol0[z];
    zwrate[z] = (zw[z] + zp[z] * dvol) / dt;

}


__device__ void hydroCalcEnergy(
        const int z,
        const double* __restrict__ zetot,
        const double* __restrict__ zm,
        double* __restrict__ ze) {

    const double fuzz = 1.e-99;
    ze[z] = zetot[z] / (zm[z] + fuzz);

}


__device__ void hydroCalcDtCourant(
        const int z,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        double& dtz,
        int& idtz) {

    const double fuzz = 1.e-99;
    double cdu = max(zdu[z], max(zss[z], fuzz));
    double dtzcour = zdl[z] * hcfl / cdu;
    dtz = dtzcour;
    idtz = z << 1;

}


__device__ void hydroCalcDtVolume(
        const int z,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
        double& dtz,
        int& idtz) {

    double zdvov = abs((zvol[z] - zvol0[z]) / zvol0[z]);
    double dtzvol = dtlast * hcflv / zdvov;

    if (dtzvol < dtz) {
        dtz = dtzvol;
        idtz = (z << 1) | 1;
    }

}


__device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(min(val,
                __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ void hydroFindMinDt(
        const int z,
        const int z0,
        const int zlength,
        const double dtz,
        const int idtz,
	double ctemp[CHUNK_SIZE],
	double2 ctemp2[CHUNK_SIZE]) {

    int* ctempi = (int*) ctemp2;

    ctemp[z0] = dtz;
    ctempi[z0] = idtz;
    __syncthreads();

    int len = zlength;
    int half = len >> 1;
    while (z0 < half) {
        len = half + (len & 1);
        if (ctemp[z0+len] < ctemp[z0]) {
            ctemp[z0]  = ctemp[z0+len];
            ctempi[z0] = ctempi[z0+len];
        }
        __syncthreads();
        half = len >> 1;
    }
    if (z0 == 0 && ctemp[0] < dtnext) {
        atomicMin(&dtnext, ctemp[0]);
        // This line isn't 100% thread-safe, but since it is only for
        // a debugging aid, I'm not going to worry about it.
        if (dtnext == ctemp[0]) idtnext = ctempi[0];
    }
}


__device__ void hydroCalcDt(
        const int z,
        const int z0,
        const int zlength,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
	double ctemp[CHUNK_SIZE],
	double2 ctemp2[CHUNK_SIZE]) {

    double dtz;
    int idtz;
    hydroCalcDtCourant(z, zdu, zss, zdl, dtz, idtz);
    hydroCalcDtVolume(z, zvol, zvol0, dt, dtz, idtz);
    hydroFindMinDt(z, z0, zlength, dtz, idtz, ctemp, ctemp2);

}


__global__ void gpuMain1()
{
  const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    double dth = 0.5 * dt;

    // save off point variable values from previous cycle
    pu0[p] = pu[p];

    // ===== Predictor step =====
    // 1. advance mesh to center of time step
    advPosHalf(p, px, pu0, dth, pxp);

}


__global__ void gpuMain2a_zb(){
  // iterate over the zones of the mesh
  constexpr int sides_per_zone = 4;
  constexpr int zone_offset = 0; // first of 4-sided zones
  constexpr int side_offset = 0; // first side of 4-sided zones

  const int block_size = blockDim.x;
  const auto global_tid = block_size * blockIdx.x + threadIdx.x;
  const auto local_tid = threadIdx.x;

  const auto first_zone_in_block = blockDim.x * blockIdx.x + zone_offset;
  const auto last_zone_in_block = min(numz, first_zone_in_block + blockDim.x);
  const auto no_zones_in_block = last_zone_in_block - first_zone_in_block;

  const auto no_sides_in_block = no_zones_in_block * sides_per_zone;
  const auto first_side_in_block = sides_per_zone * blockDim.x * blockIdx.x + side_offset;
  // const auto last_side_in_block = first_side_in_block + no_sides_in_block;

  __shared__ double2 sh_p1x[sides_per_zone * CHUNK_SIZE];
  __shared__ double2 sh_workspace[sides_per_zone * CHUNK_SIZE];
  { // load point coordinates for first point p1 of all sides in block into shared memory
    for(auto i = local_tid; i < no_sides_in_block; i += block_size){
      auto p1 = mapsp1[first_side_in_block + i];
      sh_p1x[i] = pxp[p1];
    }
    __syncthreads();
  }

  auto z = global_tid;
  // if (z >= numz){ return; }

  // save off zone variable values from previous cycle
  if(z < last_zone_in_block){
    zvol0[z] = zvol[z];
  }

  if(z < last_zone_in_block){ // TODO: optimize calcZoneCtrs_zb, and handle bounds checking there.
    calcZoneCtrs_zb(z, local_tid, sh_p1x, zxp);
  }

  if(z < last_zone_in_block){ // TODO: optimize calcZoneCtrs_zb, and handle bounds checking there.
    fusedZoneSideUpdates_zb(z, local_tid, first_side_in_block, sh_p1x, sh_workspace,
			    znump, pxp, zxp, zm, smf, zdl, ssurf,
			    sareap, svolp, zareap, zvolp,
			    zrp, cmaswt);
  }

  if(z < last_zone_in_block){ // TODO: check for optimization potential
    pgasCalcStateAtHalf(z, zr, zvolp, zvol0, ze, zwrate, zm, dt, zp, zss);
  }
}


__global__ void gpuMain2()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // save off zone variable values from previous cycle
    zvol0[z] = zvol[z];

    // 1a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, pxp, zxp, dss4, ctemp2);
    meshCalcCharLen(s, s0, s3, z, p1, p2, znump, pxp, zxp, zdl, dss4, ctemp);

    ssurf[s] = rotateCCW(0.5 * (pxp[p1] + pxp[p2]) - zxp[z]);

    calcSideVols(s, z, p1, p2, pxp, zxp, sareap, svolp);
    calcZoneVols(s, s0, z, sareap, svolp, zareap, zvolp);

    // 2. compute corner masses
    hydroCalcRho(z, zm, zvolp, zrp);
    calcCrnrMass(s, s3, z, zrp, zareap, smf, cmaswt);
    
    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    if (s3 > s) pgasCalcStateAtHalf(z, zr, zvolp, zvol0, ze, zwrate,
            zm, dt, zp, zss);
    __syncthreads();

    // 4. compute forces
    pgasCalcForce(s, z, zp, ssurf, sfp);
    ttsCalcForce(s, z, zareap, zrp, zss, sareap, smf, ssurf, sft);
    qcsCalcForce(s, s0, s3, s4, z, p1, p2, dss3, dss4, ctemp, ctemp2);

}
__global__ void gpuMain2a()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // save off zone variable values from previous cycle
    zvol0[z] = zvol[z];

    // 1a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, pxp, zxp, dss4, ctemp2);
    meshCalcCharLen(s, s0, s3, z, p1, p2, znump, pxp, zxp, zdl, dss4, ctemp);

    ssurf[s] = rotateCCW(0.5 * (pxp[p1] + pxp[p2]) - zxp[z]);

    calcSideVols(s, z, p1, p2, pxp, zxp, sareap, svolp);
    calcZoneVols(s, s0, z, sareap, svolp, zareap, zvolp);

    // 2. compute corner masses
    hydroCalcRho(z, zm, zvolp, zrp);
    calcCrnrMass(s, s3, z, zrp, zareap, smf, cmaswt);
    
    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    if (s3 > s) pgasCalcStateAtHalf(z, zr, zvolp, zvol0, ze, zwrate,
            zm, dt, zp, zss);
}

__global__ void gpuMain2b()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // 4. compute forces
    pgasCalcForce(s, z, zp, ssurf, sfp);
    ttsCalcForce(s, z, zareap, zrp, zss, sareap, smf, ssurf, sft);
    qcsCalcForce(s, s0, s3, s4, z, p1, p2, dss3, dss4, ctemp, ctemp2);
}

__global__ void gpuMain2b12()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int z  = mapsz[s];

    // 4. compute forces
    pgasCalcForce(s, z, zp, ssurf, sfp);
    ttsCalcForce(s, z, zareap, zrp, zss, sareap, smf, ssurf, sft);
}

__global__ void gpuMain2b3()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // 4. compute forces
    // inlined version of:
    // qcsCalcForce(s, s0, s3, s4, z, p1, p2, dss3, dss4, ctemp, ctemp2);
        // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv(s, s0, s3, z, p1, p2,dss4, ctemp2);

    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce(s, s3, z, p1, p2);

    // [5] Compute the Q forces
    qcsSetForce(s, s4, p1, p2);

    ctemp2[s0] = sfp[s] + sft[s] + sfq[s];
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff(s, s0, p1, p2, z, dss4, ctemp);
}


__global__ void gpuMain2b3a()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // 4. compute forces
    // inlined version of:
    // qcsCalcForce(s, s0, s3, s4, z, p1, p2, dss3, dss4, ctemp, ctemp2);
        // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv(s, s0, s3, z, p1, p2,dss4, ctemp2);
}


__global__ void gpuMain2b3b()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce(s, s3, z, p1, p2);

    // [5] Compute the Q forces
    qcsSetForce(s, s4, p1, p2);

    ctemp2[s0] = sfp[s] + sft[s] + sfq[s];
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff(s, s0, p1, p2, z, dss4, ctemp);
}

// If we use MPI, then we need to sum corner masses and forces to points locally first,
// then sum the values of the points across MPI ranks for points on the boundaries
// between ranks, and then invoke gpuMain3.
// If we don't use MPI, then the summing of corner masses and forces to points
// is done as the first step in gpuMain3 instead, to reduce the number of kernel
// invocations.
__device__ void localReduceToPoints(
        const int p,
        const double* __restrict__ cmaswt,
        double* __restrict__ pmaswt,
        const double2* __restrict__ cftot,
        double2* __restrict__ pf)
{
    double cmaswt_sum = 0.;
    double2 cftot_sum = make_double2(0., 0.);
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        cmaswt_sum += cmaswt[s];
	cftot_sum += cftot[s];
    }
    pmaswt[p] = cmaswt_sum;
    pf[p] = cftot_sum;
}


#ifdef USE_MPI
__global__ void localReduceToPoints()
{
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    // sum corner masses, forces to points
    localReduceToPoints(p, cmaswt, pmaswt, cftot, pf);
}
#endif

__global__ void gpuMain3(bool doLocalReduceToPoints)
{
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    if(doLocalReduceToPoints){
      // sum corner masses, forces to points
      localReduceToPoints(p, cmaswt, pmaswt, cftot, pf);
    }

    // 4a. apply boundary conditions
    for (int bc = 0; bc < numbcx; ++bc)
        applyFixedBC(p, pxp, pu0, pf, vfixx, bcx[bc]);
    for (int bc = 0; bc < numbcy; ++bc)
        applyFixedBC(p, pxp, pu0, pf, vfixy, bcy[bc]);

    // 5. compute accelerations
    calcAccel(p, pf, pmaswt, pap);

    // ===== Corrector step =====
    // 6. advance mesh to end of time step
    advPosFull(p, pu0, pap, dt, px, pu);

}


__global__ void gpuMain4()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // 6a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, px, zx, dss4, ctemp2);
    calcSideVols(s, z, p1, p2, px, zx, sarea, svol);
    calcZoneVols(s, s0, z, sarea, svol, zarea, zvol);

    // 7. compute work
    hydroCalcWork(s, s0, s3, z, p1, p2, sfp, sfq, pu0, pu, pxp, dt,
		  zw, zetot, dss4, ctemp);

}


__global__ void gpuMain5()
{
    const int z = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (z >= numz) return;

    const int z0 = threadIdx.x;
    const int zlength = min((int)CHUNK_SIZE, (int)(numz - blockIdx.x * CHUNK_SIZE));

    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];

    // 7. compute work
    hydroCalcWorkRate(z, zvol0, zvol, zw, zp, dt, zwrate);

    // 8. update state variables
    hydroCalcEnergy(z, zetot, zm, ze);
    hydroCalcRho(z, zm, zvol, zr);

    // 9.  compute timestep for next cycle
    hydroCalcDt(z, z0, zlength, zdu, zss, zdl, zvol, zvol0, dt,
		ctemp, ctemp2);

}


void meshCheckBadSides() {

    int numsbadH;
#ifdef USE_JIT
    CHKERR(hipMemcpy(&numsbadH, numsbadD, sizeof(int), hipMemcpyDeviceToHost));
#else
    CHKERR(hipMemcpyFromSymbol(&numsbadH, HIP_SYMBOL(numsbad), sizeof(int)));
#endif
    // if there were negative side volumes, error exit
    if (numsbadH > 0) {
        cerr << "Error: " << numsbadH << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


void computeChunks(
        const int nums,
        const int numz,
        const int* mapsz,
        const int chunksize,
        int& numsch,
        int*& schsfirst,
        int*& schslast,
        int*& schzfirst,
        int*& schzlast) {

    int* stemp1 = Memory::alloc<int>(nums/3+1);
    int* stemp2 = Memory::alloc<int>(nums/3+1);
    int* ztemp1 = Memory::alloc<int>(nums/3+1);
    int* ztemp2 = Memory::alloc<int>(nums/3+1);

    int nsch = 0;
    int s1;
    int s2 = 0;
    while (s2 < nums) {
        s1 = s2;
        s2 = min(s2 + chunksize, nums);
        if (s2 < nums) {
            while (mapsz[s2] == mapsz[s2-1]) --s2;
        }
        stemp1[nsch] = s1;
        stemp2[nsch] = s2;
        ztemp1[nsch] = mapsz[s1];
        ztemp2[nsch] = (s2 == nums ? numz : mapsz[s2]);
        ++nsch;
    }

    numsch = nsch;
    schsfirst = Memory::alloc<int>(numsch);
    schslast  = Memory::alloc<int>(numsch);
    schzfirst = Memory::alloc<int>(numsch);
    schzlast  = Memory::alloc<int>(numsch);
    copy(stemp1, stemp1 + numsch, schsfirst);
    copy(stemp2, stemp2 + numsch, schslast);
    copy(ztemp1, ztemp1 + numsch, schzfirst);
    copy(ztemp2, ztemp2 + numsch, schzlast);

    Memory::free(stemp1);
    Memory::free(stemp2);
    Memory::free(ztemp1);
    Memory::free(ztemp2);

}


void hydroInit(
        const int numpH,
        const int numzH,
        const int numsH,
        const int numcH,
        const int numeH,
        const double pgammaH,
        const double pssminH,
        const double talfaH,
        const double tssminH,
        const double qgammaH,
        const double q1H,
        const double q2H,
        const double hcflH,
        const double hcflvH,
        const int numbcxH,
        const double* bcxH,
        const int numbcyH,
        const double* bcyH,
        const double2* pxH,
        const double2* puH,
        const double* zmH,
        const double* zrH,
        const double* zvolH,
        const double* zeH,
        const double* zetotH,
        const double* zwrateH,
        const double* smfH,
        const int* mapsp1H,
        const int* mapsp2H,
        const int* mapszH,
        const int* mapzsH,
        const int* mapss4H,
        const int* mapseH,
        const int* znumpH) {

    printf("Running Hydro on device...\n");

    // temporary globals used while developing zone-based kernels. TODO: remove
    numz_zb = numzH;
    nums_zb = numsH;
    
    computeChunks(numsH, numzH, mapszH, CHUNK_SIZE, numschH,
            schsfirstH, schslastH, schzfirstH, schzlastH);
    numpchH = (numpH+CHUNK_SIZE-1) / CHUNK_SIZE;
    numzchH = (numzH+CHUNK_SIZE-1) / CHUNK_SIZE;

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(nump), &numpH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numz), &numzH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(nums), &numsH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pgamma), &pgammaH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pssmin), &pssminH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(talfa), &talfaH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(tssmin), &tssminH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(qgamma), &qgammaH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(q1), &q1H, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(q2), &q2H, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(hcfl), &hcflH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(hcflv), &hcflvH, sizeof(double)));

    const double2 vfixxH = make_double2(1., 0.);
    const double2 vfixyH = make_double2(0., 1.);
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(vfixx), &vfixxH, sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(vfixy), &vfixyH, sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numbcx), &numbcxH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numbcy), &numbcyH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(bcx), bcxH, numbcxH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(bcy), bcyH, numbcyH*sizeof(double)));

    
    CHKERR(hipMalloc(&numsbadD, sizeof(int)));
    CHKERR(hipMalloc(&schsfirstD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schslastD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schzfirstD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schzlastD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&mapsp1D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapsp2D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapszD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapzsD, numzH*sizeof(int)));
    CHKERR(hipMalloc(&mapss4D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&znumpD, numzH*sizeof(int)));

    CHKERR(hipMalloc(&pxD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&pxpD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&zxD, numzH*sizeof(double2)));
    CHKERR(hipMalloc(&zxpD, numzH*sizeof(double2)));
    CHKERR(hipMalloc(&puD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&pu0D, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&papD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&ssurfD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&zmD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zrD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zrpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&sareaD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&svolD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&zareaD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvolD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvol0D, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zdlD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zduD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zeD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zetot0D, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zetotD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zwD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zwrateD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zssD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&smfD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&careapD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&sareapD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&svolpD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&zareapD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvolpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&cmaswtD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&pmaswtD, numpH*sizeof(double)));
    CHKERR(hipMalloc(&sfpD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&sftD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&sfqD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&cftotD, numcH*sizeof(double2)));
    CHKERR(hipMalloc(&pfD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&cevolD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cduD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cdivD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&zucD, numzH*sizeof(double2)));
    CHKERR(hipMalloc(&crmuD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cqeD, 2*numcH*sizeof(double2)));
    CHKERR(hipMalloc(&ccosD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cwD, numcH*sizeof(double)));

    CHKERR(hipMalloc(&mapspkeyD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapspvalD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mappsfirstD, numpH*sizeof(int)));
    CHKERR(hipMalloc(&mapssnextD, numsH*sizeof(int)));

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(schsfirst), &schsfirstD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(schslast), &schslastD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsp1), &mapsp1D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsp2), &mapsp2D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsz), &mapszD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapzs), &mapzsD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapss4), &mapss4D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mappsfirst), &mappsfirstD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapssnext), &mapssnextD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(znump), &znumpD, sizeof(void*)));

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(px), &pxD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pxp), &pxpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zx), &zxD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zxp), &zxpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pu), &puD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pu0), &pu0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pap), &papD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ssurf), &ssurfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zm), &zmD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zr), &zrD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zrp), &zrpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sarea), &sareaD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(svol), &svolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zarea), &zareaD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol), &zvolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol0), &zvol0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdl), &zdlD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdu), &zduD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ze), &zeD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zetot), &zetotD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zw), &zwD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zwrate), &zwrateD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zp), &zpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zss), &zssD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(smf), &smfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(careap), &careapD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sareap), &sareapD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(svolp), &svolpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zareap), &zareapD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvolp), &zvolpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cmaswt), &cmaswtD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pmaswt), &pmaswtD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sfp), &sfpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sft), &sftD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sfq), &sfqD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cftot), &cftotD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pf), &pfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cevol), &cevolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cdu), &cduD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cdiv), &cdivD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zuc), &zucD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cqe), &cqeD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ccos), &ccosD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cw), &cwD, sizeof(void*)));

    CHKERR(hipMemcpy(schsfirstD, schsfirstH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schslastD, schslastH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schzfirstD, schzfirstH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schzlastD, schzlastH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapsp1D, mapsp1H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapsp2D, mapsp2H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapszD, mapszH, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapzsD, mapzsH, numzH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapss4D, mapss4H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(znumpD, znumpH, numzH*sizeof(int), hipMemcpyHostToDevice));

    CHKERR(hipMemcpy(zmD, zmH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(smfD, smfH, numsH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(pxD, pxH, numpH*sizeof(double2), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(puD, puH, numpH*sizeof(double2), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zrD, zrH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zvolD, zvolH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zeD, zeH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zetotD, zetotH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zwrateD, zwrateH, numzH*sizeof(double), hipMemcpyHostToDevice));

    // initialize temporary data structures used in the implementation of zone-based version
    // of gpuMain2. TODO: remove when done.

    CHKERR(hipMalloc(&zvol_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol_zb), &zvol_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zvol_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol_cp), &zvol_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zvol0_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol0_zb), &zvol0_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zvol0_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol0_cp), &zvol0_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zdl_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdl_zb), &zdl_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zdl_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdl_cp), &zdl_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zp_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zp_zb), &zp_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zp_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zp_cp), &zp_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zss_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zss_zb), &zss_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zss_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zss_cp), &zss_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&cmaswt_zbD, numsH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cmaswt_zb), &cmaswt_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&cmaswt_cpD, numsH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cmaswt_cp), &cmaswt_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zrp_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zrp_zb), &zrp_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zrp_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zrp_cp), &zrp_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zvolp_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvolp_zb), &zvolp_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zvolp_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvolp_cp), &zvolp_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zareap_zbD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zareap_zb), &zareap_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zareap_cpD, numzH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zareap_cp), &zareap_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&svolp_zbD, numsH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(svolp_zb), &svolp_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&svolp_cpD, numsH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(svolp_cp), &svolp_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&sareap_zbD, numsH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sareap_zb), &sareap_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&sareap_cpD, numsH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sareap_cp), &sareap_cpD, sizeof(void*)));

    CHKERR(hipMalloc(&zxp_zbD, numzH*sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zxp_zb), &zxp_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&zxp_cpD, numzH*sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zxp_cp), &zxp_cpD, sizeof(void*)));
    
    CHKERR(hipMalloc(&ssurf_zbD, numsH*sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ssurf_zb), &ssurf_zbD, sizeof(void*)));
    CHKERR(hipMalloc(&ssurf_cpD, numsH*sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ssurf_cp), &ssurf_cpD, sizeof(void*)));
    
    // end of initialization of temporary data structures.
    
    thrust::device_ptr<int> mapsp1T(mapsp1D);
    thrust::device_ptr<int> mapspkeyT(mapspkeyD);
    thrust::device_ptr<int> mapspvalT(mapspvalD);

    thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
    thrust::sequence(mapspvalT, mapspvalT + numsH);
    thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);

    int gridSize = (numsH+CHUNK_SIZE-1) / CHUNK_SIZE;
    int chunkSize = CHUNK_SIZE;
    hipLaunchKernelGGL((gpuInvMap), dim3(gridSize), dim3(chunkSize), 0, 0, mapspkeyD, mapspvalD,
		       mappsfirstD, mapssnextD);

    int zero = 0;
#ifndef USE_JIT
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numsbad), &zero, sizeof(int)));
#endif

#ifdef USE_JIT
    CHKERR(hipMemcpy(numsbadD, &zero, sizeof(int), hipMemcpyHostToDevice));

    replacement_t replacements {
      { "${CHUNK_SIZE}", jit_string(CHUNK_SIZE) },
      { "${careap}", jit_string(careapD) },
      { "${ccos}", jit_string(ccosD) },
      { "${cdiv}", jit_string(cdivD) },
      { "${cdu}", jit_string(cduD) },
      { "${cevol}", jit_string(cevolD) },
      { "${cftot}", jit_string(cftotD) },
      { "${cmaswt}", jit_string(cmaswtD) },
      { "${cqe}", jit_string(cqeD) },
      { "${cw}", jit_string(cwD) },
      { "${mapsp1}", jit_string(mapsp1D) },
      { "${mapsp2}", jit_string(mapsp2D) },
      { "${mapss4}", jit_string(mapss4D) },
      { "${mapsz}", jit_string(mapszD) },
      { "${nump}", jit_string(numpH) },
      { "${numsbad}", jit_string(numsbadD) },
      { "${pgamma}", jit_string(pgammaH) },
      { "${pssmin}", jit_string(pssminH) },
      { "${pu0}", jit_string(pu0D) },
      { "${pu}", jit_string(puD) },
      //      { "${px0}", jit_string(px0D) }, // px0 has been removed. TODO: remove from JIT code too.
      { "${pxp}", jit_string(pxpD) },
      { "${px}", jit_string(pxD) },
      { "${q1}", jit_string(q1H) },
      { "${q2}", jit_string(q2H) },
      { "${qgamma}", jit_string(qgammaH) },
      { "${sareap}", jit_string(sareapD) },
      { "${schsfirst}", jit_string(schsfirstD) },
      { "${schslast}", jit_string(schslastD) },
      { "${sfp}", jit_string(sfpD) },
      { "${sfq}", jit_string(sfqD) },
      { "${sft}", jit_string(sftD) },
      { "${smf}", jit_string(smfD) },
      { "${ssurf}", jit_string(ssurfD) },
      { "${svolp}", jit_string(svolpD) },
      { "${talfa}", jit_string(talfaH) },
      { "${tssmin}", jit_string(tssminH) },
      { "${zareap}", jit_string(zareapD) },
      { "${zdl}", jit_string(zdlD) },
      { "${zdu}", jit_string(zduD) },
      { "${ze}", jit_string(zeD) },
      { "${zm}", jit_string(zmD) },
      { "${znump}", jit_string(znumpD) },
      { "${zp}", jit_string(zpD) },
      { "${zrp}", jit_string(zrpD) },
      { "${zr}", jit_string(zrD) },
      { "${zss}", jit_string(zssD) },
      { "${zuc}", jit_string(zucD) },
      { "${zvol0}", jit_string(zvol0D) },
      { "${zvolp}", jit_string(zvolpD) },
      { "${zvol}", jit_string(zvolD) },
      { "${zwrate}", jit_string(zwrateD) },
      { "${zxp}", jit_string(zxpD) }
      //{ "${}", jit_string() },
    };
    jit = std::unique_ptr<Pajama>(new Pajama("src.jit/kernels.cc", replacements));
    jit->load_kernel("gpuMain1_jit");
    jit->load_kernel("gpuMain2_jit");
#endif // USE_JIT
}

#ifdef USE_MPI
__global__ void copySlavePointDataToMPIBuffers_kernel(double* pmaswt_slave_buffer_D,
						      double2* pf_slave_buffer_D){
  int slave = blockIdx.x * blockDim.x + threadIdx.x;
  if(slave >= numslv) { return; }
  int point = mapslvp[slave];
  pmaswt_slave_buffer_D[slave] = pmaswt[point];
  pf_slave_buffer_D[slave] = pf[point];
}

void copySlavePointDataToMPIBuffers(){
  constexpr int blocksize = 256;
  const int blocks = (numslvH + blocksize - 1) / blocksize;
  hipLaunchKernelGGL(copySlavePointDataToMPIBuffers_kernel, blocks, blocksize, 0, 0,
		     pmaswt_slave_buffer_D, pf_slave_buffer_D);
#ifndef USE_GPU_AWARE_MPI
  CHKERR(hipMemcpy(pmaswt_slave_buffer_H, pmaswt_slave_buffer_D, numslvH * sizeof(double), hipMemcpyDeviceToHost));
  CHKERR(hipMemcpy(pf_slave_buffer_H, pf_slave_buffer_D, numslvH * sizeof(double2), hipMemcpyDeviceToHost));
#endif
}


__global__ void copyMPIBuffersToSlavePointData_kernel(double* pmaswt_slave_buffer_D,
						      double2* pf_slave_buffer_D){
  int slave = blockIdx.x * blockDim.x + threadIdx.x;
  if(slave >= numslv) { return; }
  int point = mapslvp[slave];
  pmaswt[point] = pmaswt_slave_buffer_D[slave];
  pf[point] = pf_slave_buffer_D[slave];
}

void copyMPIBuffersToSlavePointData(){
#ifndef USE_GPU_AWARE_MPI
  CHKERR(hipMemcpy(pmaswt_slave_buffer_D, pmaswt_slave_buffer_H, numslvH * sizeof(double), hipMemcpyHostToDevice));
  CHKERR(hipMemcpy(pf_slave_buffer_D, pf_slave_buffer_H, numslvH * sizeof(double2), hipMemcpyHostToDevice));
#endif
  constexpr int blocksize = 256;
  const int blocks = (numslvH + blocksize - 1) / blocksize;
  hipLaunchKernelGGL(copyMPIBuffersToSlavePointData_kernel, blocks, blocksize, 0, 0,
		     pmaswt_slave_buffer_D, pf_slave_buffer_D);
}


__global__ void reduceToMasterPoints(double* pmaswt_proxy_buffer_D,
					   double2* pf_proxy_buffer_D){
  int proxy = blockIdx.x * blockDim.x + threadIdx.x;
  if(proxy >= numprx) { return; }

  int point = mapprxp[proxy];
  atomicAdd(&pmaswt[point], pmaswt_proxy_buffer_D[proxy]);
  atomicAdd(&pf[point].x, pf_proxy_buffer_D[proxy].x);
  atomicAdd(&pf[point].y, pf_proxy_buffer_D[proxy].y);
}


__global__ void copyPointValuesToProxies(double* pmaswt_proxy_buffer_D,
					 double2* pf_proxy_buffer_D){
  int proxy = blockIdx.x * blockDim.x + threadIdx.x;
  if(proxy >= numprx) { return; }

  int point = mapprxp[proxy];
  pmaswt_proxy_buffer_D[proxy] = pmaswt[point];
  pf_proxy_buffer_D[proxy] = pf[point];
}


void reduceToMasterPointsAndProxies(){
#ifndef USE_GPU_AWARE_MPI
  CHKERR(hipMemcpy(pmaswt_proxy_buffer_D, pmaswt_proxy_buffer_H, numprxH * sizeof(double), hipMemcpyHostToDevice));
  CHKERR(hipMemcpy(pf_proxy_buffer_D, pf_proxy_buffer_H, numprxH * sizeof(double2), hipMemcpyHostToDevice));
#endif

  constexpr int blocksize = 256;
  const int blocks = (numprxH + blocksize - 1) / blocksize;
  hipLaunchKernelGGL(reduceToMasterPoints, blocks, blocksize, 0, 0,
		     pmaswt_proxy_buffer_D, pf_proxy_buffer_D);
  hipLaunchKernelGGL(copyPointValuesToProxies, blocks, blocksize, 0, 0,
		     pmaswt_proxy_buffer_D, pf_proxy_buffer_D);

#ifndef USE_GPU_AWARE_MPI
  CHKERR(hipMemcpy(pmaswt_proxy_buffer_H, pmaswt_proxy_buffer_D, numprxH * sizeof(double), hipMemcpyDeviceToHost));
  CHKERR(hipMemcpy(pf_proxy_buffer_H, pf_proxy_buffer_D, numprxH * sizeof(double2), hipMemcpyDeviceToHost));
#endif
}

void globalReduceToPoints() {
  copySlavePointDataToMPIBuffers();
  parallelGather( numslvpeD, nummstrpeD,
		  mapslvpepeD,  slvpenumprxD,  mapslvpeprx1D,
		  mapmstrpepeD,  mstrpenumslvD,  mapmstrpeslv1D,
		  pmaswt_proxy_buffer, pf_proxy_buffer,
		  pmaswt_slave_buffer, pf_slave_buffer);
  reduceToMasterPointsAndProxies();
  parallelScatter( numslvpeD, nummstrpeD,
		   mapslvpepeD,  slvpenumprxD,  mapslvpeprx1D,
		   mapmstrpepeD,  mstrpenumslvD,  mapmstrpeslv1D,  mapslvpD,
		   pmaswt_proxy_buffer, pf_proxy_buffer,
		   pmaswt_slave_buffer, pf_slave_buffer);
  copyMPIBuffersToSlavePointData();
}
#endif

// temporary functions, used to validate correctness while developing a zone-base
// version of gpuMain2. TODO: remove when done.


template<typename T>
__global__ void copy_kernel(T* target, T* source, int size){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  target[tid] = source[tid];
}

template<typename T>
__global__ void zap_kernel(T* target, int size, T val = T()){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  target[tid] = val;
}

void prepare_zb_mirror_data(){
  int grid_zb = (numz_zb + CHUNK_SIZE -1) / CHUNK_SIZE;
  // hipLaunchKernelGGL(copy_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zvol_zbD, zvolD, numz_zb); // not modifying zvol yet
  hipLaunchKernelGGL(zap_kernel<double2>, grid_zb, CHUNK_SIZE, 0, 0, ssurf_zbD, nums_zb, {6.66e66, 6.66e66});
  hipLaunchKernelGGL(zap_kernel<double2>, grid_zb, CHUNK_SIZE, 0, 0, zxp_zbD, numz_zb, {6.66e66, 6.66e66});
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zvol0_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zdl_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zp_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zss_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, cmaswt_zbD, nums_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zrp_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zvolp_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zareap_zbD, numz_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, svolp_zbD, nums_zb, 6.66e66);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, sareap_zbD, nums_zb, 6.66e66);

  // zap all the checkpoint arrays
  // hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zvol_cpD, numz_zb); // not modifying zvol yet
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zvol0_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double2>, grid_zb, CHUNK_SIZE, 0, 0, zxp_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zdl_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zp_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, cmaswt_cpD, nums_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zrp_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zvolp_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zareap_cpD, numz_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, svolp_cpD, nums_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, sareap_cpD, nums_zb);
  hipLaunchKernelGGL(zap_kernel<double2>, grid_zb, CHUNK_SIZE, 0, 0, ssurf_cpD, nums_zb);
  hipLaunchKernelGGL(zap_kernel<double>, grid_zb, CHUNK_SIZE, 0, 0, zss_cpD, numz_zb);
};

inline __device__ void compare_equal(int tid, double expected, double actual, double eps,
				     int* found_difference, bool print){
  auto diff = actual - expected;
  if(fabs(diff) > eps) {
    *found_difference = 1;
    if(print){
      printf("tid = %d, expected = %g, actual = %g, diff = %g\n",
	     tid, expected, actual, diff);
    }
  }
}

inline __device__ void compare_equal(int tid, double2 expected, double2 actual, double eps,
				     int* found_difference, bool print){
  auto diff = actual - expected;
  if(fabs(diff.x) > eps or fabs(diff.y) > eps) {
    *found_difference = 1;
    if(print){
      printf("tid = %d, expected = (%g, %g), actual = (%g, %g), diff = (%g, %g)\n",
	     tid, expected.x, expected.y, actual.x, actual.y,
	     diff.x, diff.y);
    }
  }
}

template<typename T, typename E>
__global__ void compare_kernel(const T* const expected, const T* const actual,
			       int size, E eps, int* found_difference, bool print){
  // precondition: found_difference is initialized to 0 by the caller
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= size) return;
  compare_equal(tid, expected[tid], actual[tid], eps, found_difference, print);

}

template<typename T, typename E>
void compare_data(const T* const expected, const T* const actual, int size,
		  int* const found_difference_d, E eps, int cycle, const char* name, bool print){
 int grid = (size + CHUNK_SIZE -1) / CHUNK_SIZE;
 hipLaunchKernelGGL(compare_kernel<T>, grid, CHUNK_SIZE, 0, 0,
		    expected, actual, size, eps, found_difference_d, print);
 int found_difference;
 CHKERR(hipMemcpy(&found_difference, found_difference_d, sizeof(int), hipMemcpyDeviceToHost));
 if(found_difference){
   printf("found difference for %s in cycle %d\n", name, cycle);
   exit(1);
 }
}

void validate_zb_mirror_data(){
  bool print = true; // false: only report errors exist; true: print the difference between expected and actual 
  static int cycle = 1; // Pennant counts cycles starting from 1
  int found_difference = 0;
  int* found_difference_d;
  CHKERR(hipMalloc(&found_difference_d, sizeof(int)));
  CHKERR(hipMemcpy(found_difference_d, &found_difference, sizeof(int), hipMemcpyHostToDevice));

  compare_data<double>(zvol0_cpD, zvol0_zbD, numz_zb, found_difference_d, 0.0, cycle, "zvol0_zb", print);
  // zxp differences probably due to computation reordering
  compare_data<double2>(zxp_cpD, zxp_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zxp_zb", print);
  // zdl differences are caused by propagating zxp differences
  compare_data<double>(zdl_cpD, zdl_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zdl_zb", print);
  // ssurf differences are caused by propagating zxp differences
  compare_data<double2>(ssurf_cpD, ssurf_zbD, nums_zb, found_difference_d, 1.e-12, cycle, "ssurf_zb", print);
  compare_data<double>(sareap_cpD, sareap_zbD, nums_zb, found_difference_d, 1.e-12, cycle, "sareap_zb", print);
  compare_data<double>(svolp_cpD, svolp_zbD, nums_zb, found_difference_d, 1.e-12, cycle, "svolp_zb", print);
  compare_data<double>(zareap_cpD, zareap_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zareap_zb", print);
  compare_data<double>(zvolp_cpD, zvolp_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zvolp_zb", print);
  compare_data<double>(zrp_cpD, zrp_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zrp_zb", print);
  compare_data<double>(cmaswt_cpD, cmaswt_zbD, nums_zb, found_difference_d, 1.e-12, cycle, "cmaswt_zb", print);
  compare_data<double>(zp_cpD, zp_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zp_zb", print);
  compare_data<double>(zss_cpD, zss_zbD, numz_zb, found_difference_d, 1.e-12, cycle, "zss_zb", print);
 
  CHKERR(hipFree(found_difference_d));
  ++cycle;
};

// --- end of temporary functions.

void hydroDoCycle(
        const double dtH,
        double& dtnextH,
        int& idtnextH) {
    int gridSizeS, gridSizeP, gridSizeZ, chunkSize;

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(dt), &dtH, sizeof(double)));

    gridSizeS = numschH;
    gridSizeP = numpchH;
    gridSizeZ = numzchH;
    chunkSize = CHUNK_SIZE;

    // TODO: remove after optimizing gpuMain2
    // int grid_zb = (numz_zb + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
#ifdef USE_JIT
    struct {
      double dtH;
    } gpu_args;
    gpu_args.dtH = dtH;
    size_t gpu_args_size = sizeof(gpu_args);
    void* gpu_args_wrapper[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &gpu_args,
    				 HIP_LAUNCH_PARAM_BUFFER_SIZE, &gpu_args_size,
    				 HIP_LAUNCH_PARAM_END };
#endif

#ifdef USE_JIT
    jit->call_preloaded("gpuMain1_jit", gridSizeP, chunkSize, 0, 0, gpu_args_wrapper);
    // prepare_zb_mirror_data(); // TODO: remove after finishing zone-based version of gpuMain2
    jit->call_preloaded("gpuMain2_jit", gridSizeS, chunkSize, 0, 0, gpu_args_wrapper);
#else
    hipLaunchKernelGGL((gpuMain1), dim3(gridSizeP), dim3(chunkSize), 0, 0);
    // prepare_zb_mirror_data(); // TODO: remove after finishing zone-based version of gpuMain2
    hipLaunchKernelGGL((gpuMain2a_zb), dim3(gridSizeS), dim3(chunkSize), 0, 0);
    hipLaunchKernelGGL((gpuMain2b), dim3(gridSizeS), dim3(chunkSize), 0, 0);

    // hipLaunchKernelGGL((gpuMain2b12), dim3(gridSizeS), dim3(chunkSize), 0, 0);
    // hipLaunchKernelGGL((gpuMain2b3a), dim3(gridSizeS), dim3(chunkSize), 0, 0);
    // hipLaunchKernelGGL((gpuMain2b3b), dim3(gridSizeS), dim3(chunkSize), 0, 0);
#endif

    // TODO: remove after finishing zone-based version of gpuMain2
    // hipLaunchKernelGGL((gpuMain2_zb), dim3(grid_zb), dim3(chunkSize), 0, 0);
    // validate_zb_mirror_data();

    meshCheckBadSides();
    bool doLocalReduceToPointInGpuMain3 = true;

#ifdef USE_MPI
    if(Parallel::numpe > 1){
      // local reduction to points needs to be done either way, but if numpe == 1, then
      // we can do it in gpuMain3, which saves a kernel call
      doLocalReduceToPointInGpuMain3 = false;
      hipLaunchKernelGGL((localReduceToPoints), dim3(gridSizeP), dim3(chunkSize), 0, 0);
      globalReduceToPoints();
    }
#endif

    hipLaunchKernelGGL((gpuMain3), dim3(gridSizeP), dim3(chunkSize), 0, 0,
		       doLocalReduceToPointInGpuMain3);

    double bigval = 1.e99;
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(dtnext), &bigval, sizeof(double)));

    hipLaunchKernelGGL((gpuMain4), dim3(gridSizeS), dim3(chunkSize), 0, 0);

    hipLaunchKernelGGL((gpuMain5), dim3(gridSizeZ), dim3(chunkSize), 0, 0);
    meshCheckBadSides();

    CHKERR(hipMemcpyFromSymbol(&dtnextH, HIP_SYMBOL(dtnext), sizeof(double)));
    CHKERR(hipMemcpyFromSymbol(&idtnextH, HIP_SYMBOL(idtnext), sizeof(int)));
}


void hydroGetData(
        double *zareaH,
	double *zetotH,
	double *zvolH,
        const int numpH,
        const int numzH,
        double2* pxH,
        double* zrH,
        double* zeH,
        double* zpH,
	double2* puH) {

    CHKERR(hipMemcpy(zareaH, zareaD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zetotH, zetotD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zvolH, zvolD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(pxH, pxD, numpH*sizeof(double2), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zrH, zrD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zeH, zeD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zpH, zpD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(puH, puD, numpH*sizeof(double2), hipMemcpyDeviceToHost));
}

#ifdef USE_MPI
void hydroInitMPI(const int nummstrpeH,
		  const int numslvpeH,
		  const int numprxH1,
		  const int numslvH1,
		  const int* mapslvpepeH,
		  const int* mapslvpeprx1H,
		  const int* mapprxpH,
		  const int* slvpenumprxH,
		  const int* mapmstrpepeH,
		  const int* mstrpenumslvH,
		  const int* mapmstrpeslv1H,
		  const int* mapslvpH){

  numslvH = numslvH1;
  numprxH = numprxH1;

  CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numslv), &numslvH, sizeof(int)));
  CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numprx), &numprxH, sizeof(int)));

  CHKERR(hipMalloc(&mapslvpD1, numslvH*sizeof(int)));
  CHKERR(hipMalloc(&mapprxpD, numprxH*sizeof(int)));

  mapslvpepeD = Memory::alloc<int>(numslvpeH);
  mapslvpeprx1D = Memory::alloc<int>(numslvpeH);
  slvpenumprxD = Memory::alloc<int>(numslvpeH);
  mapmstrpepeD = Memory::alloc<int>(nummstrpeH);
  mstrpenumslvD=Memory::alloc<int>(nummstrpeH);
  mapmstrpeslv1D=Memory::alloc<int>(nummstrpeH);
  mapslvpD=Memory::alloc<int>(numslvH);

  memcpy(mapslvpepeD, mapslvpepeH, numslvpeH*sizeof(int));
  memcpy(mapslvpeprx1D, mapslvpeprx1H, numslvpeH*sizeof(int));
  memcpy(slvpenumprxD, slvpenumprxH, numslvpeH*sizeof(int));
  memcpy(mapmstrpepeD, mapmstrpepeH, nummstrpeH*sizeof(int));
  memcpy(mstrpenumslvD, mstrpenumslvH, nummstrpeH*sizeof(int));
  memcpy(mapmstrpeslv1D, mapmstrpeslv1H, nummstrpeH*sizeof(int));
  memcpy(mapslvpD,mapslvpH, numslvH*sizeof(int));

  CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapslvp), &mapslvpD1, sizeof(void*)));
  CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapprxp), &mapprxpD, sizeof(void*)));

  nummstrpeD = nummstrpeH;
  numslvpeD = numslvpeH;
  CHKERR(hipMemcpy( mapslvpD1, mapslvpH, numslvH*sizeof(int), hipMemcpyHostToDevice));
  CHKERR(hipMemcpy( mapprxpD, mapprxpH, numprxH*sizeof(int), hipMemcpyHostToDevice));

  if(numprxH) hipMalloc(&pmaswt_proxy_buffer_D, numprxH*sizeof(double));
  if(numslvH) hipMalloc(&pmaswt_slave_buffer_D, numslvH*sizeof(double));
  if(numprxH) hipMalloc(&pf_proxy_buffer_D, numprxH*sizeof(double2));
  if(numslvH) hipMalloc(&pf_slave_buffer_D, numslvH*sizeof(double2));
#ifndef USE_GPU_AWARE_MPI
  if(numprxH) pmaswt_proxy_buffer_H = Memory::alloc<double>(numprxH);
  if(numslvH) pmaswt_slave_buffer_H = Memory::alloc<double>(numslvH);
  if(numprxH) pf_proxy_buffer_H = Memory::alloc<double2>(numprxH);
  if(numslvH) pf_slave_buffer_H = Memory::alloc<double2>(numslvH);

  pmaswt_proxy_buffer = pmaswt_proxy_buffer_H;
  pmaswt_slave_buffer = pmaswt_slave_buffer_H;
  pf_proxy_buffer = pf_proxy_buffer_H;
  pf_slave_buffer = pf_slave_buffer_H;
#else
  pmaswt_proxy_buffer = pmaswt_proxy_buffer_D;
  pmaswt_slave_buffer = pmaswt_slave_buffer_D;
  pf_proxy_buffer = pf_proxy_buffer_D;
  pf_slave_buffer = pf_slave_buffer_D;
#endif
}
#endif

void hydroInitGPU()
{
#ifdef USE_MPI
  // TODO: consider letting slurm handle the pe to device mapping
  int nDevices;
  hipGetDeviceCount(&nDevices);
  using Parallel::mype;
  int device_num = mype % nDevices;
  hipSetDevice(device_num);
#endif
}


void hydroFinalGPU()
{
  // TODO: free resources
}

