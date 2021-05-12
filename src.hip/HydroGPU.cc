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
#include <thrust/scan.h>
#include "scoped_timers.h"

#ifdef USE_MPI
#include "Parallel.hh"
#include "HydroMPI.hh"
#endif

#include "Memory.hh"
#include "Vec2.hh"

#ifdef USE_JIT
#include "pajama.h"
#endif

using namespace std;

const int CHUNK_SIZE = 64;

#ifdef USE_MPI
__constant__ int numslv;
__constant__ int numprx;
__constant__ int* mapprxp;
__constant__ int* mapslvp;
#endif

__constant__ int nump;
__constant__ int numz;
__constant__ int nums;
__constant__ double pgamma, pssmin;
__constant__ double talfa, tssmin;
__constant__ double qgamma, q1, q2;
__constant__ double hcfl, hcflv;
__constant__ double2 vfixx, vfixy;
__constant__ int numbcx, numbcy;
__constant__ double bcx[2], bcy[2];

__constant__ const int* schsfirst;
__constant__ const int* schslast;
__constant__ const int* mapsp1;
__constant__ const int* mapsp2;
__constant__ const int* mapsz;
__constant__ const int* mapss4;
__constant__ const int *mappsfirst, *mapssnext;
__constant__ const int* znump;
__constant__ int* corners_per_point;
__constant__ int* corners_by_point;
__constant__ int* first_corner_of_point;
__constant__ int2* first_corner_and_corner_count;

__constant__ double2 *px, *pxp;
__constant__ double2 *pu, *pu0;
__constant__ const double* zm;
__constant__ double *zr;
__constant__ double *ze, *zetot;
__constant__ double *zwrate;
__constant__ double *zp, *zss;
__constant__ const double* smf;
__constant__ double   *zvolp;
__constant__ double   *zarea, *zvol, *zvol0;
__constant__ double *zdl, *zdu;
__constant__ double *cmaswt, *pmaswt;
__constant__ double2 *sfpq, *cftot, *pf;
__constant__ double* cw;

int numschH, numpchH, numzchH;
int* numsbad_pinned;
int* pinned_control_flag;
int *schsfirstH, *schslastH, *schzfirstH, *schzlastH;
int *schsfirstD, *schslastD, *schzfirstD, *schzlastD;
int *mapsp1D, *mapsp2D, *mapszD, *mapss4D, *znumpD;
int *mapspkeyD, *mapspvalD;
int *mappsfirstD, *mapssnextD;
int *corners_per_pointD, *corners_by_pointD, *first_corner_of_pointD;
int2 *first_corner_and_corner_countD;
double2 *pxD, *pxpD,  *puD, *pu0D, 
  *sfpqD, *cftotD, *pfD, *cqeD;
double *zmD, *zrD, 
      *zareaD, *zvolD, *zvol0D, *zdlD, *zduD,
    *zeD, *zetot0D, *zetotD, *zwrateD,
    *zpD, *zssD, *smfD,    *zvolpD;
double *cmaswtD, *pmaswtD;

struct double_int {
  double d;
  int i;
};
  
double_int *dtnext_D, *dtnext_H;
int* remaining_wg_D;

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

double *pmaswt_pf_proxy_buffer, *pmaswt_pf_slave_buffer;      // copies of either *_D or *_H pointers
double *pmaswt_pf_proxy_buffer_D, *pmaswt_pf_slave_buffer_D;  // pointers used to allocate device memory


#ifndef USE_GPU_AWARE_MPI
double *pmaswt_proxy_buffer_H, *pmaswt_slave_buffer_H;  // pointers used to allocat host memory in case
double2 *pf_proxy_buffer_H, *pf_slave_buffer_H;         // we can't do MPI transfers from/to device memory

double *pmaswt_pf_proxy_buffer_H, *pmaswt_pf_slave_buffer_H;

#endif // USE_GPU_AWARE_MPI
#endif // USE_MPI

#ifdef USE_JIT
std::unique_ptr<Pajama> jit;
#endif

hipEvent_t mainLoopEvent;

int checkCudaError(const hipError_t err, const char* cmd)
{
    if(err) {
        printf("CUDA error in command '%s'\n", cmd); \
        printf("Error message: %s\n", hipGetErrorString(err)); \
    }
    return err;
}

#define CHKERR(cmd) checkCudaError(cmd, #cmd)



__launch_bounds__(256)
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


__device__ void applyFixedBC_opt(
        const int p,
        const double2 px,
        double2 &pu,
        double2 &pf,
        const double2 vfix,
        const double bcconst) {

    const double eps = 1.e-12;
    double dp = dot(px, vfix);

    if (fabs(dp - bcconst) < eps) {
        pu = project(pu, vfix);
        pf = project(pf, vfix);
    }

}




__device__ void hydroCalcWork(
        const int s,
        const int s0,
        const int z,
        const int p1,
        const int p2,
        const double2* __restrict__ sf,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pu,
        const double2* __restrict__ px,
        const double dt,
        double &zwz,
        double* __restrict__ zetot,
	int dss4[CHUNK_SIZE],
	double ctemp[CHUNK_SIZE]) {

    // Compute the work done by finding, for each element/node pair
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const double sd1 = dot( sf[s], (pu0[p1] + pu[p1]));
    const double sd2 = dot(-sf[s], (pu0[p2] + pu[p2]));
    const double dwork = -0.5 * dt * (sd1 * px[p1].x + sd2 * px[p2].x);

    ctemp[s0] = dwork;
    const double etot = zetot[z];
    __syncthreads();

    double dwtot = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        dwtot += ctemp[sn];
    }
    zetot[z] = etot + dwtot;
    zwz = dwtot;



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
	double2 ctemp2[CHUNK_SIZE],
	double_int* dtnext,
	double_int* dtnext_H,
	int* remaining_wg,
	int* pinned_control_flag) {

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
    if (z0 == 0 && ctemp[0] < dtnext->d) {
      atomicMin(&(dtnext->d), ctemp[0]);
        // This line isn't 100% thread-safe, but since it is only for
        // a debugging aid, I'm not going to worry about it.
        if (dtnext->d == ctemp[0]) dtnext->i = ctempi[0];
    }
    if(threadIdx.x == 0){
      int old = atomicSub(remaining_wg, 1);
      bool this_wg_is_last = (old == 1);
      if(this_wg_is_last){
	// force reloading of dtnext->d from L2 into register
	atomicMin(&(dtnext->d), ctemp[0]);
	// write values to pinned host memory
	dtnext_H->d = dtnext->d;
	dtnext_H->i = dtnext->i;
#ifdef __CUDACC__
	*pinned_control_flag = 1;
	__threadfence_system();
#else
	int one = 1;
	__atomic_store(pinned_control_flag, &one, __ATOMIC_RELEASE);
#endif
      }
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
	double2 ctemp2[CHUNK_SIZE],
	double_int* dtnext,
	double_int* dtnext_H,
	int* remaining_wg,
	int* pinned_control_flag) {

    double dtz;
    int idtz;
    hydroCalcDtCourant(z, zdu, zss, zdl, dtz, idtz);
    hydroCalcDtVolume(z, zvol, zvol0, dtlast, dtz, idtz);
    hydroFindMinDt(z, z0, zlength, dtz, idtz, ctemp, ctemp2, dtnext, dtnext_H, remaining_wg, pinned_control_flag);

}


__launch_bounds__(256)
__global__ void calcCornersPerPoint(int* corners_per_point)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= nump) { return; }

  int count = 0;
  for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
    ++count;
  }
  corners_per_point[p] = count;
}

__device__ void quadratic_sort(int* array, int first, int last){
  for(int i = first; i != last; ++i){
    for(int j = i + 1; j != last; ++j){
      if(array[j] < array[i]){
	int tmp = array[j];
	array[j] = array[i];
	array[i] = tmp;
      }
    }
  }
}


__launch_bounds__(256)
__global__ void storeCornersByPoint(int* first_corner_of_point, int* corners_by_point,
				    int* corners_per_point, int2* first_corner_and_corner_count)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= nump) { return; }

  int first = first_corner_of_point[p];
  int c = first;
  for(int s = mappsfirst[p]; s >= 0; s = mapssnext[s], ++c){
    corners_by_point[c] = s;
  }
  first_corner_and_corner_count[p].x = first;
  first_corner_and_corner_count[p].y = corners_per_point[p];
}


__launch_bounds__(256)
__global__ void gpuMain1(double dt)
{
  const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    double dth = 0.5 * dt;

    // save off point variable values from previous cycle
    pu0[p] = pu[p];
    pxp[p] = px[p] + pu0[p] * dth;

    // ===== Predictor step =====
    // 1. advance mesh to center of time step

}



__device__ void calcZoneCtrs_SideVols_ZoneVols(
        const int s,
        const int s0,
        const double2 pxp1,
        const double2 pxp2,
        double2& __restrict__ zx,
        double&  sarea,
        double& zarea,
        double& zvol,
        int dss4[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE],
        double ctemp1[CHUNK_SIZE],
        int* __restrict__ numsbad_pinned,
	//const int s3, 
	const int z,
        const int* __restrict__ znump,
        double* __restrict__ zdl)
{
     ctemp2[s0] = pxp1;
    __syncthreads();

    double2 zxtot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zxtot += ctemp2[sn];
        zct += 1.;
    }
    zx = zxtot / zct;

    const double third = 1. / 3.;
    sarea = 0.5 * cross(pxp2 - pxp1,  zx - pxp1);
    const double sv = third * sarea * (pxp1.x + pxp2.x + zx.x);
    if (sv <= 0.) { atomicAdd(numsbad_pinned, 1); }

    ctemp[s0] = sv;
    ctemp1[s0] = sarea;
    __syncthreads();
    double zvtot = ctemp[s0];
    double zatot = ctemp1[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zvtot += ctemp[sn];
        zatot += ctemp1[sn];
    }

    zarea = zatot;
    zvol = zvtot;
    const double base = length(pxp2 - pxp1);
    const double fac = (znump[z] == 3 ? 3. : 4.);
    const double sdl = fac * sarea / base;

    ctemp[s0] = sdl;
    __syncthreads();
    double sdlmin = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        sdlmin = min(sdlmin, ctemp[sn]);
    }
    zdl[z] = sdlmin;
}


__device__ void pgasCalcStateAtHalf(
    const int z,
    const double rx,
    const double zvolp,
    const double* __restrict__ zvol0,
    const double* __restrict__ ze,
    const double* __restrict__ zwrate,
    const double  zm,
    const double dt,
    double &zp,
    double &zss)
{
    double zper;

    const double gm1 = pgamma - 1.;
    const double ss2 = max(pssmin * pssmin, 1.e-99);

    const double ex = max(ze[z], 0.0);
    const double csqd = max(ss2, gm1 * ex * pgamma);
    const double z_t = gm1 * rx * ex;
    zper = gm1 * rx;
    zss = sqrt(csqd);

    const double dth = 0.5 * dt;
    const double zminv = 1. / zm ;
    const double dv = (zvolp - zvol0[z]) * zminv;
    const double bulk = rx * rx * csqd;
    const double denom = 1. + 0.5 * zper * dv;
    const double src = zwrate[z] * dth * zminv;
    zp = z_t + (zper * src - bulk * dv) / denom;
}

static __device__ void ttsCalcForce(
        const int s,
        const int z,
        const double zareap,
        const double zrp,
        const double zssz,
        const double sareap,
        const double* __restrict__ smf,
        const double2 ssurf,
        double2 &sft) {

    const double svfacinv = zareap / sareap;
    const double srho = zrp * smf[s] * svfacinv;
    double sstmp = max(zssz, tssmin);
    sstmp = talfa * sstmp * sstmp;
    const double sdp = sstmp * (srho - zrp);
    sft = -sdp * ssurf;

}

__device__ void qcsSetCornerDiv(
        const int s,
        const int s0,
	const int s4,
        const int s04,
        const int z,
        const int p1,
        const int p2,
        const double2 pxpp0,
        const double2 pxpp1,
        const double2 pxpp2,
        const double2 pup0,
        const double2 pup1,
        const double2 pup2,
        const double2  zxp,
	const double zrp,
        const double zss1,
	double2& sfq,
        int dss4[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE],
        double sh_ccos[CHUNK_SIZE],
	double ctemp[CHUNK_SIZE],
        double2 ctemp1[CHUNK_SIZE*2]) {

    // [1] Compute a zone-centered velocity
    ctemp2[s0] = pup1;
    __syncthreads();

    double2 zutot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zutot += ctemp2[sn];
        zct += 1.;
    }
    const double2 zuc = zutot / zct;

    // [2] Divergence at the corner
    // Associated zone, corner, point
    const double2 up0 = pup1;
    const double2 xp0 = pxpp1;
    const double2 up1 = 0.5 * (pup1 + pup2);
    const double2 xp1 = 0.5 * (pxpp1 + pxpp2);
    const double2 up2 = zuc;
    const double2 xp2 = zxp;
    const double2 up3 = 0.5 * (pup0 + pup1);
    const double2 xp3 = 0.5 * (pxpp0 + pxpp1);
    // position, velocity diffs along diagonals
    const double2 up2m0 = up2 - up0;
    const double2 xp2m0 = xp2 - xp0;
    const double2 up3m1 = up3 - up1;
    const double2 xp3m1 = xp3 - xp1;

    // average corner-centered velocity
    const double2 duav = 0.25 * (up0 + up1 + up2 + up3);

    // compute cosine angle
    const double2 v1 = xp1 - xp0;
    const double2 v2 = xp3 - xp0;
    const double de1 = length(v1);
    const double de2 = length(v2);
    const double minelen = 2.0 * min(de1, de2);

    // compute 2d cartesian volume of corner
    const double careap = 0.5 * cross(xp2m0, xp3m1);

    // compute velocity divergence of corner

    const double cdiv = (cross(up2m0, xp3m1) - cross(up3m1, xp2m0)) / (2.0 * careap);

    // compute delta velocity
    const double dv1 = length2(up2m0 - up3m1);
    const double dv2 = length2(up2m0 + up3m1);
    const double du = sqrt(max(dv1, dv2));
    const double cdu   = (cdiv < 0.0 ? du   : 0.);

    // compute evolution factor
    const double2 dxx1 = 0.5 * (xp2m0 - xp3m1);
    const double2 dxx2 = 0.5 * (xp2m0 + xp3m1);
    const double dx1 = length(dxx1);
    const double dx2 = length(dxx2);

    const double test1 = abs(dot(dxx1, duav) * dx2);
    const double test2 = abs(dot(dxx2, duav) * dx1);
    const double num = (test1 > test2 ? dx1 : dx2);
    const double den = (test1 > test2 ? dx2 : dx1);
    const double r = num / den;
    double evol = sqrt(4.0 * careap * r);
    evol = min(evol, 2.0 * minelen);
    const double cevol = (cdiv < 0.0 ? evol : 0.);



    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the rmu (real Kurapatenko viscous scalar)
    // Kurapatenko form of the viscosity
    const double ztmp2 = q2 * 0.25 * gammap1 * cdu;
    const double ztmp1 = q1 * zss1;
    const double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
    // Compute rmu for each corner
    double rmu = zkur * zrp * cevol;
    rmu = (cdiv > 0. ? 0. : rmu);

 // [5.1] Preparation of extra variables
    sh_ccos[s0] = (minelen < 1.e-12 ? 0. : dot(v1, v2) / (de1 * de2));
    const double csin2 = 1. - sh_ccos[s0] * sh_ccos[s0];
    ctemp[s0]   = ((csin2 < 1.e-4) ? 0. : careap / csin2);
    sh_ccos[s0] = ((csin2 < 1.e-4) ? 0. : sh_ccos[s0]);


    // [4.2] Compute the cqe for each corner
    const double elen1 = length(pxpp1 - pxpp0);
    const double elen2 = length(pxpp2 - pxpp1);
    // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
    //          cqe(2,1,3)=edge 2, x component (1st)
      ctemp1[2*s0]    = rmu * (pup1 - pup0) / elen1;
      ctemp1[2*s0+1]    = rmu * (pup2 - pup1) / elen2;
      __syncthreads();


    // [5.2] Set-Up the forces on corners
    // Edge length for c1, c2 contribution to s
    const double elen = length(pxpp1 - pxpp2);
     sfq = (ctemp[s0] * (  ctemp1[2*s0+1] + sh_ccos[s0] * ctemp1[2*s0]) +
                     ctemp[s04] * ( ctemp1[2*s04] + sh_ccos[s04] * ctemp1[2*s04+1]))/elen;




}



// Routine number [6]  in the full algorithm
__device__ void qcsSetVelDiff(
        const int s,
        const int s0,
        const int p1,
        const int p2,
        const double2 pxpp1,
        const double2 pxpp2,
        const double2 pup1,
        const double2 pup2,
        const int z,
        const double zss1,
        int dss4[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE]) {

    const double2 du = pup2 - pup1;
    const double2 dx = pxpp2 - pxpp1;
    const double lenx = length(dx);
    double dux = dot(du, dx);
    dux = (lenx > 0. ? abs(dux) / lenx : 0.);

    ctemp[s0] = dux;
    __syncthreads();

    double ztmp = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        ztmp = max(ztmp, ctemp[sn]);
    }
    __syncthreads();

    zdu[z] = q1 * zss1 + 2. * q2 * ztmp;
}

// For older ROCm compilers, __HIP_ARCH_GFX908__ is defined for MI100.
// Newer ROCm compilers change this to __gfx908__.
// Using the second launch bound parameter forces spilling to the AccVGPRs,
// which is unique to MI100
#if defined(__gfx908__) or defined(__HIP_ARCH_GFX908__)
__launch_bounds__(64,4)
#else
__launch_bounds__(64)
#endif
__global__ void gpuMain2_opt(int* numsbad_pinned, double dt)
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
    __shared__ double ctemp1[CHUNK_SIZE];
    __shared__ double2 ctemp3[CHUNK_SIZE*2];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];


    // save off zone variable values from previous cycle
    zvol0[z] = zvol[z];

    double2 zxp = {0., 0.};
    double zareap, zvolp, sareap;
    const double2 pxpp1 = pxp[p1];
    const double2 pxpp2 = pxp[p2];

    calcZoneCtrs_SideVols_ZoneVols(s,s0,pxpp1, pxpp2, zxp,
        			sareap,zareap, zvolp,dss4,
				ctemp2, ctemp, ctemp1, numsbad_pinned,
        			z,znump, zdl);
    double2 ssurf = rotateCCW(0.5 * (pxpp1 + pxpp2) - zxp);


    // 2. compute corner masses
    double zmz = zm[z];
    double zpz = zp[z];
    double zssz = zss[z];
    double zrp = zmz / zvolp;
    cmaswt[s] = zrp * zareap * 0.5 * (smf[s] + smf[s3]);

    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    //if (s3 > s) 
    double rx = zr[z];
    pgasCalcStateAtHalf(z, rx, zvolp, zvol0, ze, zwrate, zmz, dt, zpz, zssz);

    
    // 4. compute forces
    const double2 sfp = -zpz * ssurf;
    double2 sft;
    ttsCalcForce(s, z, zareap, zrp, zssz, sareap, smf, ssurf, sft);


   double2 sfq = { 0., 0. };

   const int p0 = mapsp1[s3];
   const double2 pxpp0 = pxp[p0];
   const double2 pup0 = pu[p0];
   const double2 pup1 = pu[p1];
   const double2 pup2 = pu[p2];
   qcsSetCornerDiv(s, s0,  s4, s04, z, p1, p2, pxpp0, pxpp1, pxpp2, pup0, pup1,pup2, 
		   	zxp, zrp, zssz, sfq, dss4, ctemp2,ctemp, ctemp1, ctemp3);


   sfpq[s] = sfp + sfq;
    ctemp2[s0] = sfp + sft + sfq;
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];
    qcsSetVelDiff(s, s0, p1, p2, pxpp1, pxpp2, pup1,pup2, z, zssz,dss4,ctemp);
    zp[z] = zpz;
    zss[z] = zssz;


}





// If we use MPI, then we need to sum corner masses and forces to points locally first,
// then sum the values of the points across MPI ranks for points on the boundaries
// between ranks, and then invoke gpuMain3.
// If we don't use MPI, then the summing of corner masses and forces to points
// is done as the first step in gpuMain3 instead, to reduce the number of kernel
// invocations.
__device__ void localReduceToPoints(const int p,
				    const double* __restrict__ cmaswt,
				    double* __restrict__ pmaswt,
				    const double2* __restrict__ cftot,
				    double2* __restrict__ pf)
{
  double cmaswt_sum = 0.;
  double2 cftot_sum = make_double2(0., 0.);

  int2 first_and_count = first_corner_and_corner_count[p];
  int c = first_and_count.x;
  int count = first_and_count.y;

  union {
    int4 i4;
    int a[4];
  } corners4;

  for(; count > 0; count -=4, c += 4){
    // load in batches of 4. Safe to do, even near the end of the array
    // 'corners_by_point', since it has been over-allocated sufficiently,
    // and we don't use the over-allocated values.
#if !defined(__CUDACC__)
    corners4.i4 = * (int4*)(corners_by_point + c);
#else
    corners4.a[0] = corners_by_point[c + 0];
    corners4.a[1] = corners_by_point[c + 1];
    corners4.a[2] = corners_by_point[c + 2];
    corners4.a[3] = corners_by_point[c + 3];
#endif
    int inner_count = min(count, 4);
    for(int i = 0; i != inner_count; ++i){
      int s = corners4.a[i];
      cmaswt_sum += cmaswt[s];
      cftot_sum += cftot[s];
    }
  }

  pmaswt[p] = cmaswt_sum;
  pf[p] = cftot_sum;
}


#ifdef USE_MPI
__launch_bounds__(256)
__global__ void localReduceToPoints()
{
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    // sum corner masses, forces to points
    localReduceToPoints(p, cmaswt, pmaswt, cftot, pf);
}
#endif

__launch_bounds__(256)
__global__ void gpuMain3(double dt, bool doLocalReduceToPoints)
{
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    if(doLocalReduceToPoints){
      // sum corner masses, forces to points
      localReduceToPoints(p, cmaswt, pmaswt, cftot, pf);
    }

    // 4a. apply boundary conditions
    double2 pxpp = pxp[p];
    double2 pu0p = pu0[p];
    double2 pfp  = pf[p];
    for (int bc = 0; bc < numbcx; ++bc)
        applyFixedBC_opt(p, pxpp, pu0p, pfp, vfixx, bcx[bc]);
    for (int bc = 0; bc < numbcy; ++bc)
        applyFixedBC_opt(p, pxpp, pu0p, pfp, vfixy, bcy[bc]);

    // 5. compute accelerations
    const double fuzz = 1.e-99;
    double2 pa = pfp / max(pmaswt[p], fuzz);

    // ===== Corrector step =====
    // 6. advance mesh to end of time step
    pu[p] = pu0p + pa * dt;
    px[p] = px[p] + 0.5 * (pu[p] + pu0[p]) * dt;
    pu0[p] = pu0p;
    pf[p]  = pfp;
}

__device__ void calcZoneCtrs_SideVols_ZoneVols_main4(
        const int s,
        const int s0,
        const int z,
        const double2 px1,
        const double2 px2,
        double* __restrict__ zarea,
        double & zvol_1,
        int dss4[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE],
        double ctemp1[CHUNK_SIZE],
        int* __restrict__ numsbad_pinned)
{
            ctemp2[s0] = px1;
    __syncthreads();

    double2 zxtot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zxtot += ctemp2[sn];
        zct += 1.;
    }
    double2 zx = zxtot / zct;

    const double third = 1. / 3.;
    const double sa = 0.5 * cross(px2 - px1,  zx - px1);
    const double sv = third * sa * (px1.x + px2.x + zx.x);
    if (sv <= 0.) { atomicAdd(numsbad_pinned, 1); }

    ctemp[s0] = sv;
    ctemp1[s0] = sa;
    __syncthreads();
    double zvtot = ctemp[s0];
    double zatot = ctemp1[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zvtot += ctemp[sn];
        zatot += ctemp1[sn];
    }

    zarea[z] = zatot;
    //zvol[z] = zvtot;
    zvol_1 = zvtot;
}
__launch_bounds__(256)
__global__ void gpuMain4(double_int* dtnext, int* numsbad_pinned, double dt,
			 int* remaining_wg, int gpuMain5_gridsize)
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

    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double ctemp1[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];
    
    dss4[s0] = s04 - s0;

    __syncthreads();

    // 6a. compute new mesh geometry

    const double2 pxp1 = px[p1];
    const double2 pxp2 = px[p2];
    double zvol_1;
    calcZoneCtrs_SideVols_ZoneVols_main4(s,s0,z, pxp1, pxp2, zarea, zvol_1,dss4, ctemp2, ctemp, ctemp1, numsbad_pinned);

    // 7. compute work
    double zwz;
    hydroCalcWork(s, s0, z, p1, p2, sfpq, pu0, pu, pxp, dt,
		  zwz, zetot, dss4, ctemp);

    const double dvol = zvol_1 - zvol0[z];
    zwrate[z] = (zwz + zp[z] * dvol) / dt;
    const double fuzz = 1.e-99;
    ze[z] = zetot[z] / (zm[z] + fuzz);
    zr[z] = zm[z] / zvol_1;
    zvol[z] = zvol_1;

    // reset dtnext and remaining_wg prior to finding the minimum over all sides in the next kernel
    if(blockIdx.x == 0 and threadIdx.x == 0){
      dtnext->d = 1.e99;
      *remaining_wg = gpuMain5_gridsize;
    }
}


__launch_bounds__(256)
__global__ void gpuMain5(double_int* dtnext, double dt, double_int* dtnext_H, int* remaining_wg, int* pinned_control_flag)
{
    const int z = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (z >= numz) return;

    const int z0 = threadIdx.x;
    const int zlength = min((int)CHUNK_SIZE, (int)(numz - blockIdx.x * CHUNK_SIZE));

    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];

    // compute timestep for next cycle
    hydroCalcDt(z, z0, zlength, zdu, zss, zdl, zvol, zvol0, dt,
		ctemp, ctemp2, dtnext, dtnext_H, remaining_wg, pinned_control_flag);
}


void meshCheckBadSides() {
    if (*numsbad_pinned > 0) {
        cerr << "Error: " << *numsbad_pinned << " negative side volumes" << endl;
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
        const int* mapss4H,
        const int* mapseH,
        const int* znumpH) {

#ifdef USE_MPI
  using Parallel::mype;
#else
  int mype = 0;
#endif

    if(mype == 0){ printf("Running Hydro on device...\n"); }

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

    
    CHKERR(hipHostMalloc(&numsbad_pinned, sizeof(int),hipHostMallocCoherent));
    *numsbad_pinned = 0;
    CHKERR(hipHostMalloc(&pinned_control_flag, sizeof(int),hipHostMallocCoherent));
    CHKERR(hipHostMalloc(&dtnext_H, sizeof(double_int),hipHostMallocCoherent));
    CHKERR(hipMalloc(&dtnext_D, sizeof(double_int)));
    CHKERR(hipMalloc(&remaining_wg_D, sizeof(int)));

    CHKERR(hipMalloc(&schsfirstD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schslastD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schzfirstD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schzlastD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&mapsp1D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapsp2D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapszD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapss4D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&znumpD, numzH*sizeof(int)));

    CHKERR(hipMalloc(&pxD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&pxpD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&puD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&pu0D, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&zmD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zrD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zareaD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvolD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvol0D, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zdlD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zduD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zeD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zetot0D, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zetotD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zwrateD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zssD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&smfD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&zvolpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&cmaswtD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&pmaswtD, numpH*sizeof(double)));
    CHKERR(hipMalloc(&sfpqD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&cftotD, numcH*sizeof(double2)));
    CHKERR(hipMalloc(&pfD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&mapspkeyD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapspvalD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mappsfirstD, numpH*sizeof(int)));
    CHKERR(hipMalloc(&mapssnextD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&corners_per_pointD, numpH*sizeof(int)));
    CHKERR(hipMalloc(&corners_by_pointD, (numsH+4)*sizeof(int)));
    CHKERR(hipMalloc(&first_corner_of_pointD, numpH*sizeof(int)));
    CHKERR(hipMalloc(&first_corner_and_corner_countD, numpH*sizeof(int2)));

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(schsfirst), &schsfirstD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(schslast), &schslastD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsp1), &mapsp1D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsp2), &mapsp2D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsz), &mapszD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapss4), &mapss4D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mappsfirst), &mappsfirstD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapssnext), &mapssnextD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(znump), &znumpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(px), &pxD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pxp), &pxpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pu), &puD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pu0), &pu0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zm), &zmD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zr), &zrD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zarea), &zareaD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol), &zvolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol0), &zvol0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdl), &zdlD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdu), &zduD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ze), &zeD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zetot), &zetotD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zwrate), &zwrateD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zp), &zpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zss), &zssD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(smf), &smfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvolp), &zvolpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cmaswt), &cmaswtD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pmaswt), &pmaswtD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sfpq), &sfpqD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cftot), &cftotD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pf), &pfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(corners_per_point), &corners_per_pointD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(corners_by_point), &corners_by_pointD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(first_corner_of_point), &first_corner_of_pointD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(first_corner_and_corner_count), &first_corner_and_corner_countD, sizeof(void*)));

    CHKERR(hipMemcpy(schsfirstD, schsfirstH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schslastD, schslastH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schzfirstD, schzfirstH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schzlastD, schzlastH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapsp1D, mapsp1H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapsp2D, mapsp2H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapszD, mapszH, numsH*sizeof(int), hipMemcpyHostToDevice));
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

    thrust::device_ptr<int> mapsp1T(mapsp1D);
    thrust::device_ptr<int> mapspkeyT(mapspkeyD);
    thrust::device_ptr<int> mapspvalT(mapspvalD);

    thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
    thrust::sequence(mapspvalT, mapspvalT + numsH);
    thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);

    int gridSizeS = (numsH+CHUNK_SIZE-1) / CHUNK_SIZE;
    int chunkSize = CHUNK_SIZE;
    hipLaunchKernelGGL((gpuInvMap), dim3(gridSizeS), dim3(chunkSize), 0, 0, mapspkeyD, mapspvalD,
		       mappsfirstD, mapssnextD);

    int gridSizeP = (numpH + CHUNK_SIZE - 1) / CHUNK_SIZE;
    hipLaunchKernelGGL(calcCornersPerPoint, dim3(gridSizeP), dim3(chunkSize), 0, 0, corners_per_pointD);

    thrust::exclusive_scan(thrust::device, corners_per_pointD, corners_per_pointD + numpH, first_corner_of_pointD);
    hipLaunchKernelGGL(storeCornersByPoint, dim3(gridSizeP), dim3(chunkSize), 0, 0, first_corner_of_pointD,
		       corners_by_pointD, corners_per_pointD, first_corner_and_corner_countD);
    hipDeviceSynchronize();
    hipFree(first_corner_of_pointD);
    hipFree(corners_per_pointD);
#ifdef USE_JIT

    replacement_t replacements {
      { "${CHUNK_SIZE}", jit_string(CHUNK_SIZE) },
      { "${cftot}", jit_string(cftotD) },
      { "${cmaswt}", jit_string(cmaswtD) },
      { "${cw}", jit_string(cwD) },
      { "${mapsp1}", jit_string(mapsp1D) },
      { "${mapsp2}", jit_string(mapsp2D) },
      { "${mapss4}", jit_string(mapss4D) },
      { "${mapsz}", jit_string(mapszD) },
      { "${nump}", jit_string(numpH) },
      { "${pgamma}", jit_string(pgammaH) },
      { "${pssmin}", jit_string(pssminH) },
      { "${pu0}", jit_string(pu0D) },
      { "${pu}", jit_string(puD) },
      { "${pxp}", jit_string(pxpD) },
      { "${px}", jit_string(pxD) },
      { "${q1}", jit_string(q1H) },
      { "${q2}", jit_string(q2H) },
      { "${qgamma}", jit_string(qgammaH) },
      { "${schsfirst}", jit_string(schsfirstD) },
      { "${schslast}", jit_string(schslastD) },
      { "${sfpq}", jit_string(sfpqD) },
      { "${smf}", jit_string(smfD) },
      { "${talfa}", jit_string(talfaH) },
      { "${tssmin}", jit_string(tssminH) },
      { "${zdl}", jit_string(zdlD) },
      { "${zdu}", jit_string(zduD) },
      { "${ze}", jit_string(zeD) },
      { "${zm}", jit_string(zmD) },
      { "${znump}", jit_string(znumpD) },
      { "${zp}", jit_string(zpD) },
      { "${zr}", jit_string(zrD) },
      { "${zss}", jit_string(zssD) },
      { "${zvol0}", jit_string(zvol0D) },
      { "${zvolp}", jit_string(zvolpD) },
      { "${zvol}", jit_string(zvolD) },
      { "${zwrate}", jit_string(zwrateD) }
    };
    jit = std::unique_ptr<Pajama>(new Pajama("src.jit/kernels.cc", replacements));
    jit->load_kernel("gpuMain1_jit");
    jit->load_kernel("gpuMain2_jit");
#endif // USE_JIT
    hipEventCreate(&mainLoopEvent);
}

#ifdef USE_MPI
__launch_bounds__(256)
__global__ void copySlavePointDataToMPIBuffers_kernel(double* pmaswt_slave_buffer_D,
						      double2* pf_slave_buffer_D){
  int slave = blockIdx.x * blockDim.x + threadIdx.x;
  if(slave >= numslv) { return; }
  int point = mapslvp[slave];
  pmaswt_slave_buffer_D[slave] = pmaswt[point];
  pf_slave_buffer_D[slave] = pf[point];
}

__launch_bounds__(256)
__global__ void copySlavePointDataToMPIBuffers_kernel_test(double* pmaswt_pf_slave_buffer_D){

   int slave = blockIdx.x * blockDim.x + threadIdx.x;
  if(slave >= numslv) { return; }
  int point = mapslvp[slave];
  int pos = slave * 3;
  pmaswt_pf_slave_buffer_D[pos] = pmaswt[point];
  pmaswt_pf_slave_buffer_D[pos+1] = pf[point].x;
  pmaswt_pf_slave_buffer_D[pos+2] = pf[point].y;
}

void copySlavePointDataToMPIBuffers(){
  constexpr int blocksize = 256;
  const int blocks = (numslvH + blocksize - 1) / blocksize;
  {
    DEVICE_TIMER("MPI device overhead", "copySlavePointDataToMPIBuffers_kernel_test", 0);
    hipLaunchKernelGGL(copySlavePointDataToMPIBuffers_kernel_test, blocks, blocksize, 0, 0,
                       pmaswt_pf_slave_buffer_D);
  }

#ifndef USE_GPU_AWARE_MPI
    CHKERR(hipMemcpy(pmaswt_pf_slave_buffer_H, pmaswt_pf_slave_buffer_D, numslvH * 3 * sizeof(double), hipMemcpyDeviceToHost));
#else
  hipDeviceSynchronize();
#endif
}


__launch_bounds__(256)
__global__ void copyMPIBuffersToSlavePointData_kernel(double* pmaswt_slave_buffer_D,
						      double2* pf_slave_buffer_D){
  int slave = blockIdx.x * blockDim.x + threadIdx.x;
  if(slave >= numslv) { return; }
  int point = mapslvp[slave];
  pmaswt[point] = pmaswt_slave_buffer_D[slave];
  pf[point] = pf_slave_buffer_D[slave];
}

__launch_bounds__(256)
__global__ void copyMPIBuffersToSlavePointData_kernel_test(double* pmaswt_pf_slave_buffer_D){

  int slave = blockIdx.x * blockDim.x + threadIdx.x;
  if(slave >= numslv) { return; }
  int point = mapslvp[slave];
  int pos = slave * 3;
  pmaswt[point] = pmaswt_pf_slave_buffer_D[pos];
  pf[point].x = pmaswt_pf_slave_buffer_D[pos+1];
  pf[point].y = pmaswt_pf_slave_buffer_D[pos+2];
}

void copyMPIBuffersToSlavePointData(){
#ifndef USE_GPU_AWARE_MPI
  CHKERR(hipMemcpy(pmaswt_pf_slave_buffer_D, pmaswt_pf_slave_buffer_H, numslvH * 3 * sizeof(double), hipMemcpyHostToDevice));
#endif
  constexpr int blocksize = 256;
  const int blocks = (numslvH + blocksize - 1) / blocksize;
    {
    DEVICE_TIMER("MPI device overhead", "copyMPIBuffersToSlavePointData_kernel_test", 0);
    hipLaunchKernelGGL(copyMPIBuffersToSlavePointData_kernel_test, blocks, blocksize, 0, 0,
                       pmaswt_pf_slave_buffer_D);
  }
}


__launch_bounds__(256)
__global__ void reduceToMasterPoints(double* pmaswt_proxy_buffer_D,
					   double2* pf_proxy_buffer_D){
  int proxy = blockIdx.x * blockDim.x + threadIdx.x;
  if(proxy >= numprx) { return; }

  int point = mapprxp[proxy];
  atomicAdd(&pmaswt[point], pmaswt_proxy_buffer_D[proxy]);
  atomicAdd(&pf[point].x, pf_proxy_buffer_D[proxy].x);
  atomicAdd(&pf[point].y, pf_proxy_buffer_D[proxy].y);
}

__launch_bounds__(256)
__global__ void reduceToMasterPoints_test(double* pmaswt_pf_proxy_buffer_D){

  int proxy = blockIdx.x * blockDim.x + threadIdx.x;
  if(proxy >= numprx) { return; }

  int point = mapprxp[proxy];
  int pos = proxy * 3;
  atomicAdd(&pmaswt[point], pmaswt_pf_proxy_buffer_D[pos]);
  atomicAdd(&pf[point].x, pmaswt_pf_proxy_buffer_D[pos+1]);
  atomicAdd(&pf[point].y, pmaswt_pf_proxy_buffer_D[pos+2]);
}



__launch_bounds__(256)
__global__ void copyPointValuesToProxies(double* pmaswt_proxy_buffer_D,
					 double2* pf_proxy_buffer_D){
  int proxy = blockIdx.x * blockDim.x + threadIdx.x;
  if(proxy >= numprx) { return; }

  int point = mapprxp[proxy];
  pmaswt_proxy_buffer_D[proxy] = pmaswt[point];
  pf_proxy_buffer_D[proxy] = pf[point];
}


__launch_bounds__(256)
__global__ void copyPointValuesToProxies_test(double* pmaswt_pf_proxy_buffer_D){

  int proxy = blockIdx.x * blockDim.x + threadIdx.x;
  if(proxy >= numprx) { return; }

  int point = mapprxp[proxy];
  int pos = proxy * 3;
  pmaswt_pf_proxy_buffer_D[pos] = pmaswt[point];
  pmaswt_pf_proxy_buffer_D[pos+1] = pf[point].x;
  pmaswt_pf_proxy_buffer_D[pos+2] = pf[point].y;

}

void reduceToMasterPointsAndProxies(){
#ifndef USE_GPU_AWARE_MPI
  CHKERR(hipMemcpy(pmaswt_pf_proxy_buffer_D, pmaswt_pf_proxy_buffer_H, numprxH * 3 * sizeof(double), hipMemcpyHostToDevice));
#endif

  constexpr int blocksize = 256;
  const int blocks = (numprxH + blocksize - 1) / blocksize;
  {
    DEVICE_TIMER("Kernels", "reduceToMasterPoints_test", 0);
    hipLaunchKernelGGL(reduceToMasterPoints_test, blocks, blocksize, 0, 0,
                       pmaswt_pf_proxy_buffer_D);
  }
  {
    DEVICE_TIMER("MPI device overhead", "copyPointValuesToProxies_test", 0);
    hipLaunchKernelGGL(copyPointValuesToProxies_test, blocks, blocksize, 0, 0,
                       pmaswt_pf_proxy_buffer_D);
  }
#ifndef USE_GPU_AWARE_MPI
    CHKERR(hipMemcpy(pmaswt_pf_proxy_buffer_H, pmaswt_pf_proxy_buffer_D, numprxH * 3 * sizeof(double), hipMemcpyDeviceToHost));  
#else
  hipDeviceSynchronize();
#endif
}

void globalReduceToPoints() {
  copySlavePointDataToMPIBuffers();
  {
    HOST_TIMER("MPI", "parallelGather_test");
    parallelGather_test( numslvpeD, nummstrpeD,
                    mapslvpepeD,  slvpenumprxD,  mapslvpeprx1D,
                    mapmstrpepeD,  mstrpenumslvD,  mapmstrpeslv1D,
                    pmaswt_pf_proxy_buffer, pmaswt_pf_slave_buffer);
  }

  reduceToMasterPointsAndProxies();
  {
   HOST_TIMER("MPI", "parallelScatter_test");
      parallelScatter_test( numslvpeD, nummstrpeD,
                       mapslvpepeD,  slvpenumprxD,  mapslvpeprx1D,
                       mapmstrpepeD,  mstrpenumslvD,  mapmstrpeslv1D,  mapslvpD,
                       pmaswt_pf_proxy_buffer, pmaswt_pf_slave_buffer);
  }

  copyMPIBuffersToSlavePointData();
}
#endif


void hydroDoCycle(
        const double dt,
        double& dtnextH,
        int& idtnextH) {
    int gridSizeS, gridSizeP, gridSizeZ, chunkSize;

    gridSizeS = numschH;
    gridSizeP = numpchH;
    gridSizeZ = numzchH;
    chunkSize = CHUNK_SIZE;
    
#ifdef USE_JIT
    struct {
      double dt;
    } gpu_args;
    gpu_args.dt = dt;
    size_t gpu_args_size = sizeof(gpu_args);
    void* gpu_args_wrapper[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &gpu_args,
    				 HIP_LAUNCH_PARAM_BUFFER_SIZE, &gpu_args_size,
    				 HIP_LAUNCH_PARAM_END };
#endif

#ifdef USE_JIT
    {
      DEVICE_TIMER("Kernels", "gpuMain1_jit", 0);
      jit->call_preloaded("gpuMain1_jit", gridSizeP, chunkSize, 0, 0, gpu_args_wrapper);
    }
    {
      DEVICE_TIMER("Kernels", "gpuMain2_jit", 0);
      jit->call_preloaded("gpuMain2_jit", gridSizeS, chunkSize, 0, 0, gpu_args_wrapper);
    }
#else
    {
      DEVICE_TIMER("Kernels", "gpuMain1", 0);
      hipLaunchKernelGGL((gpuMain1), dim3(gridSizeP), dim3(chunkSize), 0, 0, dt);
    }
    {
       DEVICE_TIMER("Kernels", "gpuMain2_opt", 0);
       hipLaunchKernelGGL((gpuMain2_opt), dim3(gridSizeS), dim3(chunkSize), 0, 0, numsbad_pinned, dt);
    }
#endif

    {
      HOST_TIMER("Other", "meshCheckBadSides");
      meshCheckBadSides();
    }
    bool doLocalReduceToPointInGpuMain3 = true;

#ifdef USE_MPI
    if(Parallel::numpe > 1){
      // local reduction to points needs to be done either way, but if numpe == 1, then
      // we can do it in gpuMain3, which saves a kernel call
      doLocalReduceToPointInGpuMain3 = false;
      {
	DEVICE_TIMER("Kernels", "localReduceToPoints", 0);
	hipLaunchKernelGGL((localReduceToPoints), dim3(gridSizeP), dim3(chunkSize), 0, 0);
      }
      globalReduceToPoints();
    }
#endif
    {
      DEVICE_TIMER("Kernels", "gpuMain3", 0);
      hipLaunchKernelGGL((gpuMain3), dim3(gridSizeP), dim3(chunkSize), 0, 0,
			 dt, doLocalReduceToPointInGpuMain3);
    }

    {
      DEVICE_TIMER("Kernels", "gpuMain4", 0);
      hipLaunchKernelGGL((gpuMain4), dim3(gridSizeS), dim3(chunkSize), 0, 0, dtnext_D, numsbad_pinned, dt,
			 remaining_wg_D, gridSizeZ);
    }

    *pinned_control_flag = 0;
    {
      DEVICE_TIMER("Kernels", "gpuMain5", 0);
      hipLaunchKernelGGL((gpuMain5), dim3(gridSizeZ), dim3(chunkSize), 0, 0, dtnext_D, dt,
			 dtnext_H, remaining_wg_D, pinned_control_flag);
    }

    {
      HOST_TIMER("Other", "meshCheckBadSides");
      meshCheckBadSides();
    }

    {
      HOST_TIMER("Other", "copy dtnext values");
      for(int flag = 0; flag == 0; __atomic_load(pinned_control_flag, &flag, __ATOMIC_ACQUIRE)); // spin on flag
      dtnextH = dtnext_H->d;
      idtnextH = dtnext_H->i;
    }
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


  if(numprxH) hipMalloc(&pmaswt_pf_proxy_buffer_D, numprxH*3*sizeof(double));
  if(numslvH) hipMalloc(&pmaswt_pf_slave_buffer_D, numslvH*3*sizeof(double));


#ifndef USE_GPU_AWARE_MPI
  if(numprxH) pmaswt_proxy_buffer_H = Memory::alloc<double>(numprxH);
  if(numslvH) pmaswt_slave_buffer_H = Memory::alloc<double>(numslvH);
  if(numprxH) pf_proxy_buffer_H = Memory::alloc<double2>(numprxH);
  if(numslvH) pf_slave_buffer_H = Memory::alloc<double2>(numslvH);

  if(numprxH) pmaswt_pf_proxy_buffer_H = Memory::alloc<double>(numprxH*3);
  if(numslvH) pmaswt_pf_slave_buffer_H = Memory::alloc<double>(numslvH*3);


  pmaswt_proxy_buffer = pmaswt_proxy_buffer_H;
  pmaswt_slave_buffer = pmaswt_slave_buffer_H;
  pf_proxy_buffer = pf_proxy_buffer_H;
  pf_slave_buffer = pf_slave_buffer_H;

  pmaswt_pf_proxy_buffer = pmaswt_pf_proxy_buffer_H;
  pmaswt_pf_slave_buffer = pmaswt_pf_slave_buffer_H;

#else
  pmaswt_proxy_buffer = pmaswt_proxy_buffer_D;
  pmaswt_slave_buffer = pmaswt_slave_buffer_D;
  pf_proxy_buffer = pf_proxy_buffer_D;
  pf_slave_buffer = pf_slave_buffer_D;

  pmaswt_pf_proxy_buffer = pmaswt_pf_proxy_buffer_D;
  pmaswt_pf_slave_buffer = pmaswt_pf_slave_buffer_D;

#endif
}
#endif

void hydroInitGPU()
{
#ifdef USE_MPI
  using Parallel::mype;
#else
  constexpr int mype = 0;
#endif
  
  // TODO: consider letting slurm handle the pe to device mapping
  int nDevices;
  hipGetDeviceCount(&nDevices);
  int device_num = mype % nDevices;
  hipSetDevice(device_num);
}


void hydroFinalGPU()
{
  // TODO: free resources
}

