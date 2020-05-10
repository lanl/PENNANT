#include <hip/hip_runtime.h>
#include "../src.hip/Vec2.hh"

constexpr int CHUNK_SIZE = ${CHUNK_SIZE};

extern "C" {

  //-- gpuMain1 and supporting __device__ functions ----------------------
  __device__ void advPosHalf_jit(
				 const int p,
				 const double2* __restrict__ px0,
				 const double2* __restrict__ pu0,
				 const double dt,
				 double2* __restrict__ pxp) {
    pxp[p] = px0[p] + pu0[p] * dt;
  }

  
  __global__ void
  __launch_bounds__(CHUNK_SIZE)
    gpuMain1_jit(double dt)
  {
    constexpr int nump = ${nump};
    const double2* const px = ${px};
    double2* const px0 = ${px0};
    const double2* const pu = ${pu};
    double2* const pu0 = ${pu0};
    double2* const pxp = ${pxp};

    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;
  
    double dth = 0.5 * dt;
  
    // save off point variable values from previous cycle
    px0[p] = px[p];
    pu0[p] = pu[p];
  
    // ===== Predictor step =====
    // 1. advance mesh to center of time step
    advPosHalf_jit(p, px0, pu0, dth, pxp);
  }


  //-- gpuMain2 and supporting __device__ functions ----------------------

  __device__ void calcZoneCtrs_jit(
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


  __device__ void meshCalcCharLen_jit(
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


  __device__ void calcSideVols_jit(
				   const int s,
				   const int z,
				   const int p1,
				   const int p2,
				   const double2* __restrict__ px,
				   const double2* __restrict__ zx,
				   double* __restrict__ sarea,
				   double* __restrict__ svol,
				   int* numsbad)
  {
    constexpr double third = 1. / 3.;
    double sa = 0.5 * cross(px[p2] - px[p1],  zx[z] - px[p1]);
    double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
    sarea[s] = sa;
    svol[s] = sv;
    
    if (sv <= 0.) atomicAdd(numsbad, 1);
  }


  __device__ void calcZoneVols_jit(
				   const int s,
				   const int s0,
				   const int z,
				   const double* __restrict__ sarea,
				   const double* __restrict__ svol,
				   double* __restrict__ zarea,
				   double* __restrict__ zvol)
  {
    const int* const mapss4 = ${mapss4};
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

  __device__ void hydroCalcRho_jit(const int z,
				   const double* __restrict__ zm,
				   const double* __restrict__ zvol,
				   double* __restrict__ zr)
  {
    zr[z] = zm[z] / zvol[z];
  }

  __device__ void calcCrnrMass_jit(
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

  __device__ void pgasCalcEOS_jit(
				  const int z,
				  const double* __restrict__ zr,
				  const double* __restrict__ ze,
				  double* __restrict__ zp,
				  double& zper,
				  double* __restrict__ zss)
  {
    constexpr double pgamma = ${pgamma};
    constexpr double pssmin = ${pssmin};
  
    constexpr double gm1 = pgamma - 1.;
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


  __device__ void pgasCalcStateAtHalf_jit(
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
    pgasCalcEOS_jit(z, zr0, ze, zp, zper, zss);

    const double dth = 0.5 * dt;
    const double zminv = 1. / zm[z];
    double dv = (zvolp[z] - zvol0[z]) * zminv;
    double bulk = zr0[z] * zss[z] * zss[z];
    double denom = 1. + 0.5 * zper * dv;
    double src = zwrate[z] * dth * zminv;
    zp[z] += (zper * src - zr0[z] * bulk * dv) / denom;
  }

  __device__ void pgasCalcForce_jit(
				    const int s,
				    const int z,
				    const double* __restrict__ zp,
				    const double2* __restrict__ ssurf,
				    double2* __restrict__ sf) {
    sf[s] = -zp[z] * ssurf[s];
  }


  __device__ void ttsCalcForce_jit(
				   const int s,
				   const int z,
				   const double* __restrict__ zarea,
				   const double* __restrict__ zr,
				   const double* __restrict__ zss,
				   const double* __restrict__ sarea,
				   const double* __restrict__ smf,
				   const double2* __restrict__ ssurf,
				   double2* __restrict__ sf) {
    constexpr double tssmin = ${tssmin};
    constexpr double talfa = ${talfa};
  
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
  __device__ void qcsSetCornerDiv_jit(
				      const int s,
				      const int s0,
				      const int s3,
				      const int z,
				      const int p1,
				      const int p2,
				      int dss4[CHUNK_SIZE],
				      double2 ctemp2[CHUNK_SIZE]) {
    const double2* const pu = ${pu};
    double2* const zuc = ${zuc};
    const int* const mapsp1 = ${mapsp1};
    double2* const pxp = ${pxp};
    const double2* const zxp = ${zxp};
    double* const ccos = ${ccos};
    double* const careap = ${careap};
    double* const cdiv = ${cdiv};
    double* const cdu = ${cdu};
    double* const cevol = ${cevol};
  
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
  __device__ void qcsSetQCnForce_jit(
				     const int s,
				     const int s3,
				     const int z,
				     const int p1,
				     const int p2) {
    constexpr double qgamma = ${qgamma};
    constexpr double q1 = ${q1};
    constexpr double q2 = ${q2};
    const double* const zss = ${zss};
    const double* const zrp = ${zrp};
    const double* const cevol = ${cevol};
    const double* const cdiv = ${cdiv};
    const int* const mapsp1 = ${mapsp1};
    const double* const cdu = ${cdu};
    const double2* const pxp = ${pxp};
    double2* const cqe = ${cqe};
    const double2* const pu = ${pu};

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
  __device__ void qcsSetForce_jit(
				  const int s,
				  const int s4,
				  const int p1,
				  const int p2) {
    double* const ccos = ${ccos};
    double* const cw = ${cw};
    const double* const careap = ${careap};
    const double2* const pxp = ${pxp};
    double2* const sfq = ${sfq};
    const double2* const cqe = ${cqe};
    
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
  __device__ void qcsSetVelDiff_jit(
				    const int s,
				    const int s0,
				    const int p1,
				    const int p2,
				    const int z,
				    int dss4[CHUNK_SIZE],
				    double ctemp[CHUNK_SIZE] ) {
    const double2* const pxp = ${pxp};
    const double2* const pu = ${pu};
    double* const zdu = ${zdu}; 
    const double* const zss = ${zss};
    constexpr double q1 = ${q1};
    constexpr double q2 = ${q2};

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



  __device__ void qcsCalcForce_jit(
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
    const double2* const sfp = ${sfp};
    const double2* const sft = ${sft};
    const double2* const sfq = ${sfq};
    double2* const cftot = ${cftot};
    
    // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv_jit(s, s0, s3, z, p1, p2,dss4, ctemp2);

    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce_jit(s, s3, z, p1, p2);

    // [5] Compute the Q forces
    qcsSetForce_jit(s, s4, p1, p2);

    ctemp2[s0] = sfp[s] + sft[s] + sfq[s];
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff_jit(s, s0, p1, p2, z, dss4, ctemp);
  }


  __global__ void
  __launch_bounds__(CHUNK_SIZE)
  gpuMain2_jit(double dt)
  {
    const int* const mapsp1 = ${mapsp1};
    const int* const mapsp2 = ${mapsp2};
    const int* const mapss4 = ${mapss4};
    const int* const mapsz = ${mapsz};
    const int* const schsfirst = ${schsfirst};
    const int* const schslast = ${schslast};
    const int* const znump = ${znump};

    int* const numsbad = ${numsbad};
    const double* const smf = ${smf};
    const double* const ze = ${ze};
    const double* const zm = ${zm};
    const double* const zvol = ${zvol};
    const double* const zwrate = ${zwrate};

    double* cmaswt = ${cmaswt};
    double* const sareap = ${sareap};
    double* const svolp = ${svolp};
    double* const zareap = ${zareap};
    double* const zdl = ${zdl};
    double* const zp = ${zp};
    double* const zr = ${zr};
    double* const zrp = ${zrp};
    double* const zss = ${zss};
    double* const zvol0 = ${zvol0};
    double* const zvolp = ${zvolp};

    double2* const pxp = ${pxp};
    double2* const sfp = ${sfp};
    double2* const sft = ${sft};
    double2* const ssurf = ${ssurf};
    double2* const zxp = ${zxp};

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
    calcZoneCtrs_jit(s, s0, z, p1, pxp, zxp, dss4, ctemp2);
    meshCalcCharLen_jit(s, s0, s3, z, p1, p2, znump, pxp, zxp, zdl, dss4, ctemp);

    ssurf[s] = rotateCCW(0.5 * (pxp[p1] + pxp[p2]) - zxp[z]);
    calcSideVols_jit(s, z, p1, p2, pxp, zxp, sareap, svolp, numsbad);
    calcZoneVols_jit(s, s0, z, sareap, svolp, zareap, zvolp);

    // 2. compute corner masses
    hydroCalcRho_jit(z, zm, zvolp, zrp);
    calcCrnrMass_jit(s, s3, z, zrp, zareap, smf, cmaswt);

    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    if (s3 > s) pgasCalcStateAtHalf_jit(z, zr, zvolp, zvol0, ze, zwrate,
					zm, dt, zp, zss);
    __syncthreads();

    // 4. compute forces
    pgasCalcForce_jit(s, z, zp, ssurf, sfp);
    ttsCalcForce_jit(s, z, zareap, zrp, zss, sareap, smf, ssurf, sft);
    qcsCalcForce_jit(s, s0, s3, s4, z, p1, p2, dss3, dss4, ctemp, ctemp2);

  }
} // extern "C"
