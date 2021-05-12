#include <hip/hip_runtime.h>
#include "../src.hip/Vec2.hh"

constexpr int CHUNK_SIZE = ${CHUNK_SIZE};

extern "C" {

  //-- gpuMain1 ----------------------------------------------------------
  __launch_bounds__(256)
  __global__ void gpuMain1_jit(double dt)
  {
    constexpr int nump = ${nump};
    const double2* const px = ${px};
    const double2* const pu = ${pu};
    double2* const pu0 = ${pu0};
    double2* const pxp = ${pxp};

    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    double dth = 0.5 * dt;

    // save off point variable values from previous cycle
    pu0[p] = pu[p];
    pxp[p] = px[p] + pu0[p] * dth;

    // ===== Predictor step =====
    // 1. advance mesh to center of time step
  }


  //-- gpuMain2 and supporting device functions -------------------------

  __device__ void calcZoneCtrs_SideVols_ZoneVols_jit(
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

  __device__ void pgasCalcStateAtHalf_jit(const int z,
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
    constexpr double pgamma = ${pgamma};
    constexpr double pssmin = ${pssmin};
    constexpr double tssmin = ${tssmin};
    constexpr double talfa = ${talfa};
    constexpr double qgamma = ${qgamma};
    constexpr double q1 = ${q1};
    constexpr double q2 = ${q2};
    double* zdu = ${zdu};

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


  static __device__ void ttsCalcForce_jit(const int s,
					  const int z,
					  const double zareap,
					  const double zrp,
					  const double zssz,
					  const double sareap,
					  const double* __restrict__ smf,
					  const double2 ssurf,
					  double2 &sft) {

    constexpr double tssmin = ${tssmin};
    constexpr double talfa = ${talfa};

    const double svfacinv = zareap / sareap;
    const double srho = zrp * smf[s] * svfacinv;
    double sstmp = max(zssz, tssmin);
    sstmp = talfa * sstmp * sstmp;
    const double sdp = sstmp * (srho - zrp);
    sft = -sdp * ssurf;
  }

  __device__ void qcsSetCornerDiv_jit(const int s,
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
    
    constexpr double qgamma = ${qgamma};
    constexpr double q1 = ${q1};
    constexpr double q2 = ${q2};

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

   __device__ void qcsSetVelDiff_jit(
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
    double* zdu = ${zdu};
    constexpr double q1 = ${q1};
    constexpr double q2 = ${q2};
     
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

#if defined(__gfx908__) or defined(__HIP_ARCH_GFX908__)
__launch_bounds__(64,4)
#else
__launch_bounds__(64)
#endif
  __global__ void gpuMain2_jit(double dt)
  {
    const int* const schsfirst = ${schsfirst};
    const int* const schslast = ${schslast};
    const int* const mapsp1 = ${mapsp1};
    const int* const mapsp2 = ${mapsp2};
    const int* const mapsz = ${mapsz};
    const int* const mapss4 = ${mapss4};
    double* const zvol0 = ${zvol0};
    const double* const zvol = ${zvol};
    const double2* const pxp = ${pxp};
    const int* const znump = ${znump};
    double* const zdl = ${zdl};
    double* const zp = ${zp};
    double* const zm = ${zm};
    double* const zss = ${zss};
    double* const cmaswt = ${cmaswt};
    const double* const smf = ${smf};
    const double* const zr = ${zr};
    const double* const ze = ${ze};
    const double2* const pu = ${pu};
    double2* const sfpq = ${sfpq};
    double2* const cftot = ${cftot};
    const double* const zwrate = ${zwrate};
    int* const numsbad_pinned = ${numsbad_pinned};
    
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

    calcZoneCtrs_SideVols_ZoneVols_jit(s,s0,pxpp1, pxpp2, zxp,
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
    pgasCalcStateAtHalf_jit(z, rx, zvolp, zvol0, ze, zwrate, zmz, dt, zpz, zssz);

    
    // 4. compute forces
    const double2 sfp = -zpz * ssurf;
    double2 sft;
    ttsCalcForce_jit(s, z, zareap, zrp, zssz, sareap, smf, ssurf, sft);


    double2 sfq = { 0., 0. };

    const int p0 = mapsp1[s3];
    const double2 pxpp0 = pxp[p0];
    const double2 pup0 = pu[p0];
    const double2 pup1 = pu[p1];
    const double2 pup2 = pu[p2];
    qcsSetCornerDiv_jit(s, s0,  s4, s04, z, p1, p2, pxpp0, pxpp1, pxpp2, pup0, pup1,pup2, 
			zxp, zrp, zssz, sfq, dss4, ctemp2,ctemp, ctemp1, ctemp3);


    sfpq[s] = sfp + sfq;
    ctemp2[s0] = sfp + sft + sfq;
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];
    qcsSetVelDiff_jit(s, s0, p1, p2, pxpp1, pxpp2, pup1,pup2, z, zssz,dss4,ctemp);
    zp[z] = zpz;
    zss[z] = zssz;
  }

} // extern "C"
