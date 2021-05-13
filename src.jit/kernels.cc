#include <hip/hip_runtime.h>
#include "../src.hip/Vec2.hh"

#define OPT_INLINE

extern "C" {

  namespace {
    struct double_int {
      double d;
      int i;
    };

    constexpr int CHUNK_SIZE = ${CHUNK_SIZE};
    constexpr double bcx0 = ${bcx0};
    constexpr double bcx1 = ${bcx1};
    constexpr double bcy0 = ${bcy0};
    constexpr double bcy1 = ${bcy1};
    // const double2* const cftot = ${cftot};
    double2* const cftot = ${cftot};
    // const double* const cmaswt = ${cmaswt};
    double* const cmaswt = ${cmaswt};
    const int* const corners_by_point = ${corners_by_point};
    constexpr bool doLocalReduceToPoints = ${doLocalReduceToPoints};
    double_int* const dtnext = ${dtnext};
    double_int* const dtnext_H = ${dtnext_H};
    const int2* const first_corner_and_corner_count = ${first_corner_and_corner_count};
    constexpr int gpuMain5_gridsize = ${gpuMain5_gridsize};
    constexpr double hcfl = ${hcfl};
    constexpr double hcflv = ${hcflv};
    const int* const mapsp1 = ${mapsp1};
    const int* const mapsp2 = ${mapsp2};
    const int* const mapss4 = ${mapss4};
    const int* const mapsz = ${mapsz};
    constexpr int nump = ${nump};
    int* const numsbad_pinned = ${numsbad_pinned};
    constexpr int numz = ${numz};
    double2* const pf = ${pf};
    constexpr double pgamma = ${pgamma};
    int* const pinned_control_flag = ${pinned_control_flag};
    double* const pmaswt = ${pmaswt};
    constexpr double pssmin = ${pssmin};
    double2* const pu0 = ${pu0};
    // const double2* const pu = ${pu};
    double2* const pu = ${pu};
    // const double2* const pxp = ${pxp};
    double2* const pxp = ${pxp};
    // const double2* const px = ${px};
    double2* const px = ${px};
    constexpr double q1 = ${q1};
    constexpr double q2 = ${q2};
    constexpr double qgamma = ${qgamma};
    int* const remaining_wg = ${remaining_wg};
    const int* const schsfirst = ${schsfirst};
    const int* const schslast = ${schslast};
    double2* const sfpq = ${sfpq};
    const double* const smf = ${smf};
    constexpr double talfa = ${talfa};
    constexpr double tssmin = ${tssmin};
    constexpr double2 vfixx = ${vfixx};
    constexpr double2 vfixy = ${vfixy};
    double* const zdl = ${zdl};
    double* const zarea = ${zarea};
    double* const zdu = ${zdu};
    double* const ze = ${ze};
    double* const zetot = ${zetot};
    double* const zm = ${zm};
    const int* const znump = ${znump};
    double* const zp = ${zp};
    double* const zr = ${zr};
    double* const zss = ${zss};
    double* const zvol0 = ${zvol0};
    double* const zvol = ${zvol};
    double* const zwrate = ${zwrate};
  }


  //-- gpuMain1 ----------------------------------------------------------
  __launch_bounds__(256)
  __global__ void gpuMain1_jit(double dt)
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


  //-- gpuMain2 and supporting device functions -------------------------

  __device__ OPT_INLINE
  void calcZoneCtrs_SideVols_ZoneVols_jit(
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

  __device__ OPT_INLINE
  void pgasCalcStateAtHalf_jit(const int z,
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


  __device__ OPT_INLINE
  void ttsCalcForce_jit(const int s,
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

  __device__ OPT_INLINE
  void qcsSetCornerDiv_jit(const int s,
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

  __device__ OPT_INLINE
  void qcsSetVelDiff_jit(
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

#if defined(__gfx908__) or defined(__HIP_ARCH_GFX908__)
  __launch_bounds__(64,4)
#else
  __launch_bounds__(64)
#endif
  __global__ void gpuMain2_jit(double dt)
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

  //-- gpuMain3 and supporting device functions -------------------------

  __device__ OPT_INLINE
  void applyFixedBC_jit(const int p,
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

  __device__ OPT_INLINE
  void localReduceToPoints_jit(const int p,
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

  __launch_bounds__(256)
  __global__ void localReduceToPoints_k_jit()
  {
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    // sum corner masses, forces to points
    localReduceToPoints_jit(p, cmaswt, pmaswt, cftot, pf);
  }
  
  __launch_bounds__(256)
  __global__ void gpuMain3_jit(double dt)
  {
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    if(doLocalReduceToPoints){
      // sum corner masses, forces to points
      localReduceToPoints_jit(p, cmaswt, pmaswt, cftot, pf);
    }

    // 4a. apply boundary conditions
    double2 pxpp = pxp[p];
    double2 pu0p = pu0[p];
    double2 pfp  = pf[p];
    // bcx and bcy are arrays of size 2 in constant memory. Need to think how to
    // deal with that, For now, manually unroll.
    /*
      for (int bc = 0; bc < numbcx; ++bc)
      applyFixedBC(p, pxpp, pu0p, pfp, vfixx, bcx[bc]);
      for (int bc = 0; bc < numbcy; ++bc)
      applyFixedBC(p, pxpp, pu0p, pfp, vfixy, bcy[bc]);
    */
    applyFixedBC_jit(p, pxpp, pu0p, pfp, vfixx, bcx0);
    applyFixedBC_jit(p, pxpp, pu0p, pfp, vfixx, bcx1);
    applyFixedBC_jit(p, pxpp, pu0p, pfp, vfixy, bcy0);
    applyFixedBC_jit(p, pxpp, pu0p, pfp, vfixy, bcy1);

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

  //-- gpuMain4 and supporting device functions -------------------------

  __device__ void calcZoneCtrs_SideVols_ZoneVols_main4_jit(
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

  __device__ void hydroCalcWork_jit(
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

  __launch_bounds__(256)
  __global__ void gpuMain4_jit(double dt)
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
    calcZoneCtrs_SideVols_ZoneVols_main4_jit(s,s0,z, pxp1, pxp2, zarea, zvol_1,dss4, ctemp2, ctemp, ctemp1, numsbad_pinned);

    // 7. compute work
    double zwz;
    hydroCalcWork_jit(s, s0, z, p1, p2, sfpq, pu0, pu, pxp, dt,
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

  //-- gpuMain5 and supporting device functions -------------------------
  __device__ void hydroCalcDtCourant_jit(
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

  __device__ void hydroCalcDtVolume_jit(
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


  __device__ double atomicMin_jit(double* address, double val)
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


  __device__ void hydroFindMinDt_jit(
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
      atomicMin_jit(&(dtnext->d), ctemp[0]);
      // This line isn't 100% thread-safe, but since it is only for
      // a debugging aid, I'm not going to worry about it.
      if (dtnext->d == ctemp[0]) dtnext->i = ctempi[0];
    }
    if(threadIdx.x == 0){
      int old = atomicSub(remaining_wg, 1);
      bool this_wg_is_last = (old == 1);
      if(this_wg_is_last){
	// force reloading of dtnext->d from L2 into register
	atomicMin_jit(&(dtnext->d), ctemp[0]);
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


  __device__ void hydroCalcDt_jit(
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
    hydroCalcDtCourant_jit(z, zdu, zss, zdl, dtz, idtz);
    hydroCalcDtVolume_jit(z, zvol, zvol0, dtlast, dtz, idtz);
    hydroFindMinDt_jit(z, z0, zlength, dtz, idtz, ctemp, ctemp2, dtnext,
		       dtnext_H, remaining_wg, pinned_control_flag);
  }

  __launch_bounds__(256)
  __global__ void gpuMain5_jit(double dt)
  {
    const int z = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (z >= numz) return;

    const int z0 = threadIdx.x;
    const int zlength = min((int)CHUNK_SIZE, (int)(numz - blockIdx.x * CHUNK_SIZE));

    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];

    // compute timestep for next cycle
    hydroCalcDt_jit(z, z0, zlength, zdu, zss, zdl, zvol, zvol0, dt,
		    ctemp, ctemp2, dtnext, dtnext_H, remaining_wg, pinned_control_flag);
  }

} // extern "C"
