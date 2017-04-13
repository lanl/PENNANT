/*
 * QCS.hh
 *
 *  Created on: Feb 21, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef QCS_HH_
#define QCS_HH_

#include "Vec2.hh"
#include "Memory.hh"

class QCS {
public:

  // This structure holds the pointers to temporary spaces used by the functions
  class Temp {
  public:
    Temp(int csize, int zsize) {
      c0area = AbstractedMemory::alloc<double>(csize);
      c0evol = AbstractedMemory::alloc<double>(csize);
      c0du = AbstractedMemory::alloc<double>(csize);
      c0div = AbstractedMemory::alloc<double>(csize);
      c0cos = AbstractedMemory::alloc<double>(csize);
      c0qe = AbstractedMemory::alloc<double2>(2 * csize);
      z0uc = AbstractedMemory::alloc<double2>(zsize);
      z0tmp = AbstractedMemory::alloc<double>(zsize);
    }
    ~Temp() {
      AbstractedMemory::free(c0area);
      AbstractedMemory::free(c0evol);
      AbstractedMemory::free(c0du);
      AbstractedMemory::free(c0div);
      AbstractedMemory::free(c0cos);
      AbstractedMemory::free(c0qe);
      AbstractedMemory::free(z0uc);
      AbstractedMemory::free(z0tmp);
    }

    double* c0area;
    double* c0evol;
    double* c0du;
    double* c0div;
    double* c0cos;
    double2* c0qe;

    double2* z0uc;
    double* z0tmp;
  };

  static void calcForce(double2* sf, const int sfirst, const int slast,
      const int nums, const int numz, const double2* pu, const double2* ex,
      const double2* zx, const double* elen, const int* map_side2zone,
      const int* map_side2pt1, const int* map_side2pt2, const int* zone_pts_ptr,
      const int* map_side2edge, const double2* pt_x_pred, const double* zrp,
      const double* zss, const double qgamma, const double q1, const double q2,
      int zfirst, int zlast, double* zdu, const Temp& temp);

  static void setCornerDiv(const int sfirst, const int slast, const int nums,
      const int numz, const double2* pu, const double2* ex, const double2* zx,
      const double* elen, const int* map_side2zone, const int* map_side2pt1,
      const int* map_side2pt2, const int* zone_pts_ptr,
      const double2* pt_x_pred, const int* map_side2edge, const Temp& temp);

  static void setQCnForce(const int sfirst, const int slast, const double2* pu,
      const double* zrp, const double* zss, const double* elen,
      const double qgamma, const double q1, const double q2,
      const int* map_side2zone, const int* zone_pts_ptr,
      const int* map_side2pt1, const int* map_side2pt2,
      const int* map_side2edge, const Temp& temp);

  static void setForce(double2* sfqq, const int sfirst, const int slast,
      const double* elen, const int* map_side2zone, const int* zone_pts_ptr,
      const int* map_side2edge, const Temp& temp);

  static void setVelDiff(const int sfirst, const int slast, const int nums,
      const int numz, int zfirst, int zlast, const double2* pu,
      const double* zss, double* zdu, const double* elen,
      const double2* pt_x_pred, const int* map_side2pt1,
      const int* map_side2pt2, const int* map_side2zone,
      const int* map_side2edge, const int* zone_pts_ptr, const double q1,
      const double q2, const Temp& temp);

private:
  static int iter;

};
// class QCS

#endif /* QCS_HH_ */
