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


class QCS {
public:

    static void calcForce(
            double2* sf,
            const int sfirst,
            const int slast,
            const int nums,
            const int numz,
            const double2* pu,
            const double2* ex,
            const double2* zx,
            const double* elen,
            const int* map_side2zone,
            const int* map_side2pt1,
            const int* map_side2pt2,
            const int* zone_pts_ptr,
            const int* map_side2edge,
            const double2* pt_x_pred,
            const double* zrp,
            const double* zss,
            const double qgamma,
            const double q1,
            const double q2,
            int zfirst,
            int zlast,
            double* zdu);

    static void setCornerDiv(
            double* c0area,
            double* c0div,
            double* c0evol,
            double* c0du,
            double* c0cos,
            const int sfirst,
            const int slast,
            const int nums,
            const int numz,
            const double2* pu,
            const double2* ex,
            const double2* zx,
            const double* elen,
            const int* map_side2zone,
            const int* map_side2pt1,
            const int* map_side2pt2,
            const int* zone_pts_ptr,
            const double2* pt_x_pred,
            const int* map_side2edge);

    static void setQCnForce(
            const double* c0div,
            const double* c0du,
            const double* c0evol,
            double2* c0qe,
            const int sfirst,
            const int slast,
            const double2* pu,
            const double* zrp,
            const double* zss,
            const double* elen,
            const double qgamma,
            const double q1,
            const double q2,
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const int* map_side2pt1,
            const int* map_side2pt2,
            const int* map_side2edge);

    static void setForce(
            const double* c0area,
            const double2* c0qe,
            double* c0cos,
            double2* sfqq,
            const int sfirst,
            const int slast,
            const double* elen,
            const int* map_side2zone,
            const int* zone_pts_ptr,
            const int* map_side2edge);

    static void setVelDiff(
            const int sfirst,
            const int slast,
            const int nums,
            const int numz,
            int zfirst,
            int zlast,
            const double2* pu,
            const double* zss,
            double* zdu,
            const double* elen,
            const double2* pt_x_pred,
            const int* map_side2pt1,
            const int* map_side2pt2,
            const int* map_side2zone,
            const int* map_side2edge,
            const int* zone_pts_ptr,
            const double q1,
            const double q2);

};  // class QCS


#endif /* QCS_HH_ */
