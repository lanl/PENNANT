/*
 * AddReductionOp.cc
 *
 *  Created on: Aug 4, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#include "Add2ReductionOp.hh"

const double2 Add2ReductionOp::identity = double2();

struct simpledouble2 {
    double x,y;
};

template<>
void
Add2ReductionOp::apply<true>(LHS &lhs, RHS rhs) {
    lhs += rhs;
}

template<>
void
Add2ReductionOp::apply<false>(LHS &lhs, RHS rhs) {
    __int128 *target = (__int128 *)&lhs;
    union { __int128 as_int; simpledouble2 as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T.x = oldval.as_T.x + rhs.x;
        newval.as_T.y = oldval.as_T.y + rhs.y;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template<>
void
Add2ReductionOp::fold<true>(RHS &rhs1, RHS rhs2) {
    rhs1 += rhs2;
}

template<>
void
Add2ReductionOp::fold<false>(RHS &rhs1, RHS rhs2) {
    __int128 *target = (__int128 *)&rhs1;
    union { __int128 as_int; simpledouble2 as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T.x = oldval.as_T.x + rhs2.x;
        newval.as_T.y = oldval.as_T.y + rhs2.y;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}
 
