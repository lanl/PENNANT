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

#include "AddReductionOp.hh"

const double AddReductionOp::identity = 0.0;

template<>
void
AddReductionOp::apply<true>(LHS &lhs, RHS rhs) {
    lhs += rhs;
}

template<>
void
AddReductionOp::apply<false>(LHS &lhs, RHS rhs) {
    int64_t *target = (int64_t *)&lhs;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T = oldval.as_T + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template<>
void
AddReductionOp::fold<true>(RHS &rhs1, RHS rhs2) {
    rhs1 += rhs2;
}

template<>
void
AddReductionOp::fold<false>(RHS &rhs1, RHS rhs2) {
    int64_t *target = (int64_t *)&rhs1;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T = oldval.as_T + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

