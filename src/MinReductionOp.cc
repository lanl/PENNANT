/*
 * MinReductionOp.cc
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

#include "MinReductionOp.hh"



const TimeStep MinReductionOp::identity = TimeStep();

template<>
void
MinReductionOp::apply<true>(LHS &lhs, RHS rhs) {
    lhs.dt = std::min(lhs.dt, rhs.dt);
    if (lhs.dt == rhs.dt)
    	snprintf(lhs.message, 80, "%s", rhs.message);
}

template<>
void
MinReductionOp::apply<false>(LHS &lhs, RHS rhs) {
    int64_t *target = (int64_t *)&lhs.dt;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T = std::min(oldval.as_T, rhs.dt);
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
    if (lhs.dt == rhs.dt)
    	snprintf(lhs.message, 80, "%s", rhs.message);
}

template<>
void
MinReductionOp::fold<true>(RHS &rhs1, RHS rhs2) {
    rhs1.dt = std::min(rhs1.dt, rhs2.dt);
    if (rhs1.dt == rhs2.dt)
    	snprintf(rhs1.message, 80, "%s", rhs2.message);
}

template<>
void
MinReductionOp::fold<false>(RHS &rhs1, RHS rhs2) {
    int64_t *target = (int64_t *)&rhs1.dt;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T = std::min(oldval.as_T, rhs2.dt);
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
    if (rhs1.dt == rhs2.dt)
    	snprintf(rhs1.message, 80, "%s", rhs2.message);
}

