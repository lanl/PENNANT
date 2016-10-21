/*
 * AddInt64ReductionOp.cc
 *
 *  Created on: Oct 20, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#include "AddInt64ReductionOp.hh"

const int64_t AddInt64ReductionOp::identity = 0;

template<>
void
AddInt64ReductionOp::apply<true>(LHS &lhs, RHS rhs) {
    lhs += rhs;
}

template<>
void
AddInt64ReductionOp::apply<false>(LHS &lhs, RHS rhs) {
    int64_t *target = (int64_t *)&lhs;
    int64_t  oldval, newval;
    do {
        oldval = *target;
        newval = oldval + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
}

template<>
void
AddInt64ReductionOp::fold<true>(RHS &rhs1, RHS rhs2) {
    rhs1 += rhs2;
}

template<>
void
AddInt64ReductionOp::fold<false>(RHS &rhs1, RHS rhs2) {
    int64_t *target = (int64_t *)&rhs1;
    int64_t oldval, newval;
    do {
        oldval = *target;
        newval = oldval + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
}

