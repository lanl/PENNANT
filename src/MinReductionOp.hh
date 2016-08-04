/**
 * Copyright (c) 2014      Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * LA-CC 10-123
 */

#ifndef MINREDUCTIONOP_HH
#define MINREDUCTIONOP_HH

#include "Parallel.hh"

/**
 *
 */

const TimeStep MinReductionOp::identity = TimeStep();

template<>
void
MinReductionOp::apply<true>(LHS &lhs, RHS rhs) {
    lhs.dt_ = std::min(lhs.dt_, rhs.dt_);
    if (lhs.dt_ == rhs.dt_)
    	snprintf(lhs.message_, 80, "%s", rhs.message_);
}

template<>
void
MinReductionOp::apply<false>(LHS &lhs, RHS rhs) {
    int64_t *target = (int64_t *)&lhs.dt_;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T = std::min(oldval.as_T, rhs.dt_);
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
    if (lhs.dt_ == rhs.dt_)
    	snprintf(lhs.message_, 80, "%s", rhs.message_);
}

template<>
void
MinReductionOp::fold<true>(RHS &rhs1, RHS rhs2) {
    rhs1.dt_ = std::min(rhs1.dt_, rhs2.dt_);
    if (rhs1.dt_ == rhs2.dt_)
    	snprintf(rhs1.message_, 80, "%s", rhs2.message_);
}

template<>
void
MinReductionOp::fold<false>(RHS &rhs1, RHS rhs2) {
    int64_t *target = (int64_t *)&rhs1.dt_;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
        oldval.as_int = *target;
        newval.as_T = std::min(oldval.as_T, rhs2.dt_);
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
    if (rhs1.dt_ == rhs2.dt_)
    	snprintf(rhs1.message_, 80, "%s", rhs2.message_);
}


#endif
