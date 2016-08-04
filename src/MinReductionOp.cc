/*
 * MinReductionOp.cc
 *
 *  Created on: Aug 4, 2016
 *      Author: jgraham
 */

#include "MinReductionOp.hh"



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
