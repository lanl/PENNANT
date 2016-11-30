/*
 * LogicalStruc.cc
 *
 *  Created on: Sep 8, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */


#include "LogicalStructured.hh"

#include "Vec2.hh"


LogicalStructured::LogicalStructured(Context ctx, HighLevelRuntime *runtime) :
    LogicalUnstructured(ctx, runtime),
    nElements(-1)
{
}


LogicalStructured::LogicalStructured(Context ctx, HighLevelRuntime *runtime, PhysicalRegion pregion) :
    LogicalUnstructured(ctx, runtime, pregion),
    nElements(-2)
{
    Domain Dom = runtime->get_index_space_domain(ctx, ispace);
    Rect<1> rect = Dom.get_rect<1>();
    nElements = static_cast<int>(rect.volume());
}

void LogicalStructured::allocate(int nStrucs)
{
    assert( (nStrucs > 0) && (fIDs.size() > 0) && (!pregion.is_mapped())
            && (ispaceID == nullptr) && (lregionID == nullptr) );

    nElements = nStrucs;

    Rect<1> rect(Point<1>(0),Point<1>(nStrucs - 1));
    ispace = runtime->create_index_space(ctx, Domain::from_rect<1>(rect));
    destroy_ispace = true;
    char buf[43];
    sprintf(buf, "LogicalStruc::iSpace %d", nStrucs);
    runtime->attach_name(ispace, buf);
    ispaceID = new IndexSpaceID;
    *ispaceID = ispace.get_id();
    LogicalUnstructured::allocate();
}

/**
 * courtesy of some other legion code.
 */
template <unsigned DIM, typename T>
static inline bool
offsetsAreDense(const Rect<DIM> &bounds,
                const LegionRuntime::Accessor::ByteOffset *offset)
{
    off_t exp_offset = sizeof(T);
    for (unsigned i = 0; i < DIM; i++) {
        bool found = false;
        for (unsigned j = 0; j < DIM; j++)
            if (offset[j].offset == exp_offset) {
                found = true;
                exp_offset *= (bounds.hi[j] - bounds.lo[j] + 1);
                break;
            }
        if (!found) return false;
    }
    return true;
}


template <>
ptr_t* LogicalStructured::getRawPtr<ptr_t>(FieldID FID)
{
    getPRegion();

    ptr_t *mData = nullptr;

    PtrTAccessor tAcc = pregion.get_field_accessor(FID).typeify<ptr_t>();
    Domain tDom = runtime->get_index_space_domain(
                ctx, ispace);
    Rect<1> subrect;
    ByteOffset inOffsets[1];
    auto subGridBounds = tDom.get_rect<1>();

    mData = tAcc.template raw_rect_ptr<1>(
                subGridBounds, subrect, inOffsets);

    // Sanity.
    if (!mData || (subrect != subGridBounds) ||
            !offsetsAreDense<1, ptr_t>(subGridBounds, inOffsets)) {
        // Signifies that something went south.
        mData = nullptr;
        assert(mData != nullptr);
    }
    return mData;
}


template <>
double2* LogicalStructured::getRawPtr<double2>(FieldID FID)
{
    getPRegion();

    double2 *mData = nullptr;

    Double2Accessor tAcc = pregion.get_field_accessor(FID).typeify<double2>();
    Domain tDom = runtime->get_index_space_domain(
                ctx, ispace);
    Rect<1> subrect;
    ByteOffset inOffsets[1];
    auto subGridBounds = tDom.get_rect<1>();

    mData = tAcc.template raw_rect_ptr<1>(
                subGridBounds, subrect, inOffsets);

    // Sanity.
    if (!mData || (subrect != subGridBounds) ||
            !offsetsAreDense<1, double2>(subGridBounds, inOffsets)) {
        // Signifies that something went south.
        mData = nullptr;
        assert(mData != nullptr);
    }
    return mData;
}


template <>
double* LogicalStructured::getRawPtr<double>(FieldID FID)
{
    getPRegion();

    double *mData = nullptr;

    DoubleAccessor tAcc = pregion.get_field_accessor(FID).typeify<double>();
    Domain tDom = runtime->get_index_space_domain(
                ctx, ispace);
    Rect<1> subrect;
    ByteOffset inOffsets[1];
    auto subGridBounds = tDom.get_rect<1>();

    mData = tAcc.template raw_rect_ptr<1>(
                subGridBounds, subrect, inOffsets);

    // Sanity.
    if (!mData || (subrect != subGridBounds) ||
            !offsetsAreDense<1, double>(subGridBounds, inOffsets)) {
        // Signifies that something went south.
        mData = nullptr;
        assert(mData != nullptr);
    }
    return mData;
}


template <>
int* LogicalStructured::getRawPtr<int>(FieldID FID)
{
    getPRegion();

    int *mData = nullptr;

    IntAccessor tAcc = pregion.get_field_accessor(FID).typeify<int>();
    Domain tDom = runtime->get_index_space_domain(
                ctx, ispace);
    Rect<1> subrect;
    ByteOffset inOffsets[1];
    auto subGridBounds = tDom.get_rect<1>();

    mData = tAcc.template raw_rect_ptr<1>(
                subGridBounds, subrect, inOffsets);

    // Sanity.
    if (!mData || (subrect != subGridBounds) ||
            !offsetsAreDense<1, int>(subGridBounds, inOffsets)) {
        // Signifies that something went south.
        mData = nullptr;
        assert(mData != nullptr);
    }
    return mData;
}
