/*
 * LogicalUnstruc.cc
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


#include "LogicalUnstructured.hh"

#include "Vec2.hh"


LogicalUnstructured::LogicalUnstructured(Context ctx, HighLevelRuntime *runtime,
        IndexSpace i_space) :
    LogicalUnstructured(ctx, runtime)
{
    ispace = i_space;
    ispaceID = new IndexSpaceID;
    *ispaceID = ispace.get_id();
}


LogicalUnstructured::LogicalUnstructured(Context ctx, HighLevelRuntime *runtime,
        PhysicalRegion pregion) :
    destroy_ispace(false),
    ispace(pregion.get_logical_region().get_index_space()),
    ipartID(nullptr),
    destroy_fspace(false),
    fspace(pregion.get_logical_region().get_field_space()),
    destroy_lregion(false),
    lregion(pregion.get_logical_region()),
    lpartID(nullptr),
    pregion(pregion),
    ctx(ctx),
    runtime(runtime)
{
    ispaceID = new IndexSpaceID;
    *ispaceID = pregion.get_logical_region().get_index_space().get_id();
    fspaceID = new FieldSpaceID;
    *fspaceID = pregion.get_logical_region().get_field_space().get_id();
    lregionID = new RegionTreeID;
    *lregionID = pregion.get_logical_region().get_tree_id();
}


LogicalUnstructured::LogicalUnstructured(Context ctx, HighLevelRuntime *runtime,
        LogicalRegion lregion) :
    destroy_ispace(false),
    ispace(lregion.get_index_space()),
    ipartID(nullptr),
    destroy_fspace(false),
    fspace(lregion.get_field_space()),
    destroy_lregion(false),
    lregion(lregion),
    lpartID(nullptr),
    ctx(ctx),
    runtime(runtime)
{
    ispaceID = new IndexSpaceID;
    *ispaceID = lregion.get_index_space().get_id();
    fspaceID = new FieldSpaceID;
    *fspaceID = lregion.get_field_space().get_id();
    lregionID = new RegionTreeID;
    *lregionID = lregion.get_tree_id();
}


LogicalUnstructured::LogicalUnstructured(Context ctx, HighLevelRuntime *runtime) :
    destroy_ispace(false),
    ispaceID(nullptr),
    ipartID(nullptr),
    destroy_fspace(true),
    destroy_lregion(false),
    lregionID(nullptr),
    lpartID(nullptr),
    ctx(ctx),
    runtime(runtime)
{
    fspace = runtime->create_field_space(ctx);
    runtime->attach_name(fspace, "LogicalUnstruc::fSpace");
    fspaceID = new FieldSpaceID;
    *fspaceID = fspace.get_id();
}


LogicalUnstructured::~LogicalUnstructured() {
    if (destroy_lregion)
        runtime->destroy_logical_region(ctx, lregion);
    if (destroy_fspace)
        runtime->destroy_field_space(ctx, fspace);
    if (destroy_ispace)
        runtime->destroy_index_space(ctx, ispace);
}


void LogicalUnstructured::partition(Coloring map, bool disjoint)
{
    assert( (ispaceID != nullptr) && (lpartID == nullptr) );
    ipart = runtime->create_index_partition(ctx, ispace, map, disjoint);
    runtime->attach_name(ipart, "LogicalUnstruc::part");
    ipartID = new IndexPartitionID;
    *ipartID = ipart.id;
    lpart = runtime->get_logical_partition(ctx, lregion, ipart);
    lpartID = new RegionTreeID;
    *lpartID = lpart.get_tree_id();
}


IndexSpace LogicalUnstructured::getSubspace(Color color)
{
    assert( (ispaceID != nullptr) && (ipartID != nullptr) );
    subspace = runtime->get_index_subspace(ctx, ipart, color);
    char buf[43];
    sprintf(buf, "LogicalUnstruc::iPart %d", color);
    runtime->attach_name(subspace, buf);
    return subspace;
}


void LogicalUnstructured::allocate(int nUnstrucs)
{
    assert( (nUnstrucs > 0) && (!pregion.is_mapped())
            && (ispaceID == nullptr) && (lregionID == nullptr) );

    ispace = runtime->create_index_space(ctx, nUnstrucs);
    destroy_ispace = true;
    char buf[43];
    sprintf(buf, "LogicalUnstruc::iSpace %d", nUnstrucs);
    runtime->attach_name(ispace, buf);
    IndexAllocator allocator = runtime->create_index_allocator(ctx, ispace);
    ptr_t begin = allocator.alloc(nUnstrucs);
    assert(!begin.is_null());
    ispaceID = new IndexSpaceID;
    *ispaceID = ispace.get_id();
    if (fIDs.size() > 0)
        allocate();
}


void LogicalUnstructured::allocate()
{
    assert( (fIDs.size() > 0) && (!pregion.is_mapped())
            && (ispaceID != nullptr) && (lregionID == nullptr) );
    lregion = runtime->create_logical_region(ctx, ispace, fspace);
    destroy_lregion = true;
    runtime->attach_name(lregion, "LogicalUnstruc::lRegion");
    lregionID = new RegionTreeID;
    *lregionID = lregion.get_tree_id();
}


void LogicalUnstructured::addField(FieldID FID)
{
    assert(lregionID == nullptr);
    fIDs.push_back(FID);
}


template <>
void LogicalUnstructured::addField<ptr_t>(FieldID FID)
{
    addField(FID);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fspace);
    allocator.allocate_field(sizeof(ptr_t), FID);
}


template <>
void LogicalUnstructured::addField<double2>(FieldID FID)
{
    addField(FID);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fspace);
    allocator.allocate_field(sizeof(double2), FID);
}


template <>
void LogicalUnstructured::addField<double>(FieldID FID)
{
    addField(FID);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fspace);
    allocator.allocate_field(sizeof(double), FID);
}


template <>
void LogicalUnstructured::addField<int>(FieldID FID)
{
    addField(FID);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fspace);
    allocator.allocate_field(sizeof(int), FID);
}


void LogicalUnstructured::unMapPRegion()
{
    if (pregion.is_mapped())
        runtime->unmap_region(ctx, pregion);
}


PhysicalRegion LogicalUnstructured::getPRegion()
{
    assert(lregionID != nullptr);
    if (!pregion.is_mapped()) {
        RegionRequirement req(lregion, WRITE_DISCARD, EXCLUSIVE, lregion);
        for (int i=0; i<fIDs.size(); i++)
            req.add_field(fIDs[i]);
        InlineLauncher launcher(req);
        pregion = runtime->map_region(ctx, launcher);
        pregion.wait_until_valid(); // maybe don't trust is_mapped()
    }
    return pregion;
}


template <>
Double2SOAAccessor LogicalUnstructured::getRegionSOAAccessor<double2>(FieldID FID)
{
    getPRegion();
    Double2Accessor generic =
            pregion.get_field_accessor(FID).typeify<double2>();
    assert(generic.can_convert<AccessorType::SOA<0>>());
    return generic.convert<AccessorType::SOA<sizeof(double2)>>();
}


template <>
DoubleSOAAccessor LogicalUnstructured::getRegionSOAAccessor<double>(FieldID FID)
{
    getPRegion();
    DoubleAccessor generic =
            pregion.get_field_accessor(FID).typeify<double>();
    assert(generic.can_convert<AccessorType::SOA<0>>());
    return generic.convert<AccessorType::SOA<sizeof(double)>>();
}


template <>
IntSOAAccessor LogicalUnstructured::getRegionSOAAccessor<int>(FieldID FID)
{
    getPRegion();
    IntAccessor generic =
            pregion.get_field_accessor(FID).typeify<int>();
    assert(generic.can_convert<AccessorType::SOA<0>>());
    return generic.convert<AccessorType::SOA<sizeof(int)>>();
}
