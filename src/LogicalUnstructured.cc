/*
 * LogicalUnstructured.cc
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

#include <cstdio>
#include <iostream>

#include "Vec2.hh"

LogicalUnstructured::LogicalUnstructured(Context ctx, Runtime* runtime,
    IndexSpace i_space, string label) :
      LogicalUnstructured(ctx, runtime, label) {
  ispace = i_space;
  ispaceID = new IndexSpaceID;
  *ispaceID = ispace.get_id();
}

LogicalUnstructured::LogicalUnstructured(Context ctx, Runtime* runtime,
    PhysicalRegion pregion, string label) :
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
      runtime(runtime),
      name(label) {
  ispaceID = new IndexSpaceID;
  *ispaceID = pregion.get_logical_region().get_index_space().get_id();
  fspaceID = new FieldSpaceID;
  *fspaceID = pregion.get_logical_region().get_field_space().get_id();
  lregionID = new RegionTreeID;
  *lregionID = pregion.get_logical_region().get_tree_id();
}

LogicalUnstructured::LogicalUnstructured(Context ctx, Runtime* runtime,
    LogicalRegion lregion, string label) :
      destroy_ispace(false),
      ispace(lregion.get_index_space()),
      ipartID(nullptr),
      destroy_fspace(false),
      fspace(lregion.get_field_space()),
      destroy_lregion(false),
      lregion(lregion),
      lpartID(nullptr),
      ctx(ctx),
      runtime(runtime),
      name(label) {
  ispaceID = new IndexSpaceID;
  *ispaceID = lregion.get_index_space().get_id();
  fspaceID = new FieldSpaceID;
  *fspaceID = lregion.get_field_space().get_id();
  lregionID = new RegionTreeID;
  *lregionID = lregion.get_tree_id();
}

LogicalUnstructured::LogicalUnstructured(Context ctx, Runtime* runtime,
    string label) :
      destroy_ispace(false),
      ispaceID(nullptr),
      ipartID(nullptr),
      destroy_fspace(true),
      destroy_lregion(false),
      lregionID(nullptr),
      lpartID(nullptr),
      ctx(ctx),
      runtime(runtime),
      name(label) {
  fspace = runtime->create_field_space(ctx);
  runtime->attach_name(fspace, (name + " fSpace").c_str());
  fspaceID = new FieldSpaceID;
  *fspaceID = fspace.get_id();
}

LogicalUnstructured::~LogicalUnstructured() {
  if (destroy_lregion) runtime->destroy_logical_region(ctx, lregion);
  if (destroy_fspace) runtime->destroy_field_space(ctx, fspace);
  if (destroy_ispace) runtime->destroy_index_space(ctx, ispace);
}

void LogicalUnstructured::partition(Coloring map, bool disjoint) {
  assert((ispaceID != nullptr) && (lpartID == nullptr));
  ipart = runtime->create_index_partition(ctx, ispace, map, disjoint);
  runtime->attach_name(ipart, (name + " part").c_str());
  ipartID = new IndexPartitionID;
  *ipartID = ipart.id;
  lpart = runtime->get_logical_partition(ctx, lregion, ipart);
  lpartID = new RegionTreeID;
  *lpartID = lpart.get_tree_id();
}

IndexSpace LogicalUnstructured::getSubspace(Color color) {
  assert((ispaceID != nullptr) && (ipartID != nullptr));
  subspace = runtime->get_index_subspace(ctx, ipart, color);
  char buf[43];
  sprintf(buf, " iPart %d", color);
  runtime->attach_name(subspace, (name + buf).c_str());
  return subspace;
}

void LogicalUnstructured::allocate(int nUnstrucs) {
  assert(
    (nUnstrucs > 0) && (!pregion.is_mapped()) && (ispaceID == nullptr)
    && (lregionID == nullptr));

  ispace = runtime->create_index_space(ctx, nUnstrucs);
  destroy_ispace = true;
  char buf[43];
  sprintf(buf, " iSpace %d", nUnstrucs);
  runtime->attach_name(ispace, (name + buf).c_str());
  IndexAllocator allocator = runtime->create_index_allocator(ctx, ispace);
  ptr_t begin = allocator.alloc(nUnstrucs);
  assert(!begin.is_null());
  ispaceID = new IndexSpaceID;
  *ispaceID = ispace.get_id();
  if (fIDs.size() > 0) allocate();
}

void LogicalUnstructured::allocate() {
  assert(
    (fIDs.size() > 0) && (!pregion.is_mapped()) && (ispaceID != nullptr)
    && (lregionID == nullptr));
  lregion = runtime->create_logical_region(ctx, ispace, fspace);
  destroy_lregion = true;
  runtime->attach_name(lregion, (name + " LR").c_str());
  lregionID = new RegionTreeID;
  *lregionID = lregion.get_tree_id();
#if MESH_DEBUG
  const char* buffer;
  runtime->retrieve_name(lregion, buffer);
  std::cout << name << " just created a new LR: " << buffer << std::endl;
  runtime->retrieve_name(ispace, buffer);
  std::cout << "  by crossing " << buffer;
  runtime->retrieve_name(fspace, buffer);
  std::cout << " with " << fspace << std::endl;
  std::cout << "  which has these fields:" << std::endl;
  for (auto f : fIDs) {
    runtime->retrieve_name(fspace, f, buffer);
    std::cout << "    " << buffer << std::endl;
  }
#endif
}

template<typename T>
void LogicalUnstructured::addField(FieldID FID, const char* name) {
  assert(lregionID == nullptr);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fspace);
  allocator.allocate_field(sizeof(T), FID);
  runtime->attach_name(fspace, FID, name);
  fIDs.push_back(FID);
}

template void LogicalUnstructured::addField<ptr_t>(FieldID, const char*);
template void LogicalUnstructured::addField<int>(FieldID, const char*);
template void LogicalUnstructured::addField<double>(FieldID, const char*);
template void LogicalUnstructured::addField<double2>(FieldID, const char*);

void LogicalUnstructured::unMapPRegion() {
  if (pregion.is_mapped()) runtime->unmap_region(ctx, pregion);
}

PhysicalRegion LogicalUnstructured::getPRegion(legion_privilege_mode_t priv) {
  assert(lregionID != nullptr);
  if (!pregion.is_mapped()) {
    RegionRequirement req(lregion, priv, EXCLUSIVE, lregion);
    for (int i = 0; i < fIDs.size(); i++)
      req.add_field(fIDs[i]);
    InlineLauncher launcher(req);
    pregion = runtime->map_region(ctx, launcher);
    pregion.wait_until_valid();  // maybe don't trust is_mapped()
  }
  return pregion;
}

template<>
Double2SOAAccessor LogicalUnstructured::getRegionSOAAccessor<double2>(
    FieldID FID) {
  getPRegion();
  Double2Accessor generic = pregion.get_field_accessor(FID).typeify<double2>();
  assert(generic.can_convert<AccessorType::SOA<0>>());
  return generic.convert<AccessorType::SOA<sizeof(double2)>>();
}

template<>
DoubleSOAAccessor LogicalUnstructured::getRegionSOAAccessor<double>(
    FieldID FID) {
  getPRegion();
  DoubleAccessor generic = pregion.get_field_accessor(FID).typeify<double>();
  assert(generic.can_convert<AccessorType::SOA<0>>());
  return generic.convert<AccessorType::SOA<sizeof(double)>>();
}

template<>
IntSOAAccessor LogicalUnstructured::getRegionSOAAccessor<int>(FieldID FID) {
  getPRegion();
  IntAccessor generic = pregion.get_field_accessor(FID).typeify<int>();
  assert(generic.can_convert<AccessorType::SOA<0>>());
  return generic.convert<AccessorType::SOA<sizeof(int)>>();
}
