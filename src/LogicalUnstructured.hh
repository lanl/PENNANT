/*
 * LogicalElement.hh
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

#ifndef SRC_LOGICALUNSTRUCTURED_HH_
#define SRC_LOGICALUNSTRUCTURED_HH_

#include <cassert>
#include <string>
#include <vector>

#include "Parallel.hh"

using std::string;

class LogicalUnstructured {
public:
  LogicalUnstructured(Context ctx, Runtime* runtime, string label =
      "LogUnstruct");
  LogicalUnstructured(Context ctx, Runtime* runtime, IndexSpace ispace,
      string label = "LogUnstruct");
  LogicalUnstructured(Context ctx, Runtime* runtime, PhysicalRegion pregion,
      string label = "LogUnstruct");
  LogicalUnstructured(Context ctx, Runtime* runtime, LogicalRegion lregion,
      string label = "LogUnstruct");
  virtual ~LogicalUnstructured();
  template<typename T>
  void addField(FieldID FID, const char* name);
  virtual void allocate(int nElements);
  void allocate();
  void partition(Coloring map, bool disjoint);
  template<typename TYPE>
  RegionAccessor<AccessorType::SOA<sizeof(TYPE)>, TYPE> getRegionSOAAccessor(
      FieldID FID);
  IndexIterator getIterator() const {
    assert(ispaceID != nullptr);
    return IndexIterator(runtime, ctx, ispace);
  }
  IndexSpace getISpace() const {
    assert(ispaceID != nullptr);
    return ispace;
  }
  LogicalRegion getLRegion() const {
    assert(lregionID != nullptr);
    return lregion;
  }
  LogicalPartition getLPart() const {
    assert(lpartID != nullptr);
    return lpart;
  }
  LogicalRegion getLRegion(Color color) const {
    assert(lpartID != nullptr);
    return runtime->get_logical_subregion_by_color(ctx, lpart, color);
  }
  void unMapPRegion();
  PhysicalRegion getPRegion(legion_privilege_mode_t priv = READ_WRITE);
  PhysicalRegion getRawPRegion() {
    return pregion;
  }
  void setPRegion(PhysicalRegion region) {
    pregion = region;
  }
  IndexSpace getSubspace(Color color);
  IndexIterator getSubspaceIterator(Color color) {
    getSubspace(color);
    return IndexIterator(runtime, ctx, subspace);
  }
protected:
  bool destroy_ispace;
  IndexSpaceID* ispaceID;
  IndexSpace ispace;
  IndexPartitionID* ipartID;
  IndexPartition ipart;
  IndexSpace subspace;
  std::vector<FieldID> fIDs;
  bool destroy_fspace;
  FieldSpaceID* fspaceID;
  FieldSpace fspace;
  RegionTreeID* lregionID;
  LogicalRegion lregion;
  RegionTreeID* lpartID;
  LogicalPartition lpart;
  PhysicalRegion pregion;
  Context ctx;
  Runtime* runtime;
  string name;
};
// class LogicalUnstructured

#endif /* SRC_LOGICALUNSTRUCTURED_HH_ */
