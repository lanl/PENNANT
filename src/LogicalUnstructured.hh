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

#include "Parallel.hh"

class LogicalUnstructured {
public:
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime);
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime, IndexSpace ispace);
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime, PhysicalRegion pregion);
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime, LogicalRegion lregion);
    virtual ~LogicalUnstructured();
    template <typename TYPE>
      void addField(FieldID FID);
    virtual void allocate(int nElements);
    void allocate();
    void partition(Coloring map, bool disjoint);
    template <typename TYPE>
      RegionAccessor<AccessorType::SOA<sizeof(TYPE)>, TYPE> getRegionSOAAccessor(FieldID FID);
    IndexIterator getIterator() const { assert(ispaceID != nullptr); return  IndexIterator(runtime,ctx, ispace);}
    IndexSpace getISpace() const { assert(ispaceID != nullptr); return ispace;}
    LogicalRegion getLRegion() const {assert(lregionID != nullptr); return lregion;}
    LogicalPartition getLPart() const {assert(lpartID != nullptr); return lpart;}
    LogicalRegion getLRegion(Color color) const
    {
        assert(lpartID != nullptr);
        return runtime->get_logical_subregion_by_color(ctx, lpart, color);
    }
    void unMapPRegion();
    PhysicalRegion getPRegion();
    PhysicalRegion getRawPRegion() {return pregion;}
    void setPRegion(PhysicalRegion region) {pregion = region;}
    IndexSpace getSubspace(Color color);
    IndexIterator getSubspaceIterator(Color color) { getSubspace(color); return  IndexIterator(runtime,ctx, subspace);}
protected:
    void addField(unsigned int FID);
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
    bool destroy_lregion;
    RegionTreeID* lregionID;
    LogicalRegion lregion;
    RegionTreeID* lpartID;
    LogicalPartition lpart;
    PhysicalRegion pregion;
    Context ctx;
    HighLevelRuntime* runtime;
};  // class LogicalElement


#endif /* SRC_LOGICALUNSTRUCTURED_HH_ */
