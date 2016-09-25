/*
 * LogicalElement.hh
 *
 *  Created on: Sep 8, 2016
 *      Author: jgraham
 */

#ifndef SRC_LOGICALUNSTRUCTURED_HH_
#define SRC_LOGICALUNSTRUCTURED_HH_

#include "Parallel.hh"

class LogicalUnstructured {
public:
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime);
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime, IndexSpace ispace);
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime, PhysicalRegion pregion);
    virtual ~LogicalUnstructured();
    template <typename TYPE>
      void addField(FieldID FID);
    void allocate(int nElements);
    void allocate();
    void partition(Coloring map, bool disjoint);
    template <typename TYPE>
      RegionAccessor<AccessorType::Generic, TYPE> getRegionAccessor(FieldID FID);
    IndexIterator getIterator() const { assert(ispaceID != NULL); return  IndexIterator(runtime,ctx, ispace);}
    IndexSpace getISpace() const { assert(ispaceID != NULL); return ispace;}
    LogicalRegion getLRegion() const {assert(lregionID != NULL); return lregion;}
    LogicalPartition getLPart() const {assert(lpartID != NULL); return lpart;}
    PhysicalRegion getPRegion();
    IndexSpace getSubspace(Color color);
private:
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
