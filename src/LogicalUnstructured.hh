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
    LogicalUnstructured(Context ctx, HighLevelRuntime *runtime, LogicalRegion lregion);
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
    LogicalRegion getLRegion(Color color) const
    {
        assert(lpartID != NULL);
        return runtime->get_logical_subregion_by_color(ctx, lpart, color);
    }
    PhysicalRegion getPRegion();
    IndexSpace getSubspace(Color color);
    IndexIterator getSubspaceIterator(Color color) { getSubspace(color); return  IndexIterator(runtime,ctx, subspace);}
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
