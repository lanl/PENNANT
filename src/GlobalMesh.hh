/*
 * GlobalMesh.hh
 *
 *  Created on: Aug 8, 2016
 *      Author: jgraham
 */

#ifndef GLOBALMESH_HH_
#define GLOBALMESH_HH_

#include "InputParameters.hh"

#include "legion.h"
using namespace LegionRuntime::HighLevel;

class GlobalMesh {
public:
	GlobalMesh(const InputParameters &input_params,
			Context ctx, HighLevelRuntime *runtime);
	virtual ~GlobalMesh();

	LogicalRegion lregion_global_zones_;
	LogicalPartition lpart_zones_;

	LogicalRegion lregion_global_sides_;
	LogicalPartition lpart_sides_;

	LogicalRegion lregion_global_pts_;
	LogicalPartition lpart_pts_;

	LogicalRegion lregion_zone_pts_crs_;
	LogicalPartition lpart_zone_pts_crs_;

private:
	void init();
	void clear();
	void allocateZoneFields();
	void allocateSideFields();
	void allocatePointFields();
	void allocateZonePtsCRSFields();

	IndexSpace ispace_zones_;
	FieldSpace fspace_zones_;

	IndexSpace ispace_sides_;
	FieldSpace fspace_sides_;

	IndexSpace ispace_pts_;
	FieldSpace fspace_pts_;

	IndexSpace ispace_zone_pts_crs_;
	FieldSpace fspace_zone_pts_crs_;

	const InputParameters input_params_;
	int num_zones_;
	int num_sides_;
	int num_pts_;
	int num_zone_pts_crs_;
	Context ctx_;
	HighLevelRuntime *runtime_;
};

#endif /* GLOBALMESH_HH_ */
