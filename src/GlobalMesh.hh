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
	LogicalRegion lregion_global_pts_;
	LogicalPartition lpart_pts_;

private:
	void init();
	void clear();
	void allocateZoneFields();
	void allocatePointFields();

	IndexSpace ispace_zones_;
	FieldSpace fspace_zones_;
	IndexSpace ispace_pts_;
	FieldSpace fspace_pts_;

	const InputParameters input_params_;
	int num_zones_;
	int num_pts_;
	Context ctx_;
	HighLevelRuntime *runtime_;
};

#endif /* GLOBALMESH_HH_ */
