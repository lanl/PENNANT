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

	LogicalRegion logical_region_global_zones_;
	LogicalPartition logical_part_zones_;

private:
	void init();
	void clear();
	void allocateZoneFields();

	IndexSpace ispace_zones_;
	FieldSpace fspace_zones_;

	const InputParameters input_params_;
	int num_zones_;
	Context ctx_;
	HighLevelRuntime *runtime_;
};

#endif /* GLOBALMESH_HH_ */
