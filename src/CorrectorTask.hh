/*
 * CorrectorTask.hh
 *
 *  Created on: Oct 3, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#ifndef SRC_CORRECTORTASK_HH_
#define SRC_CORRECTORTASK_HH_

#include "Parallel.hh"

class CorrectorTask : public TaskLauncher {
public:
	CorrectorTask(LogicalRegion mesh_zones,
            LogicalRegion mesh_sides,
            LogicalRegion mesh_zone_points,
            LogicalRegion mesh_points,
            LogicalRegion mesh_edges,
            LogicalRegion mesh_local_points,
            LogicalRegion point_chunks,
            LogicalRegion side_chunks,
            LogicalRegion zone_chunks,
            LogicalRegion hydro_zones,
            LogicalRegion hydro_sides_and_corners,
            LogicalRegion hydro_points,
            LogicalRegion bcx_chunks,
            LogicalRegion bcy_chunks,
			void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = CORRECTOR_TASK_ID;
	static const bool CPU_BASE_LEAF = true;

	static TimeStep cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_CORRECTORTASK_HH_ */

