/*
 * PredictorTask.hh
 *
 *  Created on: Oct 17, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#ifndef SRC_PREDICTORTASK_HH_
#define SRC_PREDICTORTASK_HH_

#include "Parallel.hh"

class PredictorTask : public TaskLauncher {
public:
	PredictorTask(LogicalRegion mesh_zones,
	        LogicalRegion mesh_sides,
	        LogicalRegion mesh_zone_pts,
            LogicalRegion mesh_points,
            LogicalRegion side_chunks,
            LogicalRegion mesh_edges,
	        LogicalRegion hydro_zones,
            LogicalRegion hydro_sides_and_corners,
            LogicalRegion hydro_points,
            void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = PREDICTOR_TASK_ID;
	static const bool CPU_BASE_LEAF = true;

	static void cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_PREDICTORTASK_HH_ */

