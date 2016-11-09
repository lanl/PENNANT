/*
 * PredictorPointTask.hh
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

#ifndef SRC_PREDICTORPOINTTASK_HH_
#define SRC_PREDICTORPOINTTASK_HH_

#include "Parallel.hh"

class PredictorPointTask : public TaskLauncher {
public:
	PredictorPointTask(
            LogicalRegion mesh_points,
            LogicalRegion point_chunks,
            LogicalRegion hydro_points,
            void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = PREDICTOR_POINT_TASK_ID;
	static const bool CPU_BASE_LEAF = true;

	static void cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_PREDICTORPOINTTASK_HH_ */

