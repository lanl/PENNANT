/*
 * HaloTask.hh
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

#ifndef SRC_HALOTASK_HH_
#define SRC_HALOTASK_HH_

#include "Parallel.hh"

class HaloTask : public TaskLauncher {
public:
	HaloTask(LogicalRegion mesh_sides,
            LogicalRegion mesh_points,
            LogicalRegion mesh_local_points,
            LogicalRegion point_chunks,
            LogicalRegion hydro_sides_and_corners,
            void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = HALO_TASK_ID;
	static const bool CPU_BASE_LEAF = true;

	static void cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_HALOTASK_HH_ */

