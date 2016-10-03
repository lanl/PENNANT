/*
 * HydroTask2.hh
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

#ifndef SRC_HYDROTASK2_HH_
#define SRC_HYDROTASK2_HH_

#include "Parallel.hh"

class HydroTask2 : public TaskLauncher {
public:
	HydroTask2(LogicalRegion mesh_zones,
	        LogicalRegion hydro_zones,
			void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = HYDRO_TASK2_ID;
	static const bool CPU_BASE_LEAF = true;

	static TimeStep cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_HYDROTASK2_HH_ */

