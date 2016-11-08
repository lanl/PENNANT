/*
 * CalcDtTask.hh
 *
 *  Created on: Nov 8, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#ifndef SRC_CALCDTTASK_HH_
#define SRC_CALCDTTASK_HH_

#include "Parallel.hh"

class CalcDtTask : public TaskLauncher {
public:
	CalcDtTask(CalcDtTaskArgs *args);
	static const char * const TASK_NAME;
	static const int TASK_ID = CALCDT_TASK_ID;
	static const bool CPU_BASE_LEAF = true;

	static TimeStep cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_CALCDTTASK_HH_ */

