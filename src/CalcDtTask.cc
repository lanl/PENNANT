/*
 * CalcDtTask.cc
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

#include "CalcDtTask.hh"

#include "Driver.hh"


CalcDtTask::CalcDtTask(void *args, const size_t &size)
	 : TaskLauncher(CalcDtTask::TASK_ID, TaskArgument(args, size))
{
}

/*static*/ const char * const CalcDtTask::TASK_NAME = "CalcDtTask";


/*static*/
TimeStep CalcDtTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
    assert(task->arglen == sizeof(CalcDtTaskArgs));
    CalcDtTaskArgs args = *(const CalcDtTaskArgs*)task->args;

    return Driver::calcGlobalDt(args);
}

