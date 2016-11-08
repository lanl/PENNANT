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


CalcDtTask::CalcDtTask(CalcDtTaskArgs *args)
	 : TaskLauncher(CalcDtTask::TASK_ID, TaskArgument(static_cast<void*>(args), sizeof(CalcDtTaskArgs)))
{
    add_future(args->dt_hydro);
}

/*static*/ const char * const CalcDtTask::TASK_NAME = "CalcDtTask";


/*static*/
TimeStep CalcDtTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* runtime)
{
    assert(task->arglen == sizeof(CalcDtTaskArgs));
    CalcDtTaskArgs args = *(const CalcDtTaskArgs*)task->args;

    args.dt_hydro = task->futures[0]; // Cannot pass future through task->args

    return Driver::calcGlobalDt(args);
}

