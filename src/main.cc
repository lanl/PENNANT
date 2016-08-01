/*
 * main.cc
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "main.hh"

#include <cstdlib>
#include <string>
#include <iostream>

#include "Parallel.hh"
#include "Driver.hh"


using namespace std;

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
	const InputArgs &command_args = HighLevelRuntime::get_input_args();

	if (command_args.argc < 3) {
        cerr << "Usage: pennant-circuit <ntasks> <filename>" << endl;
        exit(1);
    }
	int ntasks = atoi(command_args.argv[1]);
    assert(ntasks > 0);

    const char* filename = command_args.argv[2];
    InputFile inp(filename);

    string probname(filename);
    // strip .pnt suffix from filename
    int len = probname.length();
    if (probname.substr(len - 4, 4) == ".pnt")
        probname = probname.substr(0, len - 4);

	Parallel::init();

    Driver drv(&inp, probname);

    drv.run();

    Parallel::final();

}


int main(int argc, char **argv)
{
	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
			Processor::LOC_PROC, true/*single*/, false/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(), "top_level_task");

	TaskHelper::register_cpu_variants<DriverTask>();

	return HighLevelRuntime::start(argc, argv);
}


