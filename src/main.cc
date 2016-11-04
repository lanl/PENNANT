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

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include "math.h"

#include "AddReductionOp.hh"
#include "Add2ReductionOp.hh"
#include "AddInt64ReductionOp.hh"
#include "CalcDtTask.hh"
#include "CorrectorTask.hh"
#include "Driver.hh"
#include "HaloTask.hh"
#include "InputFile.hh"
#include "InputParameters.hh"
#include "MinReductionOp.hh"
#include "Parallel.hh"
#include "PredictorPointTask.hh"
#include "PredictorTask.hh"
#include "WriteTask.hh"


using namespace std;

InputParameters parseInputFile(InputFile *inp);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
	const InputArgs &command_args = HighLevelRuntime::get_input_args();

	if (command_args.argc < 3) {
        cerr << "Usage: pennant <ntasks> <filename>" << endl;
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

    cout << "********************" << endl;
    cout << "Running PENNANT v0.9" << endl;
    cout << "********************" << endl;
    cout << endl;

    cout << "Running with " << ntasks << " Legion Tasks" << endl;

    InputParameters input_params = parseInputFile(&inp);
    input_params.directs.ntasks = ntasks;
    input_params.probname = probname;

	Parallel::run(input_params, ctx, runtime);
}


int main(int argc, char **argv)
{
	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
			Processor::LOC_PROC, true/*single*/, false/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(), "top_level_task");

    HighLevelRuntime::register_legion_task<TimeStep, CalcDtTask::cpu_run>(CALCDT_TASK_ID,
            Processor::LOC_PROC, true/*single*/, true/*index*/,
            AUTO_GENERATE_ID, TaskConfigOptions(CalcDtTask::CPU_BASE_LEAF), CalcDtTask::TASK_NAME);

    HighLevelRuntime::register_legion_task<TimeStep, CorrectorTask::cpu_run>(CORRECTOR_TASK_ID,
            Processor::LOC_PROC, true/*single*/, true/*index*/,
            AUTO_GENERATE_ID, TaskConfigOptions(CorrectorTask::CPU_BASE_LEAF), CorrectorTask::TASK_NAME);

    HighLevelRuntime::register_legion_task<RunStat, DriverTask::cpu_run>(DRIVER_TASK_ID,
            Processor::LOC_PROC, true/*single*/, true/*index*/,
            AUTO_GENERATE_ID, TaskConfigOptions(DriverTask::CPU_BASE_LEAF), DriverTask::TASK_NAME);

    TaskHelper::register_cpu_variants<HaloTask>();

    TaskHelper::register_cpu_variants<PredictorPointTask>();

    TaskHelper::register_cpu_variants<PredictorTask>();

    TaskHelper::register_cpu_variants<WriteTask>();

    HighLevelRuntime::register_legion_task<double, Parallel::globalSumTask>(GLOBAL_SUM_TASK_ID,
            Processor::LOC_PROC, true/*single*/, true/*index*/,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "globalSumTask");

    HighLevelRuntime::register_legion_task<int64_t, Parallel::globalSumInt64Task>(GLOBAL_SUM_INT64_TASK_ID,
            Processor::LOC_PROC, true/*single*/, true/*index*/,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "globalSumInt64Task");

    Runtime::register_reduction_op<AddReductionOp>(AddReductionOp::redop_id);
    Runtime::register_reduction_op<Add2ReductionOp>(Add2ReductionOp::redop_id);
    Runtime::register_reduction_op<AddInt64ReductionOp>(AddInt64ReductionOp::redop_id);

	Runtime::register_reduction_op<MinReductionOp>(MinReductionOp::redop_id);


	return HighLevelRuntime::start(argc, argv);
}


InputParameters parseInputFile(InputFile *inp) {
	InputParameters value;
	DirectInputParameters direct_values;

    direct_values.cstop = inp->getInt("cstop", 999999);
    direct_values.tstop = inp->getDouble("tstop", 1.e99);
    if (direct_values.cstop == 999999 && direct_values.tstop == 1.e99) {
    	cerr << "Must specify either cstop or tstop" << endl;
    	exit(1);
    }
    direct_values.dtmax = inp->getDouble("dtmax", 1.e99);
    direct_values.dtinit = inp->getDouble("dtinit", 1.e99);
    direct_values.dtfac = inp->getDouble("dtfac", 1.2);
    direct_values.dtreport = inp->getInt("dtreport", 10);
    direct_values.chunk_size = inp->getInt("chunksize", 0);
    if (direct_values.chunk_size < 0) {
        cerr << "Error: bad chunksize " << direct_values.chunk_size << endl;
        exit(1);
    }

    vector<double> subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    } else if (subregion.size() == 4) {
		direct_values.subregion_xmin = subregion[0];
		direct_values.subregion_xmax = subregion[1];
		direct_values.subregion_ymin = subregion[2];
		direct_values.subregion_ymax = subregion[3];
    } else if (subregion.size() == 0) {
		direct_values.subregion_xmin = std::numeric_limits<double>::max();
		direct_values.subregion_xmax = std::numeric_limits<double>::max();
		direct_values.subregion_ymin = std::numeric_limits<double>::max();
		direct_values.subregion_ymax = std::numeric_limits<double>::max();
    }

    direct_values.write_xy_file = inp->getInt("writexy", 0);
    direct_values.write_gold_file = inp->getInt("writegold", 0);

    value.meshtype = inp->getString("meshtype", "");
    if (value.meshtype.empty()) {
        cerr << "Error:  must specify meshtype" << endl;
        exit(1);
    }
    if (value.meshtype != "pie" &&
    		value.meshtype != "rect" &&
			value.meshtype != "hex") {
        cerr << "Error:  invalid meshtype " << value.meshtype << endl;
        exit(1);
    }
    vector<double> params =
            inp->getDoubleList("meshparams", vector<double>());
    if (params.empty()) {
        cerr << "Error:  must specify meshparams" << endl;
        exit(1);
    }
    if (params.size() > 4) {
        cerr << "Error:  meshparams must have <= 4 values" << endl;
        exit(1);
    }

    direct_values.nzones_x = params[0];
    direct_values.nzones_y = (params.size() >= 2 ? params[1] : direct_values.nzones_x);
    if (value.meshtype != "pie")
    	direct_values.len_x = (params.size() >= 3 ? params[2] : 1.0);
    else
        // convention:  x = theta, y = r
    	direct_values.len_x = (params.size() >= 3 ? params[2] : 90.0)
                * M_PI / 180.0;
    direct_values.len_y = (params.size() >= 4 ? params[3] : 1.0);

    if (direct_values.nzones_x <= 0 || direct_values.nzones_y <= 0 || direct_values.len_x <= 0. || direct_values.len_y <= 0. ) {
        cerr << "Error:  meshparams values must be positive" << endl;
        exit(1);
    }
    if (value.meshtype == "pie" && direct_values.len_x >= 2. * M_PI) {
        cerr << "Error:  meshparams theta must be < 360" << endl;
        exit(1);
    }

    direct_values.cfl = inp->getDouble("cfl", 0.6);
    direct_values.cflv = inp->getDouble("cflv", 0.1);
    direct_values.rho_init = inp->getDouble("rinit", 1.);
    direct_values.energy_init = inp->getDouble("einit", 0.);
    direct_values.rho_init_sub = inp->getDouble("rinitsub", 1.);
    direct_values.energy_init_sub = inp->getDouble("einitsub", 0.);
    direct_values.vel_init_radial = inp->getDouble("uinitradial", 0.);
    value.bcx = inp->getDoubleList("bcx", vector<double>());
    value.bcy = inp->getDoubleList("bcy", vector<double>());
    direct_values.gamma = inp->getDouble("gamma", 5. / 3.);
    direct_values.ssmin = inp->getDouble("ssmin", 0.);
    direct_values.alpha = inp->getDouble("alfa", 0.5);
    direct_values.qgamma = inp->getDouble("qgamma", 5. / 3.);
    direct_values.q1 = inp->getDouble("q1", 0.);
    direct_values.q2 = inp->getDouble("q2", 2.);

    value.directs = direct_values;
	return value;
}

