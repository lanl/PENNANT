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
#include "Driver.hh"
#include "InputFile.hh"
#include "InputParameters.hh"
#include "MinReductionOp.hh"
#include "Parallel.hh"


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
    input_params.directs_.ntasks_ = ntasks;
    input_params.probname_ = probname;

	Parallel parallel;
	parallel.init(input_params, ctx, runtime);
	parallel.run();
}


int main(int argc, char **argv)
{
	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
			Processor::LOC_PROC, true/*single*/, false/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(), "top_level_task");

	TaskHelper::register_cpu_variants<DriverTask>();

	Runtime::register_reduction_op<AddReductionOp>(AddReductionOp::redop_id);

	Runtime::register_reduction_op<MinReductionOp>(MinReductionOp::redop_id);


	return HighLevelRuntime::start(argc, argv);
}



InputParameters parseInputFile(InputFile *inp) {
	InputParameters value;
	DirectInputParameters direct_values;

    direct_values.cstop_ = inp->getInt("cstop", 999999);
    direct_values.tstop_ = inp->getDouble("tstop", 1.e99);
    if (direct_values.cstop_ == 999999 && direct_values.tstop_ == 1.e99) {
    	cerr << "Must specify either cstop or tstop" << endl;
    	exit(1);
    }
    direct_values.dtmax_ = inp->getDouble("dtmax", 1.e99);
    direct_values.dtinit_ = inp->getDouble("dtinit", 1.e99);
    direct_values.dtfac_ = inp->getDouble("dtfac", 1.2);
    direct_values.dtreport_ = inp->getInt("dtreport", 10);
    direct_values.chunk_size_ = inp->getInt("chunksize", 0);
    if (direct_values.chunk_size_ < 0) {
        cerr << "Error: bad chunksize " << direct_values.chunk_size_ << endl;
        exit(1);
    }

    vector<double> subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    } else if (subregion.size() == 4) {
		direct_values.subregion_xmin_ = subregion[0];
		direct_values.subregion_xmax_ = subregion[1];
		direct_values.subregion_ymin_ = subregion[2];
		direct_values.subregion_ymax_ = subregion[3];
    } else if (subregion.size() == 0) {
		direct_values.subregion_xmin_ = std::numeric_limits<double>::max();
		direct_values.subregion_xmax_ = std::numeric_limits<double>::max();
		direct_values.subregion_ymin_ = std::numeric_limits<double>::max();
		direct_values.subregion_ymax_ = std::numeric_limits<double>::max();
    }

    direct_values.write_xy_file_ = inp->getInt("writexy", 0);
    direct_values.write_gold_file_ = inp->getInt("writegold", 0);

    value.meshtype_ = inp->getString("meshtype", "");
    if (value.meshtype_.empty()) {
        cerr << "Error:  must specify meshtype" << endl;
        exit(1);
    }
    if (value.meshtype_ != "pie" &&
    		value.meshtype_ != "rect" &&
			value.meshtype_ != "hex") {
        cerr << "Error:  invalid meshtype " << value.meshtype_ << endl;
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

    direct_values.nzones_x_ = params[0];
    direct_values.nzones_y_ = (params.size() >= 2 ? params[1] : direct_values.nzones_x_);
    if (value.meshtype_ != "pie")
    	direct_values.len_x_ = (params.size() >= 3 ? params[2] : 1.0);
    else
        // convention:  x = theta, y = r
    	direct_values.len_x_ = (params.size() >= 3 ? params[2] : 90.0)
                * M_PI / 180.0;
    direct_values.len_y_ = (params.size() >= 4 ? params[3] : 1.0);

    if (direct_values.nzones_x_ <= 0 || direct_values.nzones_y_ <= 0 || direct_values.len_x_ <= 0. || direct_values.len_y_ <= 0. ) {
        cerr << "Error:  meshparams values must be positive" << endl;
        exit(1);
    }
    if (value.meshtype_ == "pie" && direct_values.len_x_ >= 2. * M_PI) {
        cerr << "Error:  meshparams theta must be < 360" << endl;
        exit(1);
    }

    direct_values.cfl_ = inp->getDouble("cfl", 0.6);
    direct_values.cflv_ = inp->getDouble("cflv", 0.1);
    direct_values.rho_init_ = inp->getDouble("rinit", 1.);
    direct_values.energy_init_ = inp->getDouble("einit", 0.);
    direct_values.rho_init_sub_ = inp->getDouble("rinitsub", 1.);
    direct_values.energy_init_sub_ = inp->getDouble("einitsub", 0.);
    direct_values.vel_init_radial_ = inp->getDouble("uinitradial", 0.);
    value.bcx_ = inp->getDoubleList("bcx", vector<double>());
    value.bcy_ = inp->getDoubleList("bcy", vector<double>());
    direct_values.gamma_ = inp->getDouble("gamma", 5. / 3.);
    direct_values.ssmin_ = inp->getDouble("ssmin", 0.);
    direct_values.alfa_ = inp->getDouble("alfa", 0.5);
    direct_values.qgamma_ = inp->getDouble("qgamma", 5. / 3.);
    direct_values.q1_ = inp->getDouble("q1", 0.);
    direct_values.q2_ = inp->getDouble("q2", 2.);

    value.directs_ = direct_values;
	return value;
}
