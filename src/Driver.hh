/*
 * Driver.hh
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef DRIVER_HH_
#define DRIVER_HH_

#include "LogicalUnstructured.hh"
#include "Parallel.hh"

#include <string>


// forward declarations
class LocalMesh;
class Hydro;

class DriverTask : public TaskLauncher {
public:
	DriverTask(int my_color,
			LogicalRegion my_zones,
			LogicalRegion all_zones,
			std::vector<LogicalRegion> halo_pts,
			void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = DRIVER_TASK_ID;
	static const bool CPU_BASE_LEAF = false;

	static RunStat cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

class Driver {
public:

    Driver(const InputParameters &params,
            DynamicCollective add_reduction,
            DynamicCollective add_int64_reduction,
            DynamicCollective min_reduction,
            PhaseBarrier pbarrier_as_master,
            std::vector<PhaseBarrier> masters_pbarriers,
            const PhysicalRegion& zones,
            IndexSpace pts,
            std::vector<LogicalUnstructured>& halos_points,
            std::vector<PhysicalRegion>& pregions_halos,
            Context ctx, HighLevelRuntime* rt);

    RunStat run();
    static TimeStep calcGlobalDt(CalcDtTaskArgs args);

private:

    std::string probname;          // problem name
    double tstop;                  // simulation stop time
    int cstop;                     // simulation stop cycle
    double dtmax;                  // maximum timestep size
    double dtinit;                 // initial timestep size
    double dtfac;                  // factor limiting timestep growth
    int dtreport;                  // frequency for timestep reports

    LocalMesh *mesh;
    Hydro *hydro;
    Future dt_hydro;

    DynamicCollective add_reduction;
    DynamicCollective add_int64_reduction;
    DynamicCollective min_reduction;
    Context ctx;
    HighLevelRuntime* runtime;
    const int my_color;
    LogicalUnstructured global_zones;
};  // class Driver


#endif /* DRIVER_HH_ */
