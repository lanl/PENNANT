/*
 * WriteTask.hh
 *
 *  Created on: Aug 12, 2016
 *      Author: jgraham
 */

#ifndef SRC_WRITETASK_HH_
#define SRC_WRITETASK_HH_

#include "Parallel.hh"

class WriteXY;
class ExportGold;


class WriteTask : public TaskLauncher {
public:
	WriteTask(LogicalRegion lregion_zone,
			void *args, const size_t &size);
	static const char * const TASK_NAME;
	static const int TASK_ID = WRITE_TASK_ID;
	static const bool CPU_BASE_LEAF = false;

	static void cpu_run(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime* rt);
};

#endif /* SRC_WRITETASK_HH_ */
