/*
 * main.hh
 *
 *  Created on: Jul 19, 2016
 *      Author: jgraham
 */

#ifndef SRC_MAIN_HH_
#define SRC_MAIN_HH_

#include "InputFile.hh"

#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

#include<string>

enum TaskIDs {
	TOP_LEVEL_TASK_ID,
	DRIVER_TASK_ID,
};

enum Variants {
  CPU_VARIANT,
  GPU_VARIANT,
};

struct SPMDArgs {
    InputFile inp_;
    std::string probname_;
};

namespace TaskHelper {
  template<typename T>
  void base_cpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
    T::cpu_run(task, regions, ctx, runtime);
  }

#ifdef USE_CUDA
  template<typename T>
  void base_gpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
	const int *p = (int*)task->local_args;
    T::gpu_run(*p, regions);
  }
#endif

  template<typename T>
  void register_cpu_variants(void)
  {
    HighLevelRuntime::register_legion_task<base_cpu_wrapper<T> >(T::TASK_ID, Processor::LOC_PROC,
                                                                 false/*single*/, true/*index*/,
                                                                 CPU_VARIANT,
                                                                 TaskConfigOptions(T::CPU_BASE_LEAF),
                                                                 T::TASK_NAME);
  }

};

#endif /* SRC_MAIN_HH_ */
