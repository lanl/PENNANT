/*
 * WriteTask.cc
 *
 *  Created on: Aug 12, 2016
 *      Author: jgraham
 */

#include "WriteTask.hh"

#include <string>

#include "WriteXY.hh"
#include "ExportGold.hh"

using namespace std;

WriteTask::WriteTask(LogicalRegion lregion_global_zones,
		void *args, const size_t &size)
	 : TaskLauncher(WriteTask::TASK_ID, TaskArgument(args, size))
{
	add_region_requirement(RegionRequirement(lregion_global_zones, READ_ONLY, EXCLUSIVE, lregion_global_zones));
	add_field(0/*idx*/, FID_ZR);
	add_field(0/*idx*/, FID_ZE);
	add_field(0/*idx*/, FID_ZP);
}

/*static*/ const char * const WriteTask::TASK_NAME = "WriteTask";

/*static*/
void WriteTask::cpu_run(const Task *task,
		const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime* rt)
{
	assert(regions.size() == 1);
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 3);

	IndexSpace ispace_zones = task->regions[0].region.get_index_space();
	DoubleAccessor zone_energy_density = regions[0].get_field_accessor(FID_ZE).typeify<double>();
	DoubleAccessor zone_pressure = regions[0].get_field_accessor(FID_ZP).typeify<double>();
	DoubleAccessor zone_rho = regions[0].get_field_accessor(FID_ZR).typeify<double>();

	// Legion cannot handle data structures with indirections in them
    unsigned char *serialized_args = (unsigned char *) task->args;
    RunStat run_stat;
	size_t next_size = sizeof(RunStat);
    memcpy((void*)(&run_stat), (void*)serialized_args, next_size);
	serialized_args += next_size;

	next_size = sizeof(DirectInputParameters);
	DirectInputParameters input_params;
	memcpy((void*)(&input_params), (void*)serialized_args, next_size);
	serialized_args += next_size;

	next_size = sizeof(size_t);
	size_t n_probname;
	memcpy((void*)(&n_probname), (void*)serialized_args, next_size);
	serialized_args += next_size;

	string probname;
    {
	  next_size = n_probname * sizeof(char);
	  char *buffer = (char *)malloc(next_size+1);
	  memcpy((void *)buffer, (void *)serialized_args, next_size);
	  buffer[next_size] = '\0';
	  probname = std::string(buffer);
	  free(buffer);
	  serialized_args += next_size;
    }


	//ExportGold* egold_ = new ExportGold();
    WriteXY* wxy_ = new WriteXY();

    if (input_params.write_xy_file_) {
		IndexIterator zr_itr(rt, ctx, ispace_zones);
		IndexIterator ze_itr(rt, ctx, ispace_zones);
		IndexIterator zp_itr(rt, ctx, ispace_zones);
        cout << "Writing .xy file..." << endl;
        wxy_->write(probname, zone_rho, zone_energy_density, zone_pressure, zr_itr, ze_itr, zp_itr);
    }
    if (input_params.write_gold_file_) {
            cout << "Writing gold file..." << endl;
        //egold_->write(probname, cycle, time, zr, ze, zp, iterator);
    }


    delete wxy_;
    //delete egold_;
}
