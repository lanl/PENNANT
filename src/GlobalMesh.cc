/*
 * GlobalMesh.cc
 *
 *  Created on: Aug 8, 2016
 *      Author: jgraham
 */

#include "GenerateMesh.hh"
#include "GlobalMesh.hh"

GlobalMesh::GlobalMesh(const InputParameters &input_params, Context ctx, HighLevelRuntime *runtime) :
	input_params_(input_params),
	ctx_(ctx),
	runtime_(runtime)
{
	init();
}

GlobalMesh::~GlobalMesh() {
	clear();
}


void GlobalMesh::init() {

	// generate mesh
	GenerateMesh gmesh(input_params_);
	num_zones_ = gmesh.numberOfZones();

	ispace_zones_ = runtime_->create_index_space(ctx_, num_zones_);
	runtime_->attach_name(ispace_zones_, "GlobalMesh::ispace_zones_");
	{
	    IndexAllocator allocator = runtime_->create_index_allocator(ctx_, ispace_zones_);
	    ptr_t begin = allocator.alloc(num_zones_);
	    assert(!begin.is_null());
	}

	fspace_zones_ = runtime_->create_field_space(ctx_);
	runtime_->attach_name(fspace_zones_, "GlobalMesh::fspace_zones_");
	allocateZoneFields();

	logical_region_global_zones_ = runtime_->create_logical_region(ctx_, ispace_zones_, fspace_zones_);
	runtime_->attach_name(logical_region_global_zones_, "GlobalMesh::logical_region_global_zones_");

	Coloring local_zones_map;
	gmesh.colorPartitions(&local_zones_map);
	//IndexPartition zones_part = runtime_->create_index_partition(ctx_,
	//		ispace_zones_, local_zones_map, true/*disjoint*/);
	//runtime_->attach_name(zones_part_, "GlobalMesh::zones_part_");

	//logical_part_zones_ = runtime_->get_logical_partition(ctx_, logical_region_global_zones_, zones_part);
}

void GlobalMesh::clear() {
	runtime_->destroy_logical_region(ctx_, logical_region_global_zones_);
	runtime_->destroy_field_space(ctx_, fspace_zones_);
	runtime_->destroy_index_space(ctx_, ispace_zones_);
}

void GlobalMesh::allocateZoneFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_zones_);
	allocator.allocate_field(sizeof(double), FID_ZR);
	allocator.allocate_field(sizeof(double), FID_ZE);
	allocator.allocate_field(sizeof(double), FID_ZP);
}
