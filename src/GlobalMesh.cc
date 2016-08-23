/*
 * GlobalMesh.cc
 *
 *  Created on: Aug 8, 2016
 *      Author: jgraham
 */

#include "GlobalMesh.hh"

#include "GenerateGlobalMesh.hh"

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
	GenerateGlobalMesh gmesh(input_params_);

	// zones
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

	lregion_global_zones_ = runtime_->create_logical_region(ctx_, ispace_zones_, fspace_zones_);
	runtime_->attach_name(lregion_global_zones_, "GlobalMesh::lregion_global_zones_");

	// points
	num_pts_ = gmesh.numberOfPoints();

	ispace_pts_ = runtime_->create_index_space(ctx_, num_pts_);
	runtime_->attach_name(ispace_pts_, "GlobalMesh::ispace_pts_");
	{
	    IndexAllocator allocator = runtime_->create_index_allocator(ctx_, ispace_pts_);
	    ptr_t begin = allocator.alloc(num_pts_);
	    assert(!begin.is_null());
	}

	fspace_pts_ = runtime_->create_field_space(ctx_);
	runtime_->attach_name(fspace_pts_, "GlobalMesh::fspace_pts_");
	allocatePointFields();

	lregion_global_pts_ = runtime_->create_logical_region(ctx_, ispace_pts_, fspace_pts_);
	runtime_->attach_name(lregion_global_pts_, "GlobalMesh::lregion_global_pts_");

    RegionRequirement pt_req(lregion_global_pts_, WRITE_DISCARD, EXCLUSIVE, lregion_global_pts_);
	pt_req.add_field(FID_PX_INIT);
	InlineLauncher local_pt_launcher(pt_req);
	PhysicalRegion pt_region = runtime_->map_region(ctx_, local_pt_launcher);
	Double2Accessor pt_x = pt_region.get_field_accessor(FID_PX_INIT).typeify<double2>();

	// generate mesh
    std::vector<double2> nodepos;
    gmesh.generate(nodepos);

    // do a few initial calculations TODO JPG move back to local mesh
//    for (int pch = 0; pch < num_pt_chunks; ++pch) {
//        int pfirst = pt_chunks_first[pch];
 //       int plast = pt_chunks_last[pch];
        // copy nodepos into px, distributed across threads
        for (int p = 0; p < num_pts_; ++p) {
        		ptr_t pt_ptr(p);
            pt_x.write(pt_ptr, nodepos[p]);
        }
//    }

	// partitions
	Coloring local_zones_map;
	Coloring local_pts_map;
	gmesh.colorPartitions(&local_zones_map, &local_pts_map);

	IndexPartition zones_part = runtime_->create_index_partition(ctx_,
			ispace_zones_, local_zones_map, true/*disjoint*/);
	runtime_->attach_name(zones_part, "GlobalMesh::zones_part");
	lpart_zones_ = runtime_->get_logical_partition(ctx_, lregion_global_zones_, zones_part);

	IndexPartition pts_part = runtime_->create_index_partition(ctx_,
			ispace_pts_, local_pts_map, false/*disjoint*/);
	runtime_->attach_name(pts_part, "GlobalMesh::pts_part");
	lpart_pts_ = runtime_->get_logical_partition(ctx_, lregion_global_pts_, pts_part);
}

void GlobalMesh::clear() {
	runtime_->destroy_logical_region(ctx_, lregion_global_zones_);
	runtime_->destroy_field_space(ctx_, fspace_zones_);
	runtime_->destroy_index_space(ctx_, ispace_zones_);
	runtime_->destroy_logical_region(ctx_, lregion_global_pts_);
	runtime_->destroy_field_space(ctx_, fspace_pts_);
	runtime_->destroy_index_space(ctx_, ispace_pts_);
}

void GlobalMesh::allocateZoneFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_zones_);
	allocator.allocate_field(sizeof(double), FID_ZR);
	allocator.allocate_field(sizeof(double), FID_ZE);
	allocator.allocate_field(sizeof(double), FID_ZP);
}

void GlobalMesh::allocatePointFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_pts_);
	allocator.allocate_field(sizeof(double2), FID_PX_INIT);
}
