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
	GenerateGlobalMesh gen_mesh(input_params_);

	// zones
	num_zones_ = gen_mesh.numberOfZones();

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
	// sides (and corners)
	num_sides_ = gen_mesh.numberOfSides();

	ispace_sides_ = runtime_->create_index_space(ctx_, num_sides_);
	runtime_->attach_name(ispace_sides_, "GlobalMesh::ispace_sides_");
	{
	    IndexAllocator allocator = runtime_->create_index_allocator(ctx_, ispace_sides_);
	    ptr_t begin = allocator.alloc(num_sides_);
 	    assert(!begin.is_null());
	}

	fspace_sides_ = runtime_->create_field_space(ctx_);
	runtime_->attach_name(fspace_sides_, "GlobalMesh::fspace_sides_");
	allocateSideFields();

	lregion_global_sides_ = runtime_->create_logical_region(ctx_, ispace_sides_, fspace_sides_);
	runtime_->attach_name(lregion_global_sides_, "GlobalMesh::lregion_global_sides_");

	// points
	num_pts_ = gen_mesh.numberOfPoints();

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


	// zone points compressed row storage
	num_zone_pts_crs_ = gen_mesh.numberOfZones() + 1;

	ispace_zone_pts_crs_ = runtime_->create_index_space(ctx_, num_zone_pts_crs_);
	runtime_->attach_name(ispace_zone_pts_crs_, "GlobalMesh::ispace_zone_pts_crs_");
	{
	    IndexAllocator allocator = runtime_->create_index_allocator(ctx_, ispace_zone_pts_crs_);
	    ptr_t begin = allocator.alloc(num_zone_pts_crs_);
	    assert(!begin.is_null());
	}

	fspace_zone_pts_crs_ = runtime_->create_field_space(ctx_);
	runtime_->attach_name(fspace_zone_pts_crs_, "GlobalMesh::fspace_zone_pts_crs_");
	allocateZonePtsCRSFields();

	lregion_zone_pts_crs_ = runtime_->create_logical_region(ctx_, ispace_zone_pts_crs_, fspace_zone_pts_crs_);
	runtime_->attach_name(lregion_zone_pts_crs_, "GlobalMesh::lregion_zone_pts_crs_");

	// generate mesh TODO JPG move back to local SPMD mesh as much as we can
    std::vector<double2> nodepos;
    std::vector<int> cellstart, cellnodes;
    gen_mesh.generate(nodepos, cellstart, cellnodes);

    RegionRequirement pt_req(lregion_global_pts_, WRITE_DISCARD, EXCLUSIVE, lregion_global_pts_);
	pt_req.add_field(FID_PX_INIT);
	InlineLauncher local_pt_launcher(pt_req);
	PhysicalRegion pt_region = runtime_->map_region(ctx_, local_pt_launcher);
	Double2Accessor pt_x = pt_region.get_field_accessor(FID_PX_INIT).typeify<double2>();

	// do a few initial calculations
        // copy nodepos into px, distributed across threads
        for (int p = 0; p < num_pts_; ++p) {
        		ptr_t pt_ptr(p);
            pt_x.write(pt_ptr, nodepos[p]);
        }

    RegionRequirement side_req(lregion_global_sides_, WRITE_DISCARD, EXCLUSIVE, lregion_global_sides_);
    	side_req.add_field(FID_ZONE_PTS);
    	InlineLauncher side_launcher(side_req);
    	PhysicalRegion side_region = runtime_->map_region(ctx_, side_launcher);
    	IntAccessor zone_pts = side_region.get_field_accessor(FID_ZONE_PTS).typeify<int>();

    	for (int s = 0; s < num_sides_; ++s) {
        	ptr_t side_ptr(s);
        zone_pts.write(side_ptr, cellnodes[s]);
    }

    	RegionRequirement crs_req(lregion_zone_pts_crs_, WRITE_DISCARD, EXCLUSIVE, lregion_zone_pts_crs_);
    	crs_req.add_field(FID_ZONE_PTS_PTR);
    	InlineLauncher crs_launcher(crs_req);
    	PhysicalRegion crs_region = runtime_->map_region(ctx_, crs_launcher);
    	IntAccessor zone_pts_ptr = crs_region.get_field_accessor(FID_ZONE_PTS_PTR).typeify<int>();

    	for (int z = 0; z < num_zone_pts_crs_; ++z) {
        	ptr_t zone_ptr(z);
        zone_pts_ptr.write(zone_ptr, cellstart[z]);
    }

	// partitions
    	Coloring zones_map;
    	Coloring sides_map;
    	Coloring local_pts_map;
    	Coloring crs_map;
	gen_mesh.colorPartitions(cellstart, &zones_map, &sides_map, &local_pts_map, &crs_map);

	IndexPartition zones_part = runtime_->create_index_partition(ctx_,
			ispace_zones_, zones_map, true/*disjoint*/);
	runtime_->attach_name(zones_part, "GlobalMesh::zones_part");
	lpart_zones_ = runtime_->get_logical_partition(ctx_, lregion_global_zones_, zones_part);

	IndexPartition sides_part = runtime_->create_index_partition(ctx_,
			ispace_sides_, sides_map, true/*disjoint*/);
	runtime_->attach_name(sides_part, "GlobalMesh::sides_part");
	lpart_sides_ = runtime_->get_logical_partition(ctx_, lregion_global_sides_, sides_part);

	IndexPartition pts_part = runtime_->create_index_partition(ctx_,
			ispace_pts_, local_pts_map, false/*disjoint*/);
	runtime_->attach_name(pts_part, "GlobalMesh::pts_part");
	lpart_pts_ = runtime_->get_logical_partition(ctx_, lregion_global_pts_, pts_part);

	IndexPartition zone_pts_crs_part = runtime_->create_index_partition(ctx_,
			ispace_zone_pts_crs_, crs_map, false/*disjoint*/);
	runtime_->attach_name(zone_pts_crs_part, "GlobalMesh::zone_pts_crs_part");
	lpart_zone_pts_crs_ = runtime_->get_logical_partition(ctx_, lregion_zone_pts_crs_, zone_pts_crs_part);

	// ghost communication
	Coloring ghost_pts_map;
	ghost_pts_map[0].points = std::set<ptr_t>(); // empty set

	for (int color=0; color < input_params_.directs_.ntasks_; ++color) {
		std::vector<int> partners;
		gen_mesh.sharePoints(color, &partners, &ghost_pts_map);
		neighbors.push_back(partners);
		ready_barriers.push_back(runtime_->create_phase_barrier(ctx_, 1));
		empty_barriers.push_back(runtime_->create_phase_barrier(ctx_, partners.size() - 1));
	}


	fspace_ghost_pts = runtime_->create_field_space(ctx_);
	runtime_->attach_name(fspace_ghost_pts, "GlobalMesh::fspace_ghost_pts");
	allocateGhostPointFields();

	IndexPartition ghost_pts_part = runtime_->create_index_partition(ctx_,
			ispace_pts_, ghost_pts_map, false/*disjoint*/);
	runtime_->attach_name(ghost_pts_part, "GlobalMesh::ghost_pts_part");
	for (int color=0; color < input_params_.directs_.ntasks_; ++color) {
		IndexSpace ispace_ghost_pts = runtime_->get_index_subspace(ctx_, ghost_pts_part, color);
		char buf[32];
		sprintf(buf, "ispace_ghost_pts %d", color);
		runtime_->attach_name(ispace_ghost_pts, buf);
	    LogicalRegion lregion_ghost_pts = runtime_->create_logical_region(ctx_, ispace_ghost_pts,
	    		fspace_ghost_pts);
		sprintf(buf, "lregion_ghost_pts %d", color);
		runtime_->attach_name(lregion_ghost_pts, buf);
		lregions_ghost.push_back(lregion_ghost_pts);
	}
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

void GlobalMesh::allocateSideFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_sides_);
	allocator.allocate_field(sizeof(int), FID_ZONE_PTS);
}

void GlobalMesh::allocatePointFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_pts_);
	allocator.allocate_field(sizeof(double2), FID_PX_INIT);
}

void GlobalMesh::allocateGhostPointFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_ghost_pts);
	allocator.allocate_field(sizeof(double), FID_GHOST_PMASWT);
	allocator.allocate_field(sizeof(double2), FID_GHOST_PF);
}

void GlobalMesh::allocateZonePtsCRSFields() {
	FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fspace_zone_pts_crs_);
	allocator.allocate_field(sizeof(int), FID_ZONE_PTS_PTR);
}
