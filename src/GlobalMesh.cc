/*
 * GlobalMesh.cc
 *
 *  Created on: Aug 8, 2016
 *      Author: jgraham
 */

#include "GlobalMesh.hh"

#include "GenerateGlobalMesh.hh"


GlobalMesh::GlobalMesh(const InputParameters &input_params, Context ctx, HighLevelRuntime *runtime) :
	inputParams(input_params),
    zones(ctx, runtime),
    sides(ctx, runtime),
    points(ctx, runtime),
    zonePointsCRS(ctx, runtime),
	ctx(ctx),
	runtime(runtime)
{
	init();
}


GlobalMesh::~GlobalMesh()
{
    // TODO destroy phase barriers
}


void GlobalMesh::init()
{
	// generate mesh
	GenerateGlobalMesh generate_mesh(inputParams);

	// zones
	numZones = generate_mesh.numberOfZones();
	zones.addField<double>(FID_ZR);
	zones.addField<double>(FID_ZP);
	zones.addField<double>(FID_ZE);
	zones.allocate(numZones);

	// sides (and corners)
	numSides = generate_mesh.numberOfSides();
	sides.addField<int>(FID_ZONE_PTS);
	sides.allocate(numSides);

	// points
	numPoints = generate_mesh.numberOfPoints();
    points.addField<double2>(FID_PX_INIT);
    points.allocate(numPoints);

	// zone points compressed row storage
	numZonePointsCRS = generate_mesh.numberOfZones() + 1;
	zonePointsCRS.addField<int>(FID_ZONE_PTS_PTR);
	zonePointsCRS.allocate(numZonePointsCRS);

	// TODO move back to local SPMD mesh as much as we can
    std::vector<double2> nodepos;
    std::vector<int> cellstart, cellnodes;
    generate_mesh.generate(nodepos, cellstart, cellnodes);

    Double2Accessor pt_x = points.getRegionAccessor<double2>(FID_PX_INIT);
    for (int p = 0; p < numPoints; ++p) {
        ptr_t pt_ptr(p);
        pt_x.write(pt_ptr, nodepos[p]);
   }

   IntAccessor zone_pts = sides.getRegionAccessor<int>(FID_ZONE_PTS);
   for (int s = 0; s < numSides; ++s) {
        	ptr_t side_ptr(s);
        zone_pts.write(side_ptr, cellnodes[s]);
    }

    	IntAccessor zone_pts_ptr = zonePointsCRS.getRegionAccessor<int>(FID_ZONE_PTS_PTR);
    	for (int z = 0; z < numZonePointsCRS; ++z) {
        	ptr_t zone_ptr(z);
        zone_pts_ptr.write(zone_ptr, cellstart[z]);
    }

	// partitions
    	Coloring zones_map;
    	Coloring sides_map;
    	Coloring local_pts_map;
    	Coloring CRS_map;
	generate_mesh.colorPartitions(cellstart, &zones_map, &sides_map, &local_pts_map, &CRS_map);
	zones.partition(zones_map, true);
	sides.partition(sides_map, true);
	points.partition(local_pts_map, false);
	zonePointsCRS.partition(CRS_map, false);

	// ghost communication
	Coloring ghost_pts_map;

	for (int color=0; color < inputParams.directs_.ntasks_; ++color) {
	    ghost_pts_map[color].points = std::set<ptr_t>(); // empty set
		std::vector<int> partners;
		generate_mesh.sharePoints(color, &partners, &ghost_pts_map);
		neighbors.push_back(partners);
		readyBarriers.push_back(runtime->create_phase_barrier(ctx, 1));
		emptyBarriers.push_back(runtime->create_phase_barrier(ctx, partners.size() - 1));
	}


	fSpaceGhostPoints = runtime->create_field_space(ctx);
	runtime->attach_name(fSpaceGhostPoints, "GlobalMesh::fspace_ghost_pts");
	allocateGhostPointFields();

	IndexPartition ghost_pts_part = runtime->create_index_partition(ctx,
			points.getISpace(), ghost_pts_map, false/*disjoint*/);
	runtime->attach_name(ghost_pts_part, "GlobalMesh::ghost_pts_part");
	for (int color=0; color < inputParams.directs_.ntasks_; ++color) {
		IndexSpace ispace_ghost_pts = runtime->get_index_subspace(ctx, ghost_pts_part, color);
		char buf[32];
		sprintf(buf, "ispace_ghost_pts %d", color);
		runtime->attach_name(ispace_ghost_pts, buf);
	    LogicalRegion lregion_ghost_pts = runtime->create_logical_region(ctx, ispace_ghost_pts,
	    		fSpaceGhostPoints);
		sprintf(buf, "lregion_ghost_pts %d", color);
		runtime->attach_name(lregion_ghost_pts, buf);
		lRegionsGhost.push_back(lregion_ghost_pts);
	}
}


void GlobalMesh::allocateGhostPointFields() {
	FieldAllocator allocator = runtime->create_field_allocator(ctx, fSpaceGhostPoints);
	allocator.allocate_field(sizeof(double), FID_GHOST_PMASWT);
	allocator.allocate_field(sizeof(double2), FID_GHOST_PF);
}
