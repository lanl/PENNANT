/*
 * GlobalMesh.cc
 *
 *  Created on: Aug 8, 2016
 *      Author: jgraham
 */

#include "GlobalMesh.hh"

#include "GenerateGlobalMesh.hh"


GlobalMesh::GlobalMesh(const InputParameters &input_params, Context ctx, HighLevelRuntime *runtime) :
    zones(ctx, runtime),
    points(ctx, runtime),
	inputParams(input_params),
	ctx(ctx),
	runtime(runtime)
{
	init();
}


GlobalMesh::~GlobalMesh()
{
    for (unsigned idx = 0; idx < phase_barriers.size(); idx++)
      runtime->destroy_phase_barrier(ctx, phase_barriers[idx]);
    phase_barriers.clear();
}


void GlobalMesh::init()
{
	GenerateGlobalMesh generate_mesh(inputParams);

	zones.addField<double>(FID_ZR);
	zones.addField<double>(FID_ZP);
	zones.addField<double>(FID_ZE);
	zones.allocate(generate_mesh.numberOfZones());

    points.allocate(generate_mesh.numberOfPoints());

    	Coloring zones_map;
    	Coloring local_pts_map;
	generate_mesh.colorPartitions(&zones_map, &local_pts_map);
	zones.partition(zones_map, true);
	points.partition(local_pts_map, false);

	Coloring ghost_pts_map;

	for (int color=0; color < inputParams.directs.ntasks; ++color) {
	    ghost_pts_map[color].points = std::set<ptr_t>(); // empty set
		std::vector<int> master_colors, slave_colors;
		generate_mesh.setupHaloCommunication(color, &master_colors, &slave_colors, &ghost_pts_map);
		masters.push_back(master_colors);
		phase_barriers.push_back(runtime->create_phase_barrier(ctx, 2 * (1 + slave_colors.size())));

		LogicalUnstructured subspace(ctx, runtime, points.getSubspace(color));
		subspace.partition(ghost_pts_map, true);
		halos_points.push_back(LogicalUnstructured(ctx, runtime, subspace.getSubspace(color)));
        halos_points[color].addField<double>(FID_GHOST_PMASWT);
        halos_points[color].addField<double2>(FID_GHOST_PF);
        halos_points[color].allocate();
	}
}
