/*
 * GlobalMesh.hh
 *
 *  Created on: Aug 8, 2016
 *      Author: jgraham
 */

#ifndef GLOBALMESH_HH_
#define GLOBALMESH_HH_

#include "InputParameters.hh"
#include "LogicalUnstructured.hh"
#include "Parallel.hh"

class GlobalMesh {
public:
	GlobalMesh(const InputParameters& input_params,
			Context ctx, HighLevelRuntime* runtime);
	virtual ~GlobalMesh();

	std::vector<PhaseBarrier> phase_barriers;
	std::vector<LogicalUnstructured> halos_points;
	std::vector<std::vector<int>> masters;

    LogicalUnstructured zones;
    LogicalUnstructured points;

private:
	void init();

	const InputParameters inputParams;
	Context ctx;
	HighLevelRuntime* runtime;
};

#endif /* GLOBALMESH_HH_ */
