

PENNANT Mini-App
Instructions and Run Rules for APEX Benchmarking

A.) Description

PENNANT is a mini-app intended for use in advanced architecture
research.  It has data structures for manipulating 2-D unstructured
finite element meshes containing arbitrary polygons.  It implements
staggered-mesh Lagrangian hydrodynamics, subzonal pressure treatment,
and artificial viscosity.

PENNANT implements a Lagrangian staggered-grid hydrodynamics
algorithm on a 2-D unstructured finite-volume mesh composed
of arbitrary polygons with arbitrary connectivity.
"Staggered-grid" means that some state variables are
defined on mesh points, while others are defined on zones.
Because of the unstructured and staggered-grid nature of the
algorithm, irregular memory access patterns will be used
frequently.

PENNANT implements parallelism at two levels.  MPI parallelism
is supported using geometric domain decomposition, with
a gather-sum-scatter operation on point data implemented
across processors using MPI calls.  PENNANT also supports
threaded parallelism using OpenMP or CUDA by dividing the mesh
into processing chunks which are nearly independent and can be
processed in parallel, using an OpenMP parallel for or a CUDA
kernel launch with one grid block per chunk.

More detailed documentation can be found in the doc directory of
the PENNANT distribution.

B.) How to Build PENNANT

A simple Makefile is provided in the top-level directory for building
the code.  Before using it, you may wish to edit the definitions of CXX
and CXXFLAGS to specify your desired C++ compiler and flags, and to
choose between optimized/debug and serial/OpenMP/MPI builds.  Then
a simple "make" command will create a build subdirectory and
build the pennant binary in that directory.

PENNANT has been tested under GCC 5.1.0, PGI 15.3, and Intel 15.0.3.
Building under other compilers should require only minor changes.

C.) How to Run PENNANT

Several test problems are provided in subdirectories under the top-level
test directory.  Each subdirectory contains a small text file
<testname>.pnt containing run parameters for that test case.
The command line

pennant <testname>.pnt

is used to run a test in serial mode.  If running under MPI, this should
be preceded by mpirun or similar command as appropriate on your system.
Use the -np argument to change the number of MPI ranks, and
the code will determine a suitable decomposition.  Similarly
for OpenMP, specify the OMP_NUM_THREADS environment parameter.

D.) Required Problems

The APEX RFP defines problem sizes to be used in benchmarking.
The PENNANT test problems corresponding to this definition are:

"Small" problem:   leblancbig
"Medium" problem:  leblancx4
"Large" problem:   leblancx16
"Grand" problem:   leblancx64

In addition, the nohpoly problem will be used for verifying that the
baseline code functionality has been preserved (see below).  Note
for reference that nohpoly would also be considered a "small" problem
by the RFP definition.

E.) Verification

Sample outputs for the APEX test problems are given in the
sample_outputs directory of the APEX PENNANT distribution.  Results
can be verified by comparing the final "energy check" diagnostics,
printed near the end of the standard output, to those in the sample
output file.  All three quantities listed (total, internal, and kinetic
energy) should match to within a relative error of 10^-4.

F.) PENNANT-Specific Run Rules

The baseline PENNANT implementation can be built in both MPI-only and
MPI+OpenMP variants.  For purposes of generating the "base results"
required by the RFP, the MPI+OpenMP variant must be used.

The purpose of the PENNANT benchmark is to demonstrate performance
on an application using general unstructured meshes, that is, meshes
which contain arbitrary polygons with arbitrary connectivity between
them.  Any code changes made by the vendor must preserve this
capability.  That is, although the leblanc* benchmark problems use
meshes with an underlying Cartesian structure, the code may not be
modified in any way that takes advantage of this structure.
As a proof of this, any modified version must be able to correctly
run the nohpoly test problem, in which the mesh is truly unstructured.

Vendors are free to make other modifications to the mesh (different
data structures, numbering, MPI decomposition, ...) so long as support
for general unstructured meshes is preserved.
                                                          
G.) Timing

The PENNANT code prints runtime information that excludes initialization
and finalization stages. It is labeled "hydro cycle run time" (without
quotes) and is near the end of the standard output.

H.) Reporting

For the electronic submission, include all the source and the makefiles
used to build on the target platform.  Include all standard output files.

I.) Authorship

PENNANT was developed at Los Alamos National Laboratory.

