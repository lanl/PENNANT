# PENNANT (hip branch)

This is a fork of the hip branch of https://github.com/lanl/PENNANT.git.

It can be obtained from:
git clone --branch frontier https://github.com/frobnitzem/PENNANT.git

Compile it with:

    cd PENNANT
    bash build.sh

Setup and run a test problem using:

    cd PENNANT # if you're not already in this dir.
    export BUILD_DIR=$PWD

    mkdir my_test4 # this directory name/location is not sigificant
    cd my_test4
    # job name and node count below are important (see table)
    sbatch -A allocation_name -J sedovflatx4 -N 1 ../slurm.sh
    cd ..
    # repeat last 4 steps to queue more tests

## Modified / Added Files:

0. README.md    (new)   this helpful write-up
1. setenv.sh    (new)   setup script for modules and compiler flags on Frontier
2. Makefile.hip (updated) simplified makefile for HIP
3. build.sh     (new)   script to run make
4. slurm.sh     (new)   source file for all test slurm jobs

## List of Test Problems:

The `sedovflatx(n)` problems are a series of scaled versions
of sedovflatx.  The simulation is 2D, so the work (and DOFs)
scale as `n*n`.

| problem name  | nodes |
| ------------  | ----- |
| sedovflatx4   |     1 |
| sedovflatx40  |    70 |
| sedovflatx120 |   600 |

## Description of `slurm.sh`

When run, the batch script `slurm.sh` checks for the
presence of appropriate input files, then prints out its modules,
environment variables, and node list to "job.{modules,environ,nodes}".

The slurm output file itself (`job.o%j`) will contain the
srun line used and the result of the `time` system call on the srun.

The output file `run.log` contains the stdout from the pennant program,
including its per-step timing and final energy info.
