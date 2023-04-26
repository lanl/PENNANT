#!/bin/bash
#SBATCH -t 10:00
#SBATCH -o job.o%j
#
# Since these two were left out above:
# --- -J sedovflatx40
# --- -N 70
# --- -A allocation_name
# We can define the problem name and node count when submitting job,
#
#     sbatch -A allocation_name -J sedovflatx40 -N 40 ../slurm.sh

# store relevant information from slurm
TEST="$SLURM_JOB_NAME" # sedovflatx120, etc.
nodes=$SLURM_JOB_NUM_NODES
tasks=$((nodes*8))

PENNANT="$BUILD_DIR/build/pennant"

# test script input data
if [ ! -s "$BUILD_DIR/setenv.sh" ]; then
  echo "File $BUILD_DIR/setenv.sh not found."
  exit 1
fi
if [ ! -x "$PENNANT" ]; then
  echo "Executable $PENNANT not found."
  exit 1
fi
if [ ! -s "$BUILD_DIR/test/$TEST" ]; then
  echo "File $BUILD_DIR/test/$TEST not found."
  exit 1
fi
if [ ! -s "$BUILD_DIR/test/$TEST/$TEST.pnt" ]; then
  echo "File $BUILD_DIR/test/$TEST/$TEST.pnt not found."
  exit 1
fi

# setup and show environment at start
source "$BUILD_DIR/setenv.sh"

module list >job.modules 2>&1
env &> job.environ
scontrol show hostnames > job.nodes

# copy input files
cp "$BUILD_DIR/test/$TEST"/* .

# short-hand for srun to use
SRUN="srun -N $nodes -n $tasks -c7 --gpus-per-task 1 --gpu-bind=closest"

# see affinity using:
# /bin/bash -c 'echo $(hostname) $(grep Cpus_allowed_list /proc/self/status) GPUS: $ROCR_VISIBLE_DEVICES' | sort -n

echo "running: $SRUN $PENNANT $TEST.pnt"
time $SRUN $PENNANT $TEST.pnt >run.log 2>&1
echo $? >retcode.txt
