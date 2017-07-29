#!/bin/bash

cd ~/Projects/pennant-mpi/test/sedovbig
systag=darwin-$SLURM_JOB_PARTITION
c=30
for x in 16 12 8 6 4 2 1 ; do
		m=$x 
		binding="-H $(scontrol show hostnames $SLURM_JOB_NODELIST | head -$(expr \( $m + 1 \) / 2) | tr '\n' ',') --map-by ppr:1:socket --bind-to socket"
		n=$x;
		OMP_NUM_THREADS=$n systag=$systag c=$c m=$m n=$n x=$x \
		  mpirun -n $m $binding -x c -x m -x x -x n -x systag ./gprof_wrapper.sh
		gprof ../../build/pennant gprof-${systag}-${m}x${n}-mpi-sedov${x}x${c}-*/gmon.out > gprof-${systag}-${m}x${n}-mpi-sedov${x}x${c}.gprof
	done
done