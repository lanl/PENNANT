#!/bin/bash

cd ~/Projects/pennant-legion/test/sedovbig
for ver in -bdma ; do
	systag=darwin-$SLURM_JOB_PARTITION
	c=30
	for x in 1 2 4 6 8 10 12 ; do 
		for m in 16 12 10 8 6 4 2 1 ; do 
			binding="-H $(scontrol show hostnames $SLURM_JOB_NODELIST | head -$(expr \( $m + 1 \) / 2) | tr '\n' ',') --map-by ppr:1:socket --bind-to socket"
			for n in 16 12 10 8 6 4 2 1 ; do 
				tag=${m}x$n${ver}-sedov${x}x${c}
				echo ${systag}-${m}x${n}-mpi-sedov${x}x${c}
				if (( ! $(grep -c "hydro cycle" ${systag}-${tag}.log) > 0 )); then
					OMP_NUM_THREADS=$n systag=$systag c=$c m=$m n=$n x=$x \
					  mpirun -n $m $binding -x c -x m -x x -x n -x systag ./gprof_wrapper.sh
					gprof ../../build/pennant gprof-${systag}-${m}x${n}-mpi-sedov${x}x${c}-*/gmon.out > gprof-${systag}-${m}x${n}-mpi-sedov${x}x${c}.gprof				fi
			done
		done
	done
done