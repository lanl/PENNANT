#!/bin/bash

cd ~/Projects/pennant-legion/test/sedovbig
for ver in -bdma{,-long,-wide}; do
	systag=psg-$SLURM_JOB_PARTITION
	c=30
	for m in 16 12 8 6 4 2 1 ; do 
		binding="-H $(scontrol show hostnames $SLURM_JOB_NODELIST | head -$(expr \( $m + 1 \) / 2) | tr '\n' ',') --map-by ppr:1:socket --bind-to socket"
		for n in 1 2 4 6 8 10 12 16 ; do 
			for x in 1 2 4 6 8 12 ; do 
				tag=${m}x$n${ver}-sedov${x}x${c}
				mpirun -n $m $binding ../../pennant${ver} $(expr $m \* $n) ./sedovbig${x}x${c}.pnt \
				-ll:cpu $n -ll:util 1 -ll:dma 1 -ll:csize 10000 -ll:stacksize 20 \
				-lg:prof $m -lg:prof_logfile trace-${tag}- 2>&1 > ${systag}-${tag}.log
				~/Projects/legion/tools/legion_prof.py -f -o trace-${systag}-${tag} trace-${tag}-*.gz
			done
		done
	done
done