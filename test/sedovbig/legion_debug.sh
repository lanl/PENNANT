#!/bin/bash

cd ~/Projects/pennant-legion/test/sedovbig
for ver in -debug ; do
	systag=darwin-$SLURM_JOB_PARTITION
	c=30
	export GASNET_BACKTRACE=1
	export DEBUG=1
	for x in 12 ; do # 10 8 6 4 2 1 ; do
		m=$x 
		binding="-H $(scontrol show hostnames $SLURM_JOB_NODELIST | head -$(expr \( $m + 1 \) / 2) | tr '\n' ',') --map-by ppr:1:socket --bind-to socket"
		n=$x; 
		tag=${m}x$n${ver}-sedov${x}x${c}
		mpirun -x DEBUG -x GASNET_BACKTRACE -n $m $binding /bin/time -v ../../pennant${ver} $(expr $m \* $n) ./sedovbig${x}x${c}.pnt \
		-ll:cpu $n -ll:util 1 -ll:dma 1 -ll:rsize 0 -ll:csize 30000 -ll:stacksize 20 \
		-lg:prof 4 -lg:prof_logfile trace-${tag}- 2>&1 > ${systag}-${tag}.log
		~/Projects/legion/tools/legion_prof.py -f -o trace-${systag}-${tag} trace-${tag}-*.gz
	done
done
