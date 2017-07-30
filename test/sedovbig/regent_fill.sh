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
				echo ${systag}-${tag}
				if [ -n $(grep ${systag}-${tag}.log ${systag}-${tag}.log) ]; then
					mpirun -n $m $binding ../../../regent.py ../../pennant${ver}.rg sedovbig${x}x${c}.pnt \
					  -npieces $(expr $m \* $n) -numpcx 1 -numpcy $(expr $m \* $n) \
					  -seq_init 0 -par_init 1 -interior 0 \
					  -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize ${n} -fvectorize-unsafe 1 \
					  -print_ts 1 -prune 0 -ll:cpu $n -ll:util 1 -ll:dma 1 -ll:csize 30000 \
					  -lg:prof 1 -lg:prof_logfile trace-${tag}- 2>&1 | tee ${systag}-${tag}.log
					../../../scripts/summarize.py ${systag}-${tag}.log >> ${systag}-${tag}.log
					~/Projects/legion/tools/legion_prof.py -f -o trace-${systag}-${tag} trace-${tag}-*.gz				fi
			done
		done
	done
done