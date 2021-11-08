#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J pennant_sbatch
#SBATCH -o output.%j
#SBATCH -e error.%j

module load craype-x86-naples               
module load craype-network-infiniband       
module load shared                          
module load slurm                   
module load gcc/8.1.0                       
module load rocm/3.5.0
module load cray-mvapich2_nogpu_gnu

problem=leblancbig
srun build_hip/pennant test/${problem}/${problem}.pnt

wait
mkdir -p output
mv output.* error.* output

