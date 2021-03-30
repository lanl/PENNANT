module load craype-x86-naples               
module load craype-network-infiniband       
module load shared                          
module load slurm                   
module load gcc/8.1.0
module load rocm

module unload cray-mvapich2
module load cray-mvapich2_nogpu_gnu

export MPI_HOME=$(echo "${PE_GNU_FIXED_PKGCONFIG_PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/lib/pkgconfig,,')


