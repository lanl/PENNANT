module load craype-x86-naples               
module load craype-network-infiniband       
module load shared                          
module load slurm                   
module load gcc
module load rocm/3.6.0
#module use /home/users/twhite/share/modulefiles
#module load ompi



module unload cray-mvapich2
module load cray-mvapich2_nogpu_gnu

export MPI_HOME=$(echo "${PE_GNU_FIXED_PKGCONFIG_PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/lib/pkgconfig,,')


