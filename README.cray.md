## Building and running Pennant on poplar, tulip, and redwood

### Fetching and building

```
git clone -b cray username@redwood.cray.com:/home/groups/amd_and_hpe/pennant-hip.git
cd pennant-hip
source setenv.sh
srun make -f Makefile.hip -j
```

### Running
```
srun -N 1 -n 2 build_hip/pennant test/leblancbig/leblancbig.pnt
```
or alternatively
```
sbatch pennant_sbatch.sh
```
