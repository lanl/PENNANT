## General information on Pennant
See the [README](README) file, or the [PDF documentation](http://gitlab1.amd.com/rvanoo/pennant-hip/blob/master/doc/pennantdoc.pdf).

## Getting the HIP version of Pennant

```
git clone -b hip http://gitlab1.amd.com/rvanoo/pennant-hip.git
```

## Building the HIP version Pennant
The HIP port of Pennant depends on HIP versions of Thrust and CUB. To install these:

```
git clone https://github.com/ROCmSoftwarePlatform/Thrust.git
cd Thrust/thrust/system/cuda/detail
git clone -b hip_port_1.7.3 https://github.com/ROCmSoftwarePlatform/cub-hip.git
ln -s cub-hip cub
sed -i.bak 's/32 - LOGICAL_WARP/64 - LOGICAL_WARP/' ./cub-hip/cub/warp/specializations/warp_scan_smem.cuh
```
Next, in Pennant's Makefile.hip, set

```
THRUSTDIR := /path/to/hip/version/of/Thrust
```

To build Pennant:

```
make -f Makefile.hip -j `nproc`
```

## Running Pennant
After building, Pennant's executable is in the build_hip directory. Pennant 
comes with a number of test inputs in subdirectories of the test directory. The
*.pnt files in each subdirectory are Pennant's input files, to be provided as
argument to the binary, e.g.:
```
build_hip/pennant test/leblancbig/leblancbig.pnt
```

Pennant runs an iterative process; each iteration is called a "cycle". To limit
the number of cycles Pennant runs on a particular input, add a line with 
`cstop <n>` to the *.pnt file. E.g., to limit to 100 iterations, add
```
cstop 100
```