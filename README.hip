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