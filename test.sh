#!/bin/bash
set -euxo pipefail

if [ -z ${1:-} ] || [ $1 != "m" ]; then
    mf=Makefile.hip
else
    mf=Makefile.mpi
fi

make -f ${mf} clean
make -f ${mf} -j
./build_hip/pennant test/leblancbig/leblancbig.pnt
