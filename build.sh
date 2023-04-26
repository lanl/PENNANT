. ./setenv.sh
make -j32 -f Makefile.hip
ldd build/pennant # output executable
