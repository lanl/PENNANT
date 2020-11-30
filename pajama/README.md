# Pajama - Pathetic Approach to JIT-Aware Module Application

Workaround project for hipRTC, which doesn't seem to be super-mature. Pajama is
throw-away code. It's ugly. It's pathetic.

Pajama provides an interface for run-time compilation of GPU kernels into 
kernel modules. The interface allows for replacement of placeholder values
in the kernel source code by values that are known at JIT time.

Driver code is in 'example.cpp'. Through Pajama, it loads a source
file with GPU code (kernel.cpp), replaces placeholders in the source
file with parameters determined at run-time, writes out the modified
source file, compiles it to a module, loads the module, loads a kernel
from the module, and invokes the kernel.

Documentation on how to replace placeholders in the module source
file, and on how to pass kernel arguments to a kernel from a module,
can be found in comments in example.cpp.

Tested with hip-clang. If you want to use this with hcc, you need to
change Pajama::compile in pajama.cpp: replace '--cuda-device-only'
with '--genco'.

## Building and running
```
make
./example
```

## Future plans

Passing kernel arguments to kernels in modules is clumsy. I plan to
toy with variadic template arguments to get the interface of calling a
kernel identical to that of hipLaunchKernelGGL (except for the first
argument, which must be the name of the kernel as a C-style string).

Error handling: Pajama functions return error codes, but since Pajama
functions call other Pajama functions, it can be hard to figure out
where things go wrong. I plan to switch to exception handling, so that
I can pass a descriptive error message from any level in the call stack.
