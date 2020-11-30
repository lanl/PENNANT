#include "pajama.h"
#include <cstdio>
#include <vector>

int grid_dim = 64;
int block_dim = 256;
int no_threads = block_dim * grid_dim;
int N = no_threads * 4;

int main(){
  // The usual stuff: allocate and fill memory on the host, allocate memory
  // on the device, copy host input data to device input data
  std::vector<float> x(N, 3.0);
  std::vector<float> y(N, 4.0);
    
  float a = 2.0;
  float *x_d, *y_d;

  hipMalloc(&x_d, N * sizeof(float));
  hipMalloc(&y_d, N * sizeof(float));
  hipMemcpy(x_d, x.data(), N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(y_d, y.data(), N * sizeof(float), hipMemcpyHostToDevice);


  // JIT-specific stuff. We are going to call a JIT-ed kernel,
  // __global__ void saxpy(float* x, float* y)  // see kernel.cpp
  // Passing parameters is currently a bit inconventient:
  // first, we must define a struct with the parameter types in order,
  // and fill in the values. Next, we put this in a wrapper array. The
  // wrapper array is then passed when calling the JIT-ed kernel.
  
  struct {
    float* x_d;
    float* y_d;
  } args;
  args.x_d = x_d;
  args.y_d = y_d;
  size_t args_size = sizeof(args);

  void* kernel_args[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
			  HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
			  HIP_LAUNCH_PARAM_END };

  
  // There is no point in JIT-ing if our kernel (saxpy, in kernel.cpp) doesn't
  // have some placeholder values that we are going to replace at run-time, when
  // we know their actual values (so these become compile-time constants for the
  // JIT-ed kernel).
  //
  // You can choose any placeholder strings you want. For the example, I chose
  // a format of ${...}. Since variables in C++ cannot start with a $, you'll get
  // a compilation error if you forget to replace the placeholder.
  //
  // replacement_t is a map from std::string to std::string. Keys are the placeholders,
  // values are the string representations of the values for the placeholders.

  replacement_t replacements {
    { "${N}", std::to_string(N) },
    { "${no_threads}", std::to_string(no_threads) },
    { "${a}", std::to_string(a) }
  };

  // From here it's easy: create a Pajama object, specifying module source file
  // and the replacement specifications for the placeholders in the source file.
  // Next, call a kernel from the source file. Specify the name of the kernel
  // as a C-string, pass grid and block dimensions, the amount of dynamic shared
  // memory in bytes, the stream on which to execute the kernel (with 0 for the
  // default stream), and the wrapping array for the kernel arguments.
  
  Pajama pajama("kernel.cpp", replacements);
  pajama.call("saxpy", grid_dim, block_dim, 0, 0, kernel_args);

  // From here, it's the usual stuff again: allocate host memory for the results,
  // copy results back from device to host, and check if the results are as expected.
  std::vector<float> result(N);
  hipMemcpy(result.data(), y_d, N * sizeof(float), hipMemcpyDeviceToHost);

  int errors = 0;
  for(std::size_t i = 0; i != result.size(); ++i){
    if(result[i] != a * x[i] + y[i]){
      ++errors;
    }
  }

  printf("errors found: %d\n", errors);
}
