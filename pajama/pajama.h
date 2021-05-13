#ifndef PAJAMA_H
#define PAJAMA_H

#include <string>
#include <map>
#include <type_traits>
#include <typeinfo>
#include <sstream>
#include <cxxabi.h>
#include <hip/hip_runtime.h>

using replacement_t = std::map<std::string, std::string>;

class Pajama {
 public:
  Pajama(std::string source_fname, replacement_t replacements = replacement_t(), int rank=0);
  void call(const char* kernel, dim3 grid_dim, dim3 block_dim,
	    unsigned int shmem_bytes, hipStream_t stream, void** kernel_args);
  void call_preloaded(const char* kernel, dim3 grid_dim, dim3 block_dim,
		     unsigned int shmem_bytes, hipStream_t stream, void** kernel_args);
  void compile(); // hard-coded forwarding to either of the two compiler-specific versions
  void hipclang_compile();
  void hcc_compile();
  void load_module();
  void load_kernel(const char* kernel);

 private:
  using kernel_map_t = std::map<const char*, hipFunction_t>;
    
 private:
  std::string source_fname_;
  std::string source_with_replacements_fname_;
  std::string co_fname_;
  std::string source_;
  hipModule_t hipModule_;
  kernel_map_t kernel_map_;
  bool module_compiled_ = false;
  bool module_loaded_ = false;
  int rank_ = 0;
};

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::string>::type
jit_string(T val){
  std::stringstream ss;
  ss.precision(20);
  ss << std::scientific << val;
  return ss.str();
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, std::string>::type
jit_string(T val){
  return std::to_string(val);
}

template<typename T>
std::string
jit_string(T* val){
  std::stringstream ss;
  int status;
  ss << "("
    << abi::__cxa_demangle(typeid(val).name(), 0, 0, &status)
    << ") 0x" << std::hex << *reinterpret_cast<int64_t*>(&val) << 'u';
  return ss.str();
}

inline std::string
jit_string(double2 val){
  std::stringstream ss;
  ss << "double2("
     << jit_string(val.x)
     << ", "
     << jit_string(val.y)
     << ")";
  return ss.str();
}

#endif
