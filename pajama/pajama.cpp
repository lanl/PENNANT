#include "pajama.h"
#include <string>
#include <fstream>
#include <streambuf>
#include <cstdlib>
#include <stdexcept>

namespace {
  std::string fname_extension_change(const std::string& fname, const std::string& new_ext, bool attach_old_extension);
  std::string read_file_as_string(const std::string& fname);
  void replace(std::string& source, replacement_t replacements);
  void save(const std::string& source, const std::string& fname);
}


Pajama::Pajama(std::string source_fname, replacement_t replacements)
  : source_fname_(source_fname),
    source_with_replacements_fname_(fname_extension_change(source_fname, ".pj", true)),
    co_fname_(fname_extension_change(source_with_replacements_fname_, ".co", false)),
    source_(read_file_as_string(source_fname))
{
  replace(source_, replacements);
  save(source_, source_with_replacements_fname_);
}


void Pajama::call(const char* kernel, dim3 grid_dim, dim3 block_dim,
		  unsigned int shmem_bytes, hipStream_t stream, void** kernel_args){
  if(kernel_map_.find(kernel) == kernel_map_.end()){
    load_kernel(kernel);
  }
  auto result = hipModuleLaunchKernel(kernel_map_[kernel],
				      grid_dim.x, grid_dim.y, grid_dim.z,
				      block_dim.x, block_dim.y, block_dim.z,
				      shmem_bytes, stream,
				      nullptr, kernel_args);
  if(hipSuccess != result){
    std::string message("Pajama::call: failed to launch kernel ");
    message += kernel;
    message += "\n* Check kernel name";
    throw std::runtime_error(message);
  }
}


int Pajama::call_preloaded(const char* kernel, dim3 grid_dim, dim3 block_dim,
			   unsigned int shmem_bytes, hipStream_t stream, void** kernel_args){
  return hipModuleLaunchKernel(kernel_map_[kernel],
			       grid_dim.x, grid_dim.y, grid_dim.z,
			       block_dim.x, block_dim.y, block_dim.z,
			       shmem_bytes, stream,
			       nullptr, kernel_args);
}


void Pajama::load_module(){
  if(not module_compiled_){
    compile();
  }
  auto result = hipModuleLoad(&hipModule_, co_fname_.c_str());
  if(hipSuccess == result){
    module_loaded_ = true;
  } else {
    std::string message("Pajama::load_module: failed to load module ");
    message += co_fname_;
    message += "\n* Check module name and path";
    throw std::runtime_error(message);
  }
}


void Pajama::load_kernel(const char* kernel){
  if(not module_loaded_){
    load_module();
  }
  auto result = hipModuleGetFunction(&kernel_map_[kernel], hipModule_, kernel);
  if(hipSuccess != result){
    std::string message("Pajama::load_kernel: failed to load kernel ");
    message += kernel;
    message += "\n* Check kernel name";
    message += "\n* Make sure JIT kernels are declared 'extern \"C\"'";
    throw std::runtime_error(message);
  }
}


void Pajama::compile(){
  // hard-coded compiler selection. I'm not going to bother
  // with an elegant solution, since in the near future, HCC
  // will be a compiler of the past.
  hipclang_compile();
}


void Pajama::hipclang_compile(){
  std::string command("hipcc ");
  command += "--cuda-device-only ";
  command += "-O3 ";
  command += "-c ";
  command += source_with_replacements_fname_.c_str();
  command += " -o ";
  command += co_fname_;
  auto result = system(command.c_str());
  if(result){
    std::string message("Pajama::compile: failed to compile ");
    message += source_with_replacements_fname_;
    message += " into ";
    message += co_fname_;
    throw std::runtime_error(message);
  }
}

void Pajama::hcc_compile(){
  std::string command("hipcc ");
  command += "--genco ";
  command += "-f=\"-O3\" -t gfx906 "; // hard-coded architecture, change to taste.
  command += source_with_replacements_fname_.c_str();
  command += " -o ";
  command += co_fname_;
  auto result = system(command.c_str());
  if(result){
    std::string message("Pajama::compile: failed to compile ");
    message += source_with_replacements_fname_;
    message += " into ";
    message += co_fname_;
    throw std::runtime_error(message);
  }
}
namespace {
  std::string fname_extension_change(const std::string& fname, const std::string& new_ext, bool attach_old_extension){
    auto dot = fname.find_last_of('.');
    auto new_fname = fname.substr(0,dot) + new_ext;
    if(attach_old_extension){
      new_fname += fname.substr(dot);
    }
    return new_fname;
  }


  std::string read_file_as_string(const std::string& fname){
    std::ifstream f(fname.c_str());
    std::string s;
    f.seekg(0, std::ios::end);
    s.reserve(f.tellg());
    f.seekg(0, std::ios::beg);
    s.assign((std::istreambuf_iterator<char>(f)),
	     std::istreambuf_iterator<char>());
    return s;      
  }


  void replace(std::string& source, const std::string& key, const std::string& val){
    auto pos = source.find(key);
    while(pos != std::string::npos){
      source.replace(pos, key.size(), val);
      pos = source.find(key, pos + key.size());
    }
  }


  void replace(std::string& source, replacement_t replacements){
    for(auto [key, val]: replacements){
      replace(source, key, val);
    }
  }


  void save(const std::string& source, const std::string& fname){
    std::ofstream f(fname);
    f << source;
  }
}


