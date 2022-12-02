#pragma once

// CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

// std lib
#include <stdexcept>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <sys/stat.h>   // mkdir
#include <sys/types.h>  // mkdir

using std::vector;
using std::tuple;
using std::string;

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK(x)                                                                         \
    do {                                                                                      \
        cudaError_t result = x;                                                               \
        if (result != cudaSuccess)                                                            \
            throw std::runtime_error(                                                         \
                string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result)); \
    } while (0)
