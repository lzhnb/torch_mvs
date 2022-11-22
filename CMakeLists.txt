cmake_minimum_required(VERSION 3.1)
project(
    LIB_MVS
    VERSION 1.0
    LANGUAGES CXX
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# add pybind11 module
add_subdirectory(dependencies/pybind11)

# add the cuda sources
cuda_add_library(
    libmvs SHARED
    cpp/src/bindings.cpp
    cpp/src/ndarray_converter.cpp
    cpp/src/mvs.cpp
    cpp/src/patch_match.cu
    cpp/src/launch.cpp)

target_include_directories(
    libmvs PRIVATE
    cpp/include
    ${OpenCV_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(
    libmvs
    ${OpenCV_LIBS}
    pybind11::module)

set_target_properties(libmvs PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(libmvs PROPERTIES PREFIX "")
target_compile_definitions(libmvs PUBLIC EXTENSION_NAME=libmvs)

pybind11_extension(libmvs)