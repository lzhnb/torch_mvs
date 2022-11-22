// Copyright 2022 Gorilla-Lab
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_NAME, m) {
    NDArrayConverter::init_numpy();
}