// Copyright 2022 Gorilla-Lab
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"
#include "launch.h"
#include "ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_NAME, m) {
    NDArrayConverter::init_numpy();

    py::class_<Problem>(m, "Problem")
        .def(py::init<>())
        .def_readwrite("ref_image_id", &Problem::ref_image_id)
        .def_readwrite("src_image_ids", &Problem::src_image_ids);

    m.def("generate_sample_list", &generate_sample_list);
    m.def("process_problem", &process_problem);
    m.def("run_fusion", &run_fusion);
}