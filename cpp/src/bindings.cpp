// Copyright 2022 Gorilla-Lab
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "launch.h"
#include "mvs.h"
#include "ndarray_converter.h"
#include "utils.h"

namespace py = pybind11;

namespace mvs {

PYBIND11_MODULE(EXTENSION_NAME, m) {
    NDArrayConverter::init_numpy();

    py::class_<PMMVS>(m, "PMMVS")
        .def(py::init<>())
        .def("load_samples", &PMMVS::load_samples)
        // .def("load_normals", &PMMVS::load_normals)
        // .def("load_costs", &PMMVS::load_costs)
        .def("load_geometry", &PMMVS::load_geometry)
        .def("release", &PMMVS::release)
        .def_readwrite("params", &PMMVS::params);

    py::class_<PatchMatchParams>(m, "PatchMatchParams")
        .def(py::init<>())
        .def_readwrite("max_iterations", &mvs::PatchMatchParams::max_iterations)
        .def_readwrite("patch_size", &mvs::PatchMatchParams::patch_size)
        .def_readwrite("num_images", &mvs::PatchMatchParams::num_images)
        .def_readwrite("radius_increment", &mvs::PatchMatchParams::radius_increment)
        .def_readwrite("sigma_spatial", &mvs::PatchMatchParams::sigma_spatial)
        .def_readwrite("sigma_color", &mvs::PatchMatchParams::sigma_color)
        .def_readwrite("top_k", &mvs::PatchMatchParams::top_k)
        .def_readwrite("baseline", &mvs::PatchMatchParams::baseline)
        .def_readwrite("depth_min", &mvs::PatchMatchParams::depth_min)
        .def_readwrite("depth_max", &mvs::PatchMatchParams::depth_max)
        .def_readwrite("disparity_min", &mvs::PatchMatchParams::disparity_min)
        .def_readwrite("disparity_max", &mvs::PatchMatchParams::disparity_max)
        .def_readwrite("geom_consistency", &mvs::PatchMatchParams::geom_consistency)
        .def_readwrite("multi_geometry", &mvs::PatchMatchParams::multi_geometry)
        .def_readwrite("planar_prior", &mvs::PatchMatchParams::planar_prior);

    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("height", &Camera::height)
        .def_readwrite("width", &Camera::width)
        .def_readwrite("depth_min", &Camera::depth_min)
        .def_readwrite("depth_max", &Camera::depth_max);

    py::class_<Problem>(m, "Problem")
        .def(py::init<>())
        .def_readwrite("ref_image_id", &Problem::ref_image_id)
        // .def_readwrite("src_image_ids", &Problem::src_image_ids)
        .def_readwrite("num_ngb", &Problem::num_ngb);

    m.def("generate_sample_list", &generate_sample_list);
    m.def("process_problem", &process_problem);
    m.def("run_fusion", &run_fusion);
}

}  // namespace mvs