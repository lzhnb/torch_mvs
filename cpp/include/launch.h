#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <sys/stat.h>   // mkdir
#include <sys/types.h>  // mkdir
#include <vector_types.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "iomanip"

#define MAX_IMAGES 256

namespace mvs {

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int height;
    int width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle(const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3)
        : pt1(_pt1), pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

std::vector<Problem> generate_sample_list(const std::string dense_folder);
void process_problem(
    const std::string dense_folder,
    const Problem problem,
    bool geom_consistency,
    bool planar_prior,
    bool multi_geometrty);
std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>> run_fusion(
    const std::string &dense_folder,
    const std::vector<Problem> &problems,
    const bool geom_consistency,
    const int32_t geom_consistent);

}  // namespace mvs
