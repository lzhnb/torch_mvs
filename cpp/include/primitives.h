#pragma once

// opencv
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "iomanip"

#include "common.h"

#define MAX_IMAGES 256

namespace mvs {

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int32_t height;
    int32_t width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int32_t ref_image_id;
    int32_t src_image_ids[MAX_IMAGES];
    int32_t num_ngb;
};

struct PatchMatchParams {
    int32_t max_iterations   = 3;
    int32_t patch_size       = 11;
    int32_t num_images       = 5;
    int32_t radius_increment = 2;
    float sigma_spatial      = 5.0f;
    float sigma_color        = 3.0f;
    int32_t top_k            = 4;
    float baseline           = 0.54f;
    float depth_min          = 0.0f;
    float depth_max          = 1.0f;
    float disparity_min      = 0.0f;
    float disparity_max      = 1.0f;
    bool geom_consistency    = false;
    bool multi_geometry      = false;
    bool planar_prior        = false;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle(const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3)
        : pt1(_pt1), pt2(_pt2), pt3(_pt3) {}
};

}  // namespace mvs
