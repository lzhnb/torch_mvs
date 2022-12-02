#pragma once

#include "primitives.h"

namespace mvs {

Camera ReadCamera(const string &cam_path);
float3 Get3DPointonRefCam(const int32_t x, const int32_t y, const float depth, const Camera camera);

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

}  // namespace mvs
