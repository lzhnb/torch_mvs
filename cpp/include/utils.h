#pragma once

#include "launch.h"

namespace mvs {

int32_t readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int32_t readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int32_t writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int32_t writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);
float3 Get3DPointonWorld(const int32_t x, const int32_t y, const float depth, const Camera camera);
float3 Get3DPointonRefCam(const int32_t x, const int32_t y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string &file, const int32_t line);
void CudaCheckError(const char *file, const int32_t line);

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
