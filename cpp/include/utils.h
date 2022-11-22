#pragma once

#include "launch.h"

namespace mvs {

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);
void RescaleImageAndCamera(
    cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud(
    const std::string &plyFilePath, const vector<PointList> &pc);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string &file, const int line);
void CudaCheckError(const char *file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations    = 3;
    int patch_size        = 11;
    int num_images        = 5;
    int radius_increment  = 2;
    float sigma_spatial   = 5.0f;
    float sigma_color     = 3.0f;
    int top_k             = 4;
    float baseline        = 0.54f;
    float depth_min       = 0.0f;
    float depth_max       = 1.0f;
    float disparity_min   = 0.0f;
    float disparity_max   = 1.0f;
    bool geom_consistency = false;
    bool multi_geometry   = false;
    bool planar_prior     = false;
};

}  // namespace mvs
