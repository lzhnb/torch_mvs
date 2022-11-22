#pragma once

#include "launch.h"
#include "utils.h"

namespace mvs {

class PMMVS {
public:
    PMMVS();
    ~PMMVS();

    void InuputInitialization(const std::string &dense_folder, const Problem &problem);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetPlanarPriorParams();
    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    void GetSupportPoints(std::vector<cv::Point> &support2DPoints);
    std::vector<Triangle> DelaunayTriangulation(
        const cv::Rect boundRC, const std::vector<cv::Point> &points);
    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
    float GetMinDepth();
    float GetMaxDepth();
    void CudaPlanarPriorInitialization(
        const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks);

private:
    int num_images;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> depths;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float *costs_host;
    float4 *prior_planes_host;
    unsigned int *plane_masks_host;
    PatchMatchParams params;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float *costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
    float4 *prior_planes_cuda;
    unsigned int *plane_masks_cuda;
};

}  // namespace mvs
