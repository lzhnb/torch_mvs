#pragma once

#include "launch.h"
#include "utils.h"

namespace mvs {

class PMMVS {
public:
    PMMVS();
    ~PMMVS();

    void load_samples(const std::string &dense_folder, const vector<Problem> problems);

    void InuputInitialization(const std::string &dense_folder, const Problem &problem);
    void Colmap2MVS(const std::string &dense_folder, vector<Problem> &problems);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetPlanarPriorParams();
    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    void GetSupportPoints(vector<cv::Point> &support2DPoints);
    vector<Triangle> DelaunayTriangulation(
        const cv::Rect boundRC, const vector<cv::Point> &points);
    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
    void CudaPlanarPriorInitialization(
        const vector<float4> &PlaneParams, const cv::Mat_<float> &masks);
    void release();

    PatchMatchParams params;

private:
    int num_images;
    vector<cv::Mat> all_images;
    vector<Camera> all_cameras;

    vector<cv::Mat> images;
    vector<cv::Mat> depths;
    vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float *costs_host;
    float4 *prior_planes_host;
    unsigned int *plane_masks_host;

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

vector<Problem> generate_sample_list(const std::string dense_folder);
void process_problem(
    const std::string dense_folder,
    const Problem problem,
    const bool geom_consistency,
    const bool planar_prior,
    const bool multi_geometrty,
    PMMVS mvs);
std::tuple<vector<cv::Mat>, vector<cv::Mat>> run_fusion(
    const std::string &dense_folder,
    const vector<Problem> &problems,
    const bool geom_consistency,
    const int32_t geom_consistent);

}  // namespace mvs
