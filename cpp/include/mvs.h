#pragma once

#include "launch.h"
#include "utils.h"

namespace mvs {

class PMMVS {
public:
    PMMVS();
    ~PMMVS();

    void load_samples(const std::string &dense_folder, const vector<Problem> problems);
    void load_depths(const std::string &dense_folder, const vector<Problem> problems);
    void load_normals(const std::string &dense_folder, const vector<Problem> problems);
    void load_costs(const std::string &dense_folder, const vector<Problem> problems);

    void InuputInitialization(const std::string &dense_folder, const Problem &problem);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    float4 GetPlaneHypothesis(const int32_t index);
    float GetCost(const int32_t index);
    void GetSupportPoints(vector<cv::Point> &support2DPoints);
    vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const vector<cv::Point> &points);
    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int32_t x, const int32_t y);
    void CudaPlanarPriorInitialization(
        const vector<float4> &PlaneParams, const cv::Mat_<float> &masks);
    void release();
    std::tuple<vector<cv::Mat>, vector<cv::Mat>> run_fusion(
        const std::string &dense_folder,
        const vector<Problem> &problems,
        const bool geom_consistency,
        const int32_t geom_consistent);

    PatchMatchParams params;

    int32_t num_images;
    vector<cv::Mat> all_images;
    vector<cv::Mat> all_depths;
    vector<cv::Mat> all_normals;
    vector<cv::Mat> all_costs;
    vector<Camera> all_cameras;

    vector<cv::Mat> images;
    vector<cv::Mat> depths;
    vector<Camera> cameras;

private:
    cudaTextureObject_t texture_images_host[MAX_IMAGES];
    cudaTextureObject_t texture_depths_host[MAX_IMAGES];
    float4 *plane_hypotheses_host;
    float *costs_host;
    float4 *prior_planes_host;
    uint32_t *plane_masks_host;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObject_t *texture_images_cuda;
    cudaTextureObject_t *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float *costs_cuda;
    curandState *rand_states_cuda;
    uint32_t *selected_views_cuda;
    float *depths_cuda;
    float4 *prior_planes_cuda;
    uint32_t *plane_masks_cuda;
};

vector<Problem> generate_sample_list(const std::string dense_folder);
void process_problem(
    const std::string dense_folder,
    const Problem problem,
    const bool geom_consistency,
    const bool planar_prior,
    const bool multi_geometrty,
    PMMVS mvs);

}  // namespace mvs
