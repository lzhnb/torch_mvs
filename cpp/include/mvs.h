#pragma once

#include "primitives.h"

namespace mvs {

Camera read_camera(const string &cam_path);
float3 get_3D_point_on_ref_cam(const int32_t x, const int32_t y, const float depth, const Camera camera);

class PMMVS {
public:
    PMMVS();
    ~PMMVS();

    void load_samples(const string &dense_folder, const vector<Problem> problems);
    void load_geometry(
        const vector<cv::Mat> &depth_maps,
        const vector<cv::Mat> &normal_maps,
        const vector<cv::Mat> &cost_maps);

    void inuput_initialization(const Problem &problem);
    void cuda_space_initialization(const Problem &problem);
    void run_patch_match();
    float4 get_plane_hypothesis(const int32_t index);
    float get_cost(const int32_t index);
    void get_support_points(vector<cv::Point> &support2DPoints);
    vector<Triangle> delaunay_triangulation(const cv::Rect boundRC, const vector<cv::Point> &points);
    float4 get_prior_plane_params(const Triangle triangle, const cv::Mat_<float> depths);
    float get_depth_from_plane_param(const float4 plane_hypothesis, const int32_t x, const int32_t y);
    void cuda_planar_prior_initialization(
        const vector<float4> &PlaneParams, const cv::Mat_<float> &masks);
    void release();

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

vector<Problem> generate_sample_list(const string dense_folder);
tuple<cv::Mat, cv::Mat, cv::Mat> process_problem(
    const string dense_folder,
    const Problem problem,
    const bool geom_consistency,
    const bool planar_prior,
    const bool multi_geometrty,
    PMMVS mvs);

tuple<vector<cv::Mat>, vector<cv::Mat>> run_fusion(
    const string &dense_folder,
    const vector<Problem> &problems,
    const vector<cv::Mat> &depth_maps,
    const vector<cv::Mat> &normal_maps,
    const bool geom_consistency,
    const int32_t geom_consistent);

}  // namespace mvs
