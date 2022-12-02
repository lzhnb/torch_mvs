#include "mvs.h"

#include <cstdarg>

namespace mvs {

Camera read_camera(const string &cam_path) {
    Camera camera;
    std::ifstream file(cam_path);

    string line;
    file >> line;

    for (int32_t i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int32_t i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    return camera;
}

float3 get_3D_point_on_ref_cam(
    const int32_t x, const int32_t y, const float depth, const Camera camera) {
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    return pointX;
}

PMMVS::PMMVS() {}

PMMVS::~PMMVS() {}

void PMMVS::release() {
    delete[] plane_hypotheses_host;
    delete[] costs_host;

    for (int32_t i = 0; i < num_images; ++i) {
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_images_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);

    if (params.geom_consistency) {
        for (int32_t i = 0; i < num_images; ++i) {
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }

    if (params.planar_prior) {
        delete[] prior_planes_host;
        delete[] plane_masks_host;

        cudaFree(prior_planes_cuda);
        cudaFree(plane_masks_cuda);
    }
}

void PMMVS::inuput_initialization(const Problem &problem) {
    images.clear();
    cameras.clear();

    images.push_back(all_images[problem.ref_image_id]);
    cameras.push_back(all_cameras[problem.ref_image_id]);

    const int32_t num_src_images = problem.num_ngb;
    for (int32_t i = 0; i < num_src_images; ++i) {
        images.push_back(all_images[problem.src_image_ids[i]]);
        cameras.push_back(all_cameras[problem.src_image_ids[i]]);
    }

    params.depth_min     = cameras[0].depth_min * 0.6f;
    params.depth_max     = cameras[0].depth_max * 1.2f;
    params.num_images    = (int)images.size();
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

    if (params.geom_consistency) {
        depths.clear();
        depths.push_back(all_depths[problem.ref_image_id]);

        int32_t num_src_images = problem.num_ngb;
        for (int32_t i = 0; i < num_src_images; ++i) {
            depths.push_back(all_depths[problem.src_image_ids[i]]);
        }
    }
}

void PMMVS::cuda_space_initialization(const Problem &problem) {
    num_images = (int)images.size();

    size_t image_size = 0;
    for (int32_t i = 0; i < num_images; ++i) {
        int32_t rows = images[i].rows;
        int32_t cols = images[i].cols;

        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray(
            cuArray[i],
            0,
            0,
            images[i].ptr<float>(),
            images[i].step[0],
            cols * sizeof(float),
            rows,
            cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_images_host[i]), &resDesc, &texDesc, NULL);
        image_size += sizeof(texture_images_host[i]);
    }
    cudaMalloc((void **)&texture_images_cuda, image_size);
    cudaMemcpy(texture_images_cuda, texture_images_host, image_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc(
        (void **)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void **)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    cudaMalloc(
        (void **)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc(
        (void **)&selected_views_cuda, sizeof(uint32_t) * (cameras[0].height * cameras[0].width));

    cudaMalloc(
        (void **)&depths_cuda,
        sizeof(float) * (cameras[0].height * cameras[0].width));  // Updated by Qingshan 2020-01-15

    if (params.geom_consistency) {
        size_t depth_size = 0;
        for (int32_t i = 0; i < num_images; ++i) {
            int32_t rows = depths[i].rows;
            int32_t cols = depths[i].cols;

            cudaChannelFormatDesc channelDesc =
                cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray(
                cuDepthArray[i],
                0,
                0,
                depths[i].ptr<float>(),
                depths[i].step[0],
                cols * sizeof(float),
                rows,
                cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType         = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0]   = cudaAddressModeWrap;
            texDesc.addressMode[1]   = cudaAddressModeWrap;
            texDesc.filterMode       = cudaFilterModeLinear;
            texDesc.readMode         = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_depths_host[i]), &resDesc, &texDesc, NULL);
            depth_size += sizeof(texture_depths_host[i]);
        }
        cudaMalloc((void **)&texture_depths_cuda, depth_size);
        cudaMemcpy(texture_depths_cuda, texture_depths_host, depth_size, cudaMemcpyHostToDevice);

        cv::Mat_<float> ref_depth      = all_depths[problem.ref_image_id];
        cv::Mat_<cv::Vec3f> ref_normal = all_normals[problem.ref_image_id];
        cv::Mat_<float> ref_cost       = all_costs[problem.ref_image_id];
        depths.push_back(ref_depth);
        int32_t width  = ref_depth.cols;
        int32_t height = ref_depth.rows;
        for (int32_t col = 0; col < width; ++col) {
            for (int32_t row = 0; row < height; ++row) {
                int32_t center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x            = ref_normal(row, col)[0];
                plane_hypothesis.y            = ref_normal(row, col)[1];
                plane_hypothesis.z            = ref_normal(row, col)[2];
                plane_hypothesis.w            = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
                costs_host[center]            = ref_cost(row, col);
            }
        }

        cudaMemcpy(
            plane_hypotheses_cuda,
            plane_hypotheses_host,
            sizeof(float4) * width * height,
            cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }
}

void PMMVS::cuda_planar_prior_initialization(
    const vector<float4> &PlaneParams, const cv::Mat_<float> &masks) {
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc(
        (void **)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new uint32_t[cameras[0].height * cameras[0].width];
    cudaMalloc(
        (void **)&plane_masks_cuda, sizeof(uint32_t) * (cameras[0].height * cameras[0].width));

    for (int32_t i = 0; i < cameras[0].width; ++i) {
        for (int32_t j = 0; j < cameras[0].height; ++j) {
            int32_t center           = j * cameras[0].width + i;
            plane_masks_host[center] = (uint32_t)masks(j, i);
            if (masks(j, i) > 0) {
                prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
            }
        }
    }

    cudaMemcpy(
        prior_planes_cuda,
        prior_planes_host,
        sizeof(float4) * (cameras[0].height * cameras[0].width),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        plane_masks_cuda,
        plane_masks_host,
        sizeof(uint32_t) * (cameras[0].height * cameras[0].width),
        cudaMemcpyHostToDevice);
}

float4 PMMVS::get_plane_hypothesis(const int32_t index) { return plane_hypotheses_host[index]; }

float PMMVS::get_cost(const int32_t index) { return costs_host[index]; }

void PMMVS::get_support_points(vector<cv::Point> &support2DPoints) {
    support2DPoints.clear();
    const int32_t step_size = 5;
    const int32_t width     = cameras[0].width;
    const int32_t height    = cameras[0].height;
    for (int32_t col = 0; col < width; col += step_size) {
        for (int32_t row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int32_t c_bound = std::min(width, col + step_size);
            int32_t r_bound = std::min(height, row + step_size);
            for (int32_t c = col; c < c_bound; ++c) {
                for (int32_t r = row; r < r_bound; ++r) {
                    int32_t center = r * width + c;
                    if (get_cost(center) < 2.0f && min_cost > get_cost(center)) {
                        temp_point = cv::Point(c, r);
                        min_cost   = get_cost(center);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

vector<Triangle> PMMVS::delaunay_triangulation(
    const cv::Rect boundRC, const vector<cv::Point> &points) {
    if (points.empty()) {
        return vector<Triangle>();
    }

    vector<Triangle> results;

    vector<cv::Vec6f> temp_results;
    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points) {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    subdiv2d.getTriangleList(temp_results);

    for (const auto temp_vec : temp_results) {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        results.push_back(Triangle(pt1, pt2, pt3));
    }
    return results;
}

float4 PMMVS::get_prior_plane_params(const Triangle triangle, const cv::Mat_<float> depths) {
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);

    float3 ptX1 = get_3D_point_on_ref_cam(
        triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0]);
    float3 ptX2 = get_3D_point_on_ref_cam(
        triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0]);
    float3 ptX3 = get_3D_point_on_ref_cam(
        triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x), cameras[0]);

    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B);
    float4 n4 =
        make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float norm2 = sqrt(pow(n4.x, 2) + pow(n4.y, 2) + pow(n4.z, 2));
    if (n4.w < 0) {
        norm2 *= -1;
    }
    n4.x /= norm2;
    n4.y /= norm2;
    n4.z /= norm2;
    n4.w /= norm2;

    return n4;
}

float PMMVS::get_depth_from_plane_param(
    const float4 plane_hypothesis, const int32_t x, const int32_t y) {
    return -plane_hypothesis.w * cameras[0].K[0] /
           ((x - cameras[0].K[2]) * plane_hypothesis.x +
            (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5]) * plane_hypothesis.y +
            cameras[0].K[0] * plane_hypothesis.z);
}

}  // namespace mvs
