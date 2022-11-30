#include "mvs.h"

namespace mvs {

__device__ float3
get_3D_point_on_world(const float x, const float y, const float depth, const Camera camera) {
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x      = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y      = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z      = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

__device__ float2 project_on_camera(const float3 PointX, const mvs::Camera camera, float& depth) {
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    float2 point;
    depth   = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
    return point;
}

__device__ float get_angle(const float3 v1, const float3 v2) {
    float dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    float angle       = acosf(dot_product);
    // if angle is not a number the dot product was 1 and thus the two vectors
    // should be identical --> return 0
    if (angle != angle) return 0.0f;

    return angle;
}

__global__ void fuions_kernel(
    const float* __restrict__ depths_ptr,
    const float* __restrict__ normals_ptr,
    const Camera* __restrict__ cameras_ptr,
    const Problem* __restrict__ problems_ptr,
    const int32_t rows,
    const int32_t cols,
    const int32_t geom_const,
    // output
    bool* __restrict__ masks_ptr,
    float* __restrict__ proj_depth_ptr,
    float* __restrict__ proj_normal_ptr) {
    const int32_t ref_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ref_idx >= rows * cols) return;

    const int32_t ref_r = ref_idx / cols, ref_c = ref_idx % cols;

    float ref_depth   = depths_ptr[ref_idx];
    float3 ref_normal = make_float3(
        normals_ptr[ref_idx * 3 + 0], normals_ptr[ref_idx * 3 + 1], normals_ptr[ref_idx * 3 + 2]);

    if (ref_depth <= 0.0) return;

    int2 used_list[MAX_IMAGES];
    for (int32_t i = 0; i < problems_ptr->num_ngb; ++i) {
        used_list[i] = make_int2(-1, -1);
    }

    if (masks_ptr[problems_ptr->ref_image_id * rows * cols + ref_idx]) return;

    const int32_t ref_id = problems_ptr->ref_image_id;
    float3 ref_point     = get_3D_point_on_world(ref_c, ref_r, ref_depth, cameras_ptr[ref_id]);
    float3 const_point   = ref_point;
    float3 const_normal  = ref_normal;
    int32_t num_const    = 0;

    for (int32_t j = 0; j < problems_ptr->num_ngb; ++j) {
        const int32_t src_id     = problems_ptr->src_image_ids[j];
        const int32_t src_offset = src_id * rows * cols;
        float proj_depth;
        const float2 point  = project_on_camera(ref_point, cameras_ptr[src_id], proj_depth);
        const int32_t src_r = int(point.y + 0.5f);
        const int32_t src_c = int(point.x + 0.5f);
        if (src_c >= 0 && src_c < cols && src_r >= 0 && src_r < rows) {
            const int32_t src_idx   = src_r * cols + src_c;
            const float src_depth   = depths_ptr[src_offset + src_idx];
            const float3 src_normal = make_float3(
                normals_ptr[(src_offset + src_idx) * 3 + 0],
                normals_ptr[(src_offset + src_idx) * 3 + 1],
                normals_ptr[(src_offset + src_idx) * 3 + 2]);
            if (src_depth <= 0) continue;

            const float3 src_point =
                get_3D_point_on_world(src_c, src_r, src_depth, cameras_ptr[src_id]);
            const float2 tmp_pt = project_on_camera(src_point, cameras_ptr[ref_id], proj_depth);
            const float reproj_error = sqrt(pow(ref_c - tmp_pt.x, 2) + pow(ref_r - tmp_pt.y, 2));
            const float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
            const float angle               = get_angle(ref_normal, src_normal);

            if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                const_point.x += src_point.x;
                const_point.y += src_point.y;
                const_point.z += src_point.z;
                const_normal.x += src_normal.x;
                const_normal.y += src_normal.y;
                const_normal.z += src_normal.z;

                used_list[j].x = src_c;
                used_list[j].y = src_r;
                ++num_const;
            }
        }
    }
    __syncthreads();

    if (num_const >= geom_const) {
        const_point.x /= (num_const + 1.0f);
        const_point.y /= (num_const + 1.0f);
        const_point.z /= (num_const + 1.0f);
        const_normal.x /= (num_const + 1.0f);
        const_normal.y /= (num_const + 1.0f);
        const_normal.z /= (num_const + 1.0f);

        // get valid depth and normal
        float proj_depth;
        const float2 proj_point  = project_on_camera(const_point, cameras_ptr[ref_id], proj_depth);
        const int32_t proj_ref_r = int32_t(proj_point.y + 0.5f),
                      proj_ref_c = int32_t(proj_point.x + 0.5f);

        if (proj_ref_c >= 0 && proj_ref_c < cols && proj_ref_r >= 0 && proj_ref_r < rows &&
            proj_depth > 0.001f) {
            const int32_t proj_index            = proj_ref_r * cols + proj_ref_c;
            proj_depth_ptr[proj_index]          = proj_depth;
            proj_normal_ptr[proj_index * 3 + 0] = const_normal.x;
            proj_normal_ptr[proj_index * 3 + 1] = const_normal.y;
            proj_normal_ptr[proj_index * 3 + 2] = const_normal.z;
        }
        for (int j = 0; j < problems_ptr->num_ngb; ++j) {
            if (used_list[j].x == -1) continue;
            const int32_t offset = problems_ptr->src_image_ids[j] * rows * cols +
                                   used_list[j].y * cols + used_list[j].x;
            masks_ptr[offset] = true;
        }
    }
    __syncthreads();
}

std::tuple<vector<cv::Mat>, vector<cv::Mat>> PMMVS::run_fusion(
    const std::string& dense_folder,
    const vector<Problem>& problems,
    const bool geom_consistency,
    const int32_t geom_consistent) {
    size_t num_images        = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder   = dense_folder + std::string("/cams");

    vector<cv::Mat> images;
    vector<Camera> cameras;
    vector<cv::Mat_<float>> depths;
    vector<cv::Mat_<cv::Vec3f>> normals;
    vector<cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(4) << std::setfill('0')
                   << problems[i].ref_image_id << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(4) << std::setfill('0')
                 << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/ACMP/" << std::setw(4) << std::setfill('0')
                    << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix        = "/depths.dmb";
        if (geom_consistency) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path  = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);
        camera.height = depth.rows;
        camera.width  = depth.cols;

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }

    vector<PointList> PointCloud;
    PointCloud.clear();

    // output
    vector<cv::Mat> output_proj_depths;
    vector<cv::Mat> output_proj_normals;

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(4) << std::setfill('0') << i << "..."
                  << std::endl;
        const int32_t cols    = depths[i].cols;
        const int32_t rows    = depths[i].rows;
        const int32_t num_ngb = problems[i].num_ngb;
        vector<int2> used_list(num_ngb, make_int2(-1, -1));

        // output
        cv::Mat output_proj_depth  = cv::Mat::zeros(rows, cols, CV_32FC1);
        cv::Mat output_proj_normal = cv::Mat::zeros(rows, cols, CV_32FC3);

        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1) continue;
                float ref_depth      = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0) continue;

                float3 PointX               = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_point     = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_color[3]   = {
                      (float)images[i].at<cv::Vec3b>(r, c)[0],
                      (float)images[i].at<cv::Vec3b>(r, c)[1],
                      (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int32_t num_consistent = 0;

                for (int32_t j = 0; j < num_ngb; ++j) {
                    int32_t src_id         = problems[i].src_image_ids[j];
                    const int32_t src_cols = depths[src_id].cols;
                    const int32_t src_rows = depths[src_id].rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int32_t src_r = int32_t(point.y + 0.5f);
                    int32_t src_c = int32_t(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1) continue;

                        float src_depth      = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0) continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle               = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f &&
                            angle < 0.174533f) {
                            consistent_point.x += tmp_X.x;
                            consistent_point.y += tmp_X.y;
                            consistent_point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_color[0] += images[src_id].at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_color[1] += images[src_id].at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_color[2] += images[src_id].at<cv::Vec3b>(src_r, src_c)[2];

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= geom_consistent) {
                    consistent_point.x /= (num_consistent + 1.0f);
                    consistent_point.y /= (num_consistent + 1.0f);
                    consistent_point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_color[0] /= (num_consistent + 1.0f);
                    consistent_color[1] /= (num_consistent + 1.0f);
                    consistent_color[2] /= (num_consistent + 1.0f);

                    // get valid depth and normal
                    float2 proj_point;
                    float proj_depth;
                    ProjectonCamera(consistent_point, cameras[i], proj_point, proj_depth);
                    const int32_t proj_ref_r = int32_t(proj_point.y + 0.5f),
                                  proj_ref_c = int32_t(proj_point.x + 0.5f);

                    if (proj_ref_c >= 0 && proj_ref_c < cols && proj_ref_r >= 0 &&
                        proj_ref_r < rows && proj_depth > 0.001f) {
                        output_proj_depth.at<float>(proj_ref_r, proj_ref_c)      = proj_depth;
                        output_proj_normal.at<cv::Vec3f>(proj_ref_r, proj_ref_c) = cv::Vec3f(
                            consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    }

                    PointList point3D;
                    point3D.coord  = consistent_point;
                    point3D.normal = make_float3(
                        consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color =
                        make_float3(consistent_color[0], consistent_color[1], consistent_color[2]);
                    PointCloud.push_back(point3D);

                    for (int32_t j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1) continue;
                        masks[problems[i].src_image_ids[j]].at<uchar>(
                            used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }

        output_proj_depths.push_back(output_proj_depth);
        output_proj_normals.push_back(output_proj_normal);
    }

    std::string ply_path = dense_folder + "/ACMP/ACMP_model.ply";
    StoreColorPlyFileBinaryPointCloud(ply_path, PointCloud);

    return std::make_tuple(output_proj_depths, output_proj_normals);
}

}  // namespace mvs