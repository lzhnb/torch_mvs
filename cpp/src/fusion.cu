#include <cstdio>

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

__global__ void fusion_kernel(
    const float* __restrict__ depths_ptr,
    const float* __restrict__ normals_ptr,
    const Camera* __restrict__ cameras_ptr,
    const Problem* __restrict__ problem_ptr,
    const int32_t rows,
    const int32_t cols,
    const int32_t geom_const,
    // output
    uint8_t* __restrict__ masks_ptr,
    float* __restrict__ proj_depth_ptr,
    float* __restrict__ proj_normal_ptr) {
    const int32_t ref_c   = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t ref_r   = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t ref_idx = ref_r * cols + ref_c;

    if (ref_c >= cols || ref_r >= rows) return;

    const int32_t ref_id     = problem_ptr->ref_image_id;
    const int32_t ref_offset = ref_id * rows * cols;
    const float ref_depth    = depths_ptr[ref_offset + ref_idx];
    const float3 ref_normal  = make_float3(
        normals_ptr[(ref_offset + ref_idx) * 3 + 0],
        normals_ptr[(ref_offset + ref_idx) * 3 + 1],
        normals_ptr[(ref_offset + ref_idx) * 3 + 2]);

    if (ref_depth <= 0.0) return;

    int2 used_list[MAX_IMAGES];
    for (int32_t i = 0; i < problem_ptr->num_ngb; ++i) { used_list[i] = make_int2(-1, -1); }

    if (masks_ptr[ref_offset + ref_idx] == 1) return;

    const Camera ref_cam   = cameras_ptr[ref_id];
    const float3 ref_point = get_3D_point_on_world(ref_c, ref_r, ref_depth, ref_cam);
    float3 const_point     = ref_point;
    float3 const_normal    = ref_normal;
    int32_t num_const      = 0;

    for (int32_t j = 0; j < problem_ptr->num_ngb; ++j) {
        const int32_t src_id     = problem_ptr->src_image_ids[j];
        const Camera src_cam     = cameras_ptr[src_id];
        const int32_t src_offset = src_id * rows * cols;
        float proj_depth;
        const float2 point  = project_on_camera(ref_point, src_cam, proj_depth);
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

            const float3 src_point   = get_3D_point_on_world(src_c, src_r, src_depth, src_cam);
            const float2 tmp_pt      = project_on_camera(src_point, ref_cam, proj_depth);
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
        const float2 proj_point  = project_on_camera(const_point, ref_cam, proj_depth);
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
        for (int j = 0; j < problem_ptr->num_ngb; ++j) {
            if (used_list[j].x == -1) continue;
            const int32_t offset = problem_ptr->src_image_ids[j] * rows * cols +
                                   used_list[j].y * cols + used_list[j].x;
            masks_ptr[offset] = 1;
        }
    }
    __syncthreads();
}

std::tuple<vector<cv::Mat>, vector<cv::Mat>> run_fusion(
    const std::string& dense_folder,
    const vector<Problem>& problems,
    const bool geom_consistency,
    const int32_t geom_consistent) {
    size_t num_images      = problems.size();
    std::string cam_folder = dense_folder + std::string("/cams");

    vector<Camera> cameras;
    vector<cv::Mat_<float>> depths;
    vector<cv::Mat_<cv::Vec3f>> normals;
    vector<cv::Mat> masks;
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(4) << std::setfill('0')
                 << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/ACMP/" << std::setw(4) << std::setfill('0')
                    << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix        = "/depths.dmb";
        if (geom_consistency) { suffix = "/depths_geom.dmb"; }
        std::string depth_path  = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);
        camera.height = depth.rows;
        camera.width  = depth.cols;

        cv::Mat_<cv::Vec3b> scaled_image;
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }

    // collect all data
    cv::Mat all_depths, all_normals, all_masks;
    cv::vconcat(depths, all_depths);
    cv::vconcat(normals, all_normals);
    cv::vconcat(masks, all_masks);

    // put the cameras and problems on GPU
    Camera* cameras_ptr = nullptr;
    cudaMalloc((void**)&cameras_ptr, num_images * sizeof(Camera));
    cudaMemcpy(cameras_ptr, &cameras[0], num_images * sizeof(Camera), cudaMemcpyHostToDevice);
    Problem* problems_ptr = nullptr;
    cudaMalloc((void**)&problems_ptr, num_images * sizeof(Problem));
    cudaMemcpy(problems_ptr, &problems[0], num_images * sizeof(Problem), cudaMemcpyHostToDevice);

    // put the depths and normals on GPU
    float* depths_ptr;
    float* normals_ptr;
    uint8_t* masks_ptr;
    const int32_t rows = depths[0].rows, cols = depths[0].cols;
    cudaMalloc((void**)&depths_ptr, num_images * rows * cols * sizeof(float));
    cudaMemcpy(
        depths_ptr,
        all_depths.ptr<float>(),
        num_images * rows * cols * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMalloc((void**)&normals_ptr, num_images * rows * cols * sizeof(float) * 3);
    cudaMemcpy(
        normals_ptr,
        all_normals.ptr<cv::Vec3f>(),
        num_images * rows * cols * sizeof(float) * 3,
        cudaMemcpyHostToDevice);
    cudaMalloc((void**)&masks_ptr, num_images * rows * cols * sizeof(uint8_t));
    cudaMemcpy(
        masks_ptr,
        all_masks.ptr<uint8_t>(),
        num_images * rows * cols * sizeof(uint8_t),
        cudaMemcpyHostToDevice);

    // output
    vector<cv::Mat> output_proj_depths;
    vector<cv::Mat> output_proj_normals;

    for (size_t i = 0; i < num_images; ++i) {
        const int32_t num_ngb = problems[i].num_ngb;
        vector<int2> used_list(num_ngb, make_int2(-1, -1));

        // output
        cv::Mat output_proj_depth  = cv::Mat::zeros(rows, cols, CV_32FC1);
        cv::Mat output_proj_normal = cv::Mat::zeros(rows, cols, CV_32FC3);

        float* proj_depth_ptr;
        float* proj_normal_ptr;
        const int32_t rows = depths[0].rows, cols = depths[0].cols;
        cudaMalloc((void**)&proj_depth_ptr, rows * cols * sizeof(float));
        cudaMemcpy(
            proj_depth_ptr,
            output_proj_depth.ptr<float>(),
            rows * cols * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMalloc((void**)&proj_normal_ptr, rows * cols * sizeof(float) * 3);
        cudaMemcpy(
            proj_normal_ptr,
            output_proj_normal.ptr<cv::Vec3f>(),
            rows * cols * sizeof(float) * 3,
            cudaMemcpyHostToDevice);

        const int32_t kernel_size = 16;
        dim3 grid_size;
        grid_size.x = (cols + kernel_size - 1) / kernel_size;
        grid_size.y = (rows + kernel_size - 1) / kernel_size;
        grid_size.z = 1;
        dim3 block_size;
        block_size.x = kernel_size;
        block_size.y = kernel_size;
        block_size.z = 1;
        fusion_kernel<<<grid_size, block_size>>>(
            depths_ptr,
            normals_ptr,
            cameras_ptr,
            &problems_ptr[i],
            rows,
            cols,
            geom_consistent,
            masks_ptr,
            proj_depth_ptr,
            proj_normal_ptr);
        cudaMemcpy(
            output_proj_depth.ptr<float>(),
            proj_depth_ptr,
            rows * cols * sizeof(float),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            output_proj_normal.ptr<cv::Vec3f>(),
            proj_normal_ptr,
            rows * cols * sizeof(float) * 3,
            cudaMemcpyDeviceToHost);

        output_proj_depths.push_back(output_proj_depth);
        output_proj_normals.push_back(output_proj_normal);
        cudaFree(proj_depth_ptr);
        cudaFree(proj_normal_ptr);
    }

    cudaFree(cameras_ptr);
    cudaFree(problems_ptr);
    cudaFree(depths_ptr);
    cudaFree(normals_ptr);

    return std::make_tuple(output_proj_depths, output_proj_normals);
}

}  // namespace mvs