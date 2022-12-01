#include "mvs.h"

#include <cstdarg>

namespace mvs {

void StringAppendV(std::string *dst, const char *format, va_list ap) {
    // First try with a small fixed size buffer.
    static const int32_t kFixedBufferSize = 1024;
    char fixed_buffer[kFixedBufferSize];

    // It is possible for methods that use a va_list to invalidate
    // the data in it upon use.  The fix is to make a copy
    // of the structure before using it and use that copy instead.
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int32_t result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
    va_end(backup_ap);

    if (result < kFixedBufferSize) {
        if (result >= 0) {
            // Normal case - everything fits.
            dst->append(fixed_buffer, result);
            return;
        }

#ifdef _MSC_VER
        // Error or MSVC running out of space.  MSVC 8.0 and higher
        // can be asked about space needed with the special idiom below:
        va_copy(backup_ap, ap);
        result = vsnprintf(nullptr, 0, format, backup_ap);
        va_end(backup_ap);
#endif

        if (result < 0) {
            // Just an error.
            return;
        }
    }

    // Increase the buffer size to the size requested by vsnprintf,
    // plus one for the closing \0.
    const int32_t variable_buffer_size = result + 1;
    std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

    // Restore the va_list before we use it again.
    va_copy(backup_ap, ap);
    result = vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
    va_end(backup_ap);

    if (result >= 0 && result < variable_buffer_size) {
        dst->append(variable_buffer.get(), result);
    }
}

std::string StringPrintf(const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    std::string result;
    StringAppendV(&result, format, ap);
    va_end(ap);
    return result;
}

void CudaSafeCall(const cudaError_t error, const std::string &file, const int32_t line) {
    if (error != cudaSuccess) {
        std::cerr << StringPrintf(
                         "%s in %s at line %i", cudaGetErrorString(error), file.c_str(), line)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CudaCheckError(const char *file, const int32_t line) {
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << StringPrintf(
                         "cudaCheckError() failed at %s:%i : %s",
                         file,
                         line,
                         cudaGetErrorString(error))
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    error = cudaDeviceSynchronize();
    if (cudaSuccess != error) {
        std::cerr << StringPrintf(
                         "cudaCheckError() with sync failed at %s:%i : %s",
                         file,
                         line,
                         cudaGetErrorString(error))
                  << std::endl;
        std::cerr << "This error is likely caused by the graphics card timeout "
                     "detection mechanism of your operating system. Please refer to "
                     "the FAQ in the documentation on how to solve this problem."
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

PMMVS::PMMVS() {}

PMMVS::~PMMVS() {}

Camera ReadCamera(const std::string &cam_path) {
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
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

float3 Get3DPointonWorld(const int32_t x, const int32_t y, const float depth, const Camera camera) {
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

float3 Get3DPointonRefCam(
    const int32_t x, const int32_t y, const float depth, const Camera camera) {
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth) {
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth   = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2) {
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle       = acosf(dot_product);
    // if angle is not a number the dot product was 1 and thus the two vectors should be identical
    // --> return 0
    if (angle != angle) return 0.0f;

    return angle;
}

int32_t readDepthDmb(const std::string file_path, cv::Mat_<float> &depth) {
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type, sizeof(int32_t), 1, inimage);
    fread(&h, sizeof(int32_t), 1, inimage);
    fread(&w, sizeof(int32_t), 1, inimage);
    fread(&nb, sizeof(int32_t), 1, inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h * w * nb;

    float *data;
    data = (float *)malloc(sizeof(float) * dataSize);
    fread(data, sizeof(float), dataSize, inimage);

    depth = cv::Mat(h, w, CV_32F, data);

    fclose(inimage);
    return 0;
}

int32_t writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth) {
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h    = depth.rows;
    int32_t w    = depth.cols;
    int32_t nb   = 1;

    fwrite(&type, sizeof(int32_t), 1, outimage);
    fwrite(&h, sizeof(int32_t), 1, outimage);
    fwrite(&w, sizeof(int32_t), 1, outimage);
    fwrite(&nb, sizeof(int32_t), 1, outimage);

    float *data = (float *)depth.data;

    int32_t datasize = w * h * nb;
    fwrite(data, sizeof(float), datasize, outimage);

    fclose(outimage);
    return 0;
}

int32_t readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal) {
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type, sizeof(int32_t), 1, inimage);
    fread(&h, sizeof(int32_t), 1, inimage);
    fread(&w, sizeof(int32_t), 1, inimage);
    fread(&nb, sizeof(int32_t), 1, inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h * w * nb;

    float *data;
    data = (float *)malloc(sizeof(float) * dataSize);
    fread(data, sizeof(float), dataSize, inimage);

    normal = cv::Mat(h, w, CV_32FC3, data);

    fclose(inimage);
    return 0;
}

int32_t writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal) {
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h    = normal.rows;
    int32_t w    = normal.cols;
    int32_t nb   = 3;

    fwrite(&type, sizeof(int32_t), 1, outimage);
    fwrite(&h, sizeof(int32_t), 1, outimage);
    fwrite(&w, sizeof(int32_t), 1, outimage);
    fwrite(&nb, sizeof(int32_t), 1, outimage);

    float *data = (float *)normal.data;

    int32_t datasize = w * h * nb;
    fwrite(data, sizeof(float), datasize, outimage);

    fclose(outimage);
    return 0;
}

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

void PMMVS::InuputInitialization(const std::string &dense_folder, const Problem &problem) {
    images.clear();
    cameras.clear();

    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder   = dense_folder + std::string("/cams");

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

void PMMVS::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem) {
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

        std::stringstream result_path;
        result_path << dense_folder << "/ACMP/" << std::setw(4) << std::setfill('0')
                    << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix        = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string cost_path          = result_folder + "/costs.dmb";
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

void PMMVS::CudaPlanarPriorInitialization(
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

float4 PMMVS::GetPlaneHypothesis(const int32_t index) { return plane_hypotheses_host[index]; }

float PMMVS::GetCost(const int32_t index) { return costs_host[index]; }

void PMMVS::GetSupportPoints(vector<cv::Point> &support2DPoints) {
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
                    if (GetCost(center) < 2.0f && min_cost > GetCost(center)) {
                        temp_point = cv::Point(c, r);
                        min_cost   = GetCost(center);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

vector<Triangle> PMMVS::DelaunayTriangulation(
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

float4 PMMVS::GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths) {
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);

    float3 ptX1 = Get3DPointonRefCam(
        triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0]);
    float3 ptX2 = Get3DPointonRefCam(
        triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0]);
    float3 ptX3 = Get3DPointonRefCam(
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

float PMMVS::GetDepthFromPlaneParam(
    const float4 plane_hypothesis, const int32_t x, const int32_t y) {
    return -plane_hypothesis.w * cameras[0].K[0] /
           ((x - cameras[0].K[2]) * plane_hypothesis.x +
            (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5]) * plane_hypothesis.y +
            cameras[0].K[0] * plane_hypothesis.z);
}

}  // namespace mvs
