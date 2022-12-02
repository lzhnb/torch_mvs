#include "mvs.h"

namespace mvs {

vector<Problem> generate_sample_list(const string cluster_list_path) {
    vector<Problem> problems;
    problems.clear();

    std::ifstream file(cluster_list_path);

    int32_t num_images;
    file >> num_images;

    for (int32_t i = 0; i < num_images; ++i) {
        Problem problem;
        file >> problem.ref_image_id;

        int32_t num_src_images;
        file >> num_src_images;
        problem.num_ngb = 0;
        for (int32_t j = 0; j < num_src_images; ++j) {
            int32_t id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids[j] = id;
            ++problem.num_ngb;
        }
        problems.push_back(problem);
    }

    return problems;
}

void PMMVS::load_samples(const string &dense_folder, const vector<Problem> problems) {
    all_images.clear();
    all_cameras.clear();

    string image_folder = dense_folder + string("/images");
    string cam_folder   = dense_folder + string("/cams");

    const int32_t num_images = problems.size();
    for (size_t i = 0; i < num_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(4) << std::setfill('0')
                   << problems[i].ref_image_id << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        all_images.push_back(image_float);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(4) << std::setfill('0')
                 << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());
        camera.height = image_float.rows;
        camera.width  = image_float.cols;
        all_cameras.push_back(camera);
    }
}

void PMMVS::load_geometry(
    const vector<cv::Mat> &depth_maps,
    const vector<cv::Mat> &normal_maps,
    const vector<cv::Mat> &cost_maps) {
    all_depths.clear();
    all_normals.clear();
    all_costs.clear();

    size_t num_images = depth_maps.size();
    for (size_t i = 0; i < num_images; ++i) {
        all_depths.push_back(depth_maps[i]);
        all_normals.push_back(normal_maps[i]);
        all_costs.push_back(cost_maps[i]);
    }
}

tuple<cv::Mat, cv::Mat, cv::Mat> process_problem(
    const string dense_folder,
    const Problem problem,
    const bool geom_consistency,
    const bool planar_prior,
    const bool multi_geometrty,
    PMMVS mvs) {
    cudaSetDevice(0);
    mvs.InuputInitialization(problem);
    mvs.CudaSpaceInitialization(problem);
    mvs.RunPatchMatch();

    const int32_t width  = mvs.cameras[0].width;
    const int32_t height = mvs.cameras[0].height;

    cv::Mat_<float> depths      = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs       = cv::Mat::zeros(height, width, CV_32FC1);

    // TODO: put the export on CUDA
    for (int32_t col = 0; col < width; ++col) {
        for (int32_t row = 0; row < height; ++row) {
            int32_t center          = row * width + col;
            float4 plane_hypothesis = mvs.GetPlaneHypothesis(center);
            depths(row, col)        = plane_hypothesis.w;
            normals(row, col) =
                cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = mvs.GetCost(center);
        }
    }

    if (planar_prior) {
        // std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        mvs.params.planar_prior = true;

        const cv::Rect imageRC(0, 0, width, height);
        vector<cv::Point> support2DPoints;

        mvs.GetSupportPoints(support2DPoints);
        const auto triangles = mvs.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat ref_image    = mvs.images[0].clone();
        vector<cv::Mat> mbgr(3);
        mbgr[0] = ref_image.clone();
        mbgr[1] = ref_image.clone();
        mbgr[2] = ref_image.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) &&
                imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        // string triangulation_path = result_folder + "/triangulation.png";
        // cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) &&
                imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(
                    pow(triangle.pt1.x - triangle.pt2.x, 2) +
                    pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(
                    pow(triangle.pt1.x - triangle.pt3.x, 2) +
                    pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(
                    pow(triangle.pt2.x - triangle.pt3.x, 2) +
                    pow(triangle.pt2.y - triangle.pt3.y, 2));

                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step            = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int32_t x = p * triangle.pt1.x + q * triangle.pt2.x +
                                    (1.0 - p - q) * triangle.pt3.x;
                        int32_t y = p * triangle.pt1.y + q * triangle.pt2.y +
                                    (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) =
                            idx + 1.0;  // To distinguish from the label of non-triangulated areas
                    }
                }

                // estimate plane parameter
                float4 n4 = mvs.GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int32_t i = 0; i < width; ++i) {
            for (int32_t j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = mvs.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= mvs.params.depth_max && d >= mvs.params.depth_min) {
                        priordepths(j, i) = d;
                    } else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }

        mvs.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        mvs.RunPatchMatch();

        for (int32_t col = 0; col < width; ++col) {
            for (int32_t row = 0; row < height; ++row) {
                int32_t center          = row * width + col;
                float4 plane_hypothesis = mvs.GetPlaneHypothesis(center);
                depths(row, col)        = plane_hypothesis.w;
                normals(row, col) =
                    cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = mvs.GetCost(center);
            }
        }
    }

    mvs.release();
    return std::make_tuple(depths, normals, costs);
}

}  // namespace mvs
