#include "mvs.h"

namespace mvs {

vector<Problem> generate_sample_list(const std::string cluster_list_path) {
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
            if (score <= 0.0f) { continue; }
            problem.src_image_ids[j] = id;
            ++problem.num_ngb;
        }
        problems.push_back(problem);
    }

    return problems;
}

void PMMVS::load_samples(const std::string &dense_folder, const vector<Problem> problems) {
    all_images.clear();
    all_cameras.clear();

    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder   = dense_folder + std::string("/cams");

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

void PMMVS::load_depths(const std::string &dense_folder, const vector<Problem> problems) {
    all_depths.clear();

    const std::string suffix = params.multi_geometry ? "/depths_geom.dmb" : "/depths.dmb";
    size_t num_problems      = problems.size();
    for (size_t i = 0; i < num_problems; ++i) {
        std::stringstream result_path;
        result_path << dense_folder << "/ACMP/" << std::setw(4) << std::setfill('0')
                    << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string depth_path    = result_folder + suffix;
        cv::Mat_<float> depth;
        readDepthDmb(depth_path, depth);
        all_depths.push_back(depth);
    }
}

void process_problem(
    const std::string dense_folder,
    const Problem problem,
    const bool geom_consistency,
    const bool planar_prior,
    const bool multi_geometrty,
    PMMVS mvs) {
    // std::cout << "Processing image " << std::setw(4) << std::setfill('0') << problem.ref_image_id
    //           << "..." << std::endl;
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << dense_folder << "/ACMP/" << std::setw(4) << std::setfill('0')
                << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    // PMMVS mvs;
    if (geom_consistency) { mvs.SetGeomConsistencyParams(multi_geometrty); }
    mvs.InuputInitialization(dense_folder, problem);

    mvs.CudaSpaceInitialization(dense_folder, problem);
    mvs.RunPatchMatch();

    const int32_t width  = mvs.GetReferenceImageWidth();
    const int32_t height = mvs.GetReferenceImageHeight();

    cv::Mat_<float> depths      = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs       = cv::Mat::zeros(height, width, CV_32FC1);

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
        mvs.SetPlanarPriorParams();

        const cv::Rect imageRC(0, 0, width, height);
        vector<cv::Point> support2DPoints;

        mvs.GetSupportPoints(support2DPoints);
        const auto triangles = mvs.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage     = mvs.GetReferenceImage().clone();
        vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
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
        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, srcImage);

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
        // std::string depth_path = result_folder + "/depths_prior.dmb";
        //  writeDepthDmb(depth_path, priordepths);

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

    std::string suffix = "/depths.dmb";
    if (geom_consistency) { suffix = "/depths_geom.dmb"; }
    std::string depth_path  = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path   = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);

    mvs.release();
    // std::cout << "Processing image " << std::setw(4) << std::setfill('0') << problem.ref_image_id
    //           << " done!" << std::endl;
}

}  // namespace mvs
