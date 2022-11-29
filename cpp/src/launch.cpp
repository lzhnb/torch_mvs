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

std::tuple<vector<cv::Mat>, vector<cv::Mat>> run_fusion(
    const std::string &dense_folder,
    const vector<Problem> &problems,
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
        // std::cout << "Reading image " << std::setw(4) << std::setfill('0') << i << "..."
        //           << std::endl;
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
