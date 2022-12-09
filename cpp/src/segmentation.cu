#include "segmentation.h"

namespace mvs {
cv::Mat fusion_textureless_mask(
    const cv::Mat& seg_ids,
    const cv::Mat& textureless_mask,
    const cv::Mat& filter_normals,
    const int32_t rows,
    const int32_t cols,
    const int32_t thresh                = 100,
    const float non_textureless_percent = 0.75,
    const float cos_sim_thresh          = 0.8f,
    const float match_ratio_thresh      = 0.9f) {
    double min_val, max_val;
    cv::minMaxLoc(seg_ids, &min_val, &max_val);

    vector<vector<cv::Vec3f>> seg_normals;
    vector<vector<cv::Point2i>> seg_ids_map;
    vector<cv::Point2i> seg_ids_textureless_count;
    seg_normals.resize(static_cast<uint32_t>(max_val) + 1);
    seg_ids_map.resize(static_cast<uint32_t>(max_val) + 1);
    seg_ids_textureless_count.resize(static_cast<uint32_t>(max_val) + 1);

    // fullfill the seg_ids_map
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const cv::Point2i p(col, row);
            const int32_t seg_id = seg_ids.at<int32_t>(p);
            seg_ids_map[seg_id].push_back(p);
            if (textureless_mask.at<int32_t>(p) == 0) {  // non-textureless region
                ++seg_ids_textureless_count[seg_id].x;
                // statistic normals
                seg_normals[seg_id].push_back(filter_normals.at<cv::Vec3f>(p));
            } else {  // textureless region
                ++seg_ids_textureless_count[seg_id].y;
            }
        }
    }

    // too textureless region, where mvs could not handle, segid begin from 301
    uint32_t area_seg_id = 301;
    cv::Mat fusion_mask  = cv::Mat::zeros(rows, cols, CV_32SC1);
    for (int32_t seg_id = 1; seg_id < static_cast<uint32_t>(max_val) + 1; ++seg_id) {
        const float non_textureless = static_cast<float>(seg_ids_textureless_count[seg_id].x);
        const float textureless     = static_cast<float>(seg_ids_textureless_count[seg_id].y);
        const float amount          = non_textureless + textureless;
        // filter out the small, useless(too many non_textureless) superpixels
        if (amount <= thresh) {
            continue;
        }
        bool skip = false;
        // filter out the truly non_textureless superpixels
        if ((non_textureless / amount) >= non_textureless_percent) {
            skip = true;
            // calculate the average normal
            cv::Vec3f avg_n(0.0f, 0.0f, 0.0f);
            for (cv::Vec3f n : seg_normals[seg_id]) {
                avg_n += n;
            }
            avg_n /= non_textureless;

            // count the number of the normals that match the average normal
            int32_t valid = 0;
            for (cv::Vec3f n : seg_normals[seg_id]) {
                const float cos_sim = avg_n.dot(n);
                valid += (cos_sim > cos_sim_thresh ? 1 : 0);
            }
            // define the superpixel as textureless superpixel if there are enough
            // normals matching the average normal, and vice versa
            const float ratio = static_cast<float>(valid) / non_textureless;
            if (ratio > match_ratio_thresh) {  // too many textureless area
                skip = false;
            } else {  // really object
                for (cv::Point2i p : seg_ids_map[seg_id]) {
                    fusion_mask.at<int32_t>(p) = 300;
                }
            }
        } else if ((non_textureless / amount) < 0.25) {
            // too textureless region, where mvs could not hanle keep this region
            for (cv::Point2i p : seg_ids_map[seg_id]) {
                fusion_mask.at<int32_t>(p) = area_seg_id;
            }
            area_seg_id += 1;
            skip = true;
        } else {
            skip = false;
        }
        // skip the superpixel whose normals are not quite consistent(non textureless)
        if (skip) {
            continue;
        }
        for (cv::Point2i p : seg_ids_map[seg_id]) {
            fusion_mask.at<int32_t>(p) = seg_id;
        }
    }

    return fusion_mask;
}

cv::Mat filter_by_var_map(
    const cv::Mat& var_map, const cv::Mat& segment_ids_map, const float var_thresh) {
    const int32_t rows = var_map.rows, cols = var_map.cols;
    cv::Mat output_segment_ids_map = cv::Mat::zeros(rows, cols, CV_32SC1);

    double min_val, max_val;
    cv::minMaxLoc(segment_ids_map, &min_val, &max_val);
    vector<float> var_accum;
    vector<float> count;
    vector<bool> filter_mask;
    const int32_t max_seg_id = static_cast<int32_t>(max_val);
    var_accum.resize(max_seg_id + 1);
    count.resize(max_seg_id + 1);
    filter_mask.resize(max_seg_id + 1);

    for (int32_t seg_id = 1; seg_id < max_seg_id; ++seg_id) {
        var_accum[seg_id]   = 0;
        count[seg_id]       = 0;
        filter_mask[seg_id] = false;
    }
    // statistic
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int32_t seg_id = segment_ids_map.at<int32_t>(row, col);
            if (seg_id == 0) continue;
            ++count[seg_id];
            var_accum[seg_id] += var_map.at<float>(row, col);
        }
    }

    // calculate the average var and fileter
    for (int32_t seg_id = 1; seg_id < max_seg_id + 1; ++seg_id) {
        const float mean_var = var_accum[seg_id] / static_cast<float>(count[seg_id]);
        filter_mask[seg_id]  = mean_var <= var_thresh;
    }

    // filter
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int32_t seg_id = segment_ids_map.at<int32_t>(row, col);
            if (filter_mask[seg_id]) {
                output_segment_ids_map.at<int32_t>(row, col) = seg_id;
            }
        }
    }

    return output_segment_ids_map;
}
}  // namespace mvs
