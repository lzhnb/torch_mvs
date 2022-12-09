#pragma once

#include "primitives.h"

namespace mvs {

cv::Mat fusion_textureless_mask(
    const cv::Mat& seg_ids,
    const cv::Mat& planar_mask,
    const cv::Mat& filter_normals,
    const int32_t rows,
    const int32_t cols,
    const int32_t thresh,
    const float nonplanar_percent,
    const float cos_sim_thresh,
    const float match_ratio_thresh);

cv::Mat filter_by_var_map(
    const cv::Mat& var_map, const cv::Mat& segment_ids_map, const float var_thresh);
}  // namespace mvs
