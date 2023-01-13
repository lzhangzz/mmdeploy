// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_tracker/pose_tracker.h"

#include <cmath>
#include <numeric>

#include "mmdeploy/core/utils/formatter.h"
#include "pose_tracker/utils.h"

namespace mmdeploy::mmpose::_pose_tracker {

Tracker::Tracker(const mmdeploy_pose_tracker_param_t& _params) : params_(_params) {
  if (params_.keypoint_sigmas && params_.keypoint_sigmas_size) {
    std::copy_n(params_.keypoint_sigmas, params_.keypoint_sigmas_size,
                std::back_inserter(key_point_sigmas_));
    params_.keypoint_sigmas = key_point_sigmas_.data();
  }
}

void Tracker::SuppressOverlappingBoxes(const vector<Bbox>& bboxes,
                                       vector<std::pair<int, float>>& ranks,
                                       vector<int>& is_valid_bbox) const {
  vector<float> iou(ranks.size() * ranks.size());
  for (int i = 0; i < bboxes.size(); ++i) {
    for (int j = 0; j < i; ++j) {
      iou[i * bboxes.size() + j] = iou[j * bboxes.size() + i] =
          intersection_over_union(bboxes[i], bboxes[j]);
    }
  }
  suppress_non_maximum(ranks, iou, is_valid_bbox, params_.det_nms_thr);
}

void Tracker::SuppressOverlappingPoses(const vector<Points>& keypoints,
                                       const vector<Scores>& scores, const vector<Bbox>& bboxes,
                                       const vector<int64_t>& track_ids, vector<int>& is_valid,
                                       float oks_thr) {
  assert(keypoints.size() == is_valid.size());
  assert(scores.size() == is_valid.size());
  assert(bboxes.size() == is_valid.size());
  const auto size = is_valid.size();
  vector<float> similarity(size * size);
  for (int i = 0; i < size; ++i) {
    if (is_valid[i]) {
      for (int j = 0; j < i; ++j) {
        if (is_valid[j]) {
          similarity[i * size + j] = similarity[j * size + i] =
              GetPosePoseSimilarity(bboxes[i], keypoints[i], bboxes[j], keypoints[j]);
        }
      }
    }
  }
  vector<std::pair<bool, float>> ranks;
  ranks.reserve(size);
  for (int i = 0; i < size; ++i) {
    bool is_visible = false;
    for (const auto& track : tracks_) {
      if (track->track_id() == track_ids[i]) {
        is_visible = track->missing() == 0;
        break;
      }
    }
    auto score = std::accumulate(scores[i].begin(), scores[i].end(), 0.f);
    // prevents bboxes from missing tracks to suppress visible tracks
    ranks.emplace_back(is_visible, score);
  }
  suppress_non_maximum(ranks, similarity, is_valid, oks_thr);
}

std::tuple<vector<Bbox>, vector<int64_t>> Tracker::ProcessBboxes(
    const std::optional<Detections>& dets) {
  vector<Bbox> bboxes;
  vector<int64_t> prev_ids;

  // 2 - visible tracks
  // 1 - detection
  // 0 - missing tracks
  vector<int> types;

  GetDetectedObjects(dets, bboxes, prev_ids, types);

  GetTrackedObjects(bboxes, prev_ids, types);

  vector<int> is_valid_bboxes(bboxes.size(), 1);

  auto count = [&] {
    std::array<int, 3> acc{};
    for (size_t i = 0; i < is_valid_bboxes.size(); ++i) {
      if (is_valid_bboxes[i]) {
        ++acc[types[i]];
      }
    }
    return acc;
  };
  POSE_TRACKER_DEBUG("frame {}, bboxes {}", frame_id_, count());

  vector<std::pair<int, float>> ranks;
  ranks.reserve(bboxes.size());
  for (int i = 0; i < bboxes.size(); ++i) {
    ranks.emplace_back(types[i], get_area(bboxes[i]));
  }
  SuppressOverlappingBoxes(bboxes, ranks, is_valid_bboxes);
  POSE_TRACKER_DEBUG("frame {}, bboxes after nms: {}", frame_id_, count());

  vector<int> idxs;
  idxs.reserve(bboxes.size());
  for (int i = 0; i < bboxes.size(); ++i) {
    if (is_valid_bboxes[i]) {
      idxs.push_back(i);
    }
  }

  std::stable_sort(idxs.begin(), idxs.end(), [&](int i, int j) { return ranks[i] > ranks[j]; });
  std::fill(is_valid_bboxes.begin(), is_valid_bboxes.end(), 0);
  {
    vector<Bbox> tmp_bboxes;
    vector<int64_t> tmp_track_ids;
    for (const auto& i : idxs) {
      if (tmp_bboxes.size() >= params_.pose_max_num_bboxes) {
        break;
      }
      tmp_bboxes.push_back(bboxes[i]);
      tmp_track_ids.push_back(prev_ids[i]);
      is_valid_bboxes[i] = 1;
    }
    bboxes = std::move(tmp_bboxes);
    prev_ids = std::move(tmp_track_ids);
  }

  POSE_TRACKER_DEBUG("frame {}, bboxes after sort: {}", frame_id_, count());
  return {bboxes, prev_ids};
}

void Tracker::TrackStep(vector<Points>& keypoints, vector<Scores>& scores,
                        const vector<int64_t>& prev_ids) noexcept {
  vector<Bbox> bboxes(keypoints.size());
  vector<int> is_unused_bbox(keypoints.size(), 1);

  // key-points to bboxes
  for (size_t i = 0; i < keypoints.size(); ++i) {
    if (auto bbox = KeypointsToBbox(keypoints[i], scores[i])) {
      bboxes[i] = *bbox;
    } else {
      is_unused_bbox[i] = false;
    }
  }

  SuppressOverlappingPoses(keypoints, scores, bboxes, prev_ids, is_unused_bbox,
                           params_.pose_nms_thr);
  assert(is_unused_bbox.size() == bboxes.size());

  vector<float> similarity0;
  vector<float> similarity1;
  GetSimilarityMatrices(bboxes, keypoints, prev_ids, similarity0, similarity1);

  vector<int> is_unused_track(tracks_.size(), 1);
  // disable missing tracks in the #1 assignment
  for (int i = 0; i < tracks_.size(); ++i) {
    if (tracks_[i]->missing()) {
      is_unused_track[i] = 0;
    }
  }
  const auto assignment_visible =
      greedy_assignment(similarity0, is_unused_bbox, is_unused_track, params_.track_visible_thr);
  POSE_TRACKER_DEBUG("frame {}, assignment for visible {}", frame_id_, assignment_visible);

  // enable missing tracks in the #2 assignment
  for (int i = 0; i < tracks_.size(); ++i) {
    if (tracks_[i]->missing()) {
      is_unused_track[i] = 1;
    }
  }
  const auto assignment_missing =
      greedy_assignment(similarity1, is_unused_bbox, is_unused_track, params_.track_missing_thr);
  POSE_TRACKER_DEBUG("frame {}, assignment for missing {}", frame_id_, assignment_missing);

  // merge assignments
  auto assignment = assignment_visible;
  assignment.insert(assignment.end(), assignment_missing.begin(), assignment_missing.end());

  // update assigned tracks
  for (auto [i, j, _] : assignment) {
    tracks_[j]->UpdateVisible(bboxes[i], keypoints[i], scores[i]);
  }

  // generating new tracks from detected bboxes
  for (size_t i = 0; i < is_unused_bbox.size(); ++i) {
    if (is_unused_bbox[i] && prev_ids[i] == -1) {
      CreateTrack(bboxes[i], keypoints[i], scores[i]);
    }
  }

  // update missing tracks
  for (size_t i = 0; i < is_unused_track.size(); ++i) {
    if (is_unused_track[i]) {
      tracks_[i]->UpdateMissing();
    }
  }

  // diagnostic for missing tracks
  //  DiagnosticMissingTracks(is_valid_tracks, is_valid_bboxes, similarity0, similarity1);

  RemoveMissingTracks();

  for (auto& track : tracks_) {
    track->Predict();
  }

  ++frame_id_;

  // print track summary
  //  SummaryTracks();
}

void Tracker::GetTrackedObjects(vector<Bbox>& bboxes, vector<int64_t>& track_ids,
                                vector<int>& types) const {
  for (auto& track : tracks_) {
    std::optional<Bbox> bbox;
    if (track->missing()) {
      bbox = track->predicted_bbox();
    } else {
      bbox = keypoints_to_bbox(track->predicted_kpts(), track->scores(), frame_h_, frame_w_,
                               params_.track_bbox_scale, params_.pose_kpt_thr,
                               params_.pose_min_keypoints);
    }
    if (bbox && get_area(*bbox) >= params_.pose_min_bbox_size * params_.pose_min_bbox_size) {
      bboxes.push_back(*bbox);
      track_ids.push_back(track->track_id());
      types.push_back(track->missing() ? 0 : 2);
    }
  }
}

void Tracker::GetDetectedObjects(const std::optional<Detections>& dets, vector<Bbox>& _bboxes,
                                 vector<int64_t>& track_ids, vector<int>& types) const {
  if (dets) {
    auto& [bboxes, scores, labels] = *dets;
    for (size_t i = 0; i < bboxes.size(); ++i) {
      if (labels[i] == params_.det_label && scores[i] > params_.det_thr &&
          get_area(bboxes[i]) >= params_.det_min_bbox_size * params_.det_min_bbox_size) {
        _bboxes.push_back(bboxes[i]);
        track_ids.push_back(-1);
        types.push_back(1);
      }
    }
  }
}

std::pair<float, float> Tracker::GetTrackPoseSimilarity(const Bbox& track_bbox,
                                                        const Points& track_kpts, const Bbox& bbox,
                                                        const Points& kpts) const {
  auto iou = intersection_over_union(track_bbox, bbox);
  if (key_point_sigmas_.empty()) {
    return {iou, iou};
  }
  // asymmetric
  auto oks =
      object_keypoint_similarity(track_kpts, track_bbox, kpts, track_bbox, key_point_sigmas_);
  return {oks, iou};
}

void Tracker::GetSimilarityMatrices(const vector<Bbox>& bboxes, const vector<Points>& keypoints,
                                    const vector<int64_t>& prev_ids, vector<float>& similarity0,
                                    vector<float>& similarity1) {
  const auto n_rows = static_cast<int>(bboxes.size());
  const auto n_cols = static_cast<int>(tracks_.size());

  // generate similarity matrix
  similarity0.resize(n_rows * n_cols);
  similarity1.resize(n_rows * n_cols);
  for (size_t i = 0; i < n_rows; ++i) {
    const auto& bbox = bboxes[i];
    const auto& kpts = keypoints[i];
    for (size_t j = 0; j < n_cols; ++j) {
      const auto& track = tracks_[j];
      if (prev_ids[i] != -1 && prev_ids[i] != track->track_id()) {
        continue;
      }
      const auto index = i * n_cols + j;
      std::tie(similarity0[index], similarity1[index]) =
          GetTrackPoseSimilarity(track->predicted_bbox(), track->predicted_kpts(), bbox, kpts);
      track->Distance(bbox);
    }
  }
}

float Tracker::GetPosePoseSimilarity(const Bbox& bbox0, const Points& kpts0, const Bbox& bbox1,
                                     const Points& kpts1) {
  if (key_point_sigmas_.empty()) {
    return intersection_over_union(bbox0, bbox1);
  }
  // symmetric
  return object_keypoint_similarity(kpts0, bbox0, kpts1, bbox1, key_point_sigmas_);
}

void Tracker::CreateTrack(const Bbox& bbox, const Points& kpts, const Scores& scores) {
  *tracks_.emplace_back(std::make_unique<Track>(&params_, bbox, kpts, scores, next_id_++));
}

std::optional<Bbox> Tracker::KeypointsToBbox(const Points& kpts, const Scores& scores) const {
  return keypoints_to_bbox(kpts, scores, frame_h_, frame_w_, params_.track_bbox_scale,
                           params_.pose_kpt_thr, params_.pose_min_keypoints);
}

void Tracker::RemoveMissingTracks() {
  size_t count{};
  for (auto& track : tracks_) {
    if (track->missing() <= params_.track_max_missing) {
      tracks_[count++] = std::move(track);
    }
  }
  tracks_.resize(count);
}

void Tracker::DiagnosticMissingTracks(const vector<int>& is_unused_track,
                                      const vector<int>& is_unused_bbox,
                                      const vector<float>& similarity0,
                                      const vector<float>& similarity1) {
  int missing = 0;
  const auto n_cols = static_cast<int>(is_unused_track.size());
  const auto n_rows = static_cast<int>(is_unused_bbox.size());
  for (int i = 0; i < is_unused_track.size(); ++i) {
    if (is_unused_track[i]) {
      float best_oks = 0.f;
      float best_iou = 0.f;
      for (int j = 0; j < is_unused_bbox.size(); ++j) {
        if (is_unused_bbox[j]) {
          best_oks = std::max(similarity0[j * n_cols + i], best_oks);
          best_iou = std::max(similarity1[j * n_cols + i], best_iou);
        }
      }
      POSE_TRACKER_DEBUG("frame {}: track missing {}, best_oks={}, best_iou={}", frame_id_,
                         tracks_[i]->track_id(), best_oks, best_iou);
      ++missing;
    }
  }
  if (missing) {
    std::stringstream ss;
    ss << cv::Mat_<float>(n_rows, n_cols, const_cast<float*>(similarity0.data()));
    POSE_TRACKER_DEBUG("frame {}, similarity#0: \n{}", frame_id_, ss.str());
    ss = {};
    ss << cv::Mat_<float>(n_rows, n_cols, const_cast<float*>(similarity1.data()));
    POSE_TRACKER_DEBUG("frame {}, similarity#1: \n{}", frame_id_, ss.str());
  }
}

void Tracker::SummaryTracks() {
  vector<std::tuple<int64_t, int>> summary;
  for (const auto& track : tracks_) {
    summary.emplace_back(track->track_id(), track->missing());
  }
  POSE_TRACKER_DEBUG("frame {}, track summary {}", frame_id_, summary);
  for (const auto& track : tracks_) {
    if (!track->missing()) {
      POSE_TRACKER_DEBUG("frame {}, track {}, scores {}", frame_id_, track->track_id(),
                         track->scores());
    }
  }
}

}  // namespace mmdeploy::mmpose::_pose_tracker
