// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_DETECTOR_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_DETECTOR_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/text_detector.h"

namespace mmdeploy {

using TextDetection = mmdeploy_text_detection_t;

class TextDetector : public NonMovable {
 public:
  TextDetector(const Model& model, const Device& device) {
    auto ec = mmdeploy_text_detector_create(model, device.name(), device.index(), &detector_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~TextDetector() {
    if (detector_) {
      mmdeploy_text_detector_destroy(detector_);
      detector_ = {};
    }
  }

  class Result {
   public:
    Result(size_t offset, size_t size, std::shared_ptr<TextDetection> data)
        : offset_(offset), size_(size), data_(std::move(data)) {}

    TextDetection* operator[](int index) const { return data_.get() + offset_ + index; }

    size_t size() const noexcept { return size_; }

   private:
    size_t offset_{};
    size_t size_{};
    std::shared_ptr<TextDetection> data_;
  };

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }
    auto mats = GetMats(images);

    TextDetection* results{};
    int* result_count{};
    auto ec = mmdeploy_text_detector_apply(detector_, mats.data(), static_cast<int>(mats.size()),
                                           &results, &result_count);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::shared_ptr<TextDetection> data(results, [result_count, count = mats.size()](auto p) {
      mmdeploy_text_detector_release_result(p, result_count, count);
    });

    std::vector<Result> rets;
    rets.reserve(images.size());

    size_t offset = 0;
    for (size_t i = 0; i < mats.size(); ++i) {
      offset += rets.emplace_back(offset, result_count[i], data).size();
    }

    return rets;
  }

 private:
  mmdeploy_text_detector_t detector_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_DETECTOR_HPP_
