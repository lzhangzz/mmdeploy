// Copyright (c) OpenMMLab. All rights reserved.

#include "classifier.h"

#include <numeric>

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/handle.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmcls/mmcls.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace mmdeploy;
using namespace std;

namespace {

Value& config_template() {
  // clang-format off
  static Value v{
    {
      "pipeline", {
        {"input", {"img"}},
        {"output", {"cls"}},
        {
          "tasks", {
            {
              {"name", "classifier"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"cls"}}
            }
          }
        }
      }
    }
  };
  // clang-format on
  return v;
}

int mmdeploy_classifier_create_impl(mmdeploy_model_t model, const char* device_name, int device_id,
                                    mmdeploy_exec_info_t exec_info,
                                    mmdeploy_classifier_t* classifier) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *Cast(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info,
                                  (mmdeploy_pipeline_t*)classifier);
}

}  // namespace

int mmdeploy_classifier_create(mmdeploy_model_t model, const char* device_name, int device_id,
                               mmdeploy_classifier_t* classifier) {
  return mmdeploy_classifier_create_impl(model, device_name, device_id, nullptr, classifier);
}

int mmdeploy_classifier_create_v2(mmdeploy_model_t model, const char* device_name, int device_id,
                                  mmdeploy_exec_info_t exec_info,
                                  mmdeploy_classifier_t* classifier) {
  return mmdeploy_classifier_create_impl(model, device_name, device_id, exec_info, classifier);
}

int mmdeploy_classifier_create_by_path(const char* model_path, const char* device_name,
                                       int device_id, mmdeploy_classifier_t* classifier) {
  mmdeploy_model_t model{};

  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_classifier_create_impl(model, device_name, device_id, nullptr, classifier);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_classifier_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                     mmdeploy_value_t* value) {
  return mmdeploy_common_create_input(mats, mat_count, value);
}

int mmdeploy_classifier_apply(mmdeploy_classifier_t classifier, const mmdeploy_mat_t* mats,
                              int mat_count, mmdeploy_classification_t** results, int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_classifier_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_classifier_apply_v2(classifier, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_classifier_get_result(output, results, result_count)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_classifier_apply_v2(mmdeploy_classifier_t classifier, mmdeploy_value_t input,
                                 mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)classifier, input, output);
}

int mmdeploy_classifier_apply_async(mmdeploy_classifier_t classifier, mmdeploy_sender_t input,
                                    mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)classifier, input, output);
}

int mmdeploy_classifier_get_result(mmdeploy_value_t output, mmdeploy_classification_t** results,
                                   int** result_count) {
  if (!output || !results || !result_count) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    Value& value = Cast(output)->front();

    auto classify_outputs = from_value<vector<mmcls::ClassifyOutput>>(value);

    vector<int> _result_count;
    _result_count.reserve(classify_outputs.size());

    for (const auto& cls_output : classify_outputs) {
      _result_count.push_back((int)cls_output.labels.size());
    }

    auto total = std::accumulate(begin(_result_count), end(_result_count), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mmdeploy_classification_t[]> result_data(new mmdeploy_classification_t[total]{});
    auto result_ptr = result_data.get();
    for (const auto& cls_output : classify_outputs) {
      for (const auto& label : cls_output.labels) {
        result_ptr->label_id = label.label_id;
        result_ptr->score = label.score;
        ++result_ptr;
      }
    }

    *result_count = result_count_data.release();
    *results = result_data.release();

    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_classifier_release_result(mmdeploy_classification_t* results, const int* result_count, int count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_classifier_destroy(mmdeploy_classifier_t classifier) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)classifier);
}
