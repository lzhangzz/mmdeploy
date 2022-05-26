// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_

#include <cstdint>

#include "cuda_runtime.h"

namespace mmdeploy::mmocr {

namespace panet {

void SigmoidAndThreshold(const float* d_logit, int n, float thr, uint8_t* d_mask, float* d_score,
                         cudaStream_t stream);

void Transpose(const float* d_input, int h, int w, float* d_output, cudaStream_t stream);

}

namespace dbnet {

void Threshold(const float* d_score, int n, float thr, uint8_t* d_mask, cudaStream_t stream);

}

}  // namespace mmdeploy::mmocr

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_
