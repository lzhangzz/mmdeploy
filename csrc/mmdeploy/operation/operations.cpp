//
// Created by zhangli on 11/3/22.
//
#include "mmdeploy/operation/operations.h"

namespace mmdeploy::operation {

MMDEPLOY_DEFINE_REGISTRY(CvtColor);
MMDEPLOY_DEFINE_REGISTRY(Resize);
MMDEPLOY_DEFINE_REGISTRY(Pad);
MMDEPLOY_DEFINE_REGISTRY(ToFloat);
MMDEPLOY_DEFINE_REGISTRY(HWC2CHW);
MMDEPLOY_DEFINE_REGISTRY(Normalize);
MMDEPLOY_DEFINE_REGISTRY(Crop);
MMDEPLOY_DEFINE_REGISTRY(Flip);

}  // namespace mmdeploy::operation
