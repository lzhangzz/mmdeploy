// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/operation/vision.h"

namespace mmdeploy::operation::cuda {

namespace impl {

template <typename T, int channels>
void Crop(const T* src, int src_w, T* dst, int dst_h, int dst_w, int offset_h, int offset_w,
          cudaStream_t stream);

}

class CropImpl : public Crop {
 public:
  using Crop::Crop;

  Result<Tensor> crop(const Tensor& tensor, int top, int left, int bottom, int right) override {
    OUTCOME_TRY(auto device_tensor, MakeAvailableOnDevice(tensor, device(), stream()));

    SyncOnScopeExit sync(stream(), device_tensor.buffer() != tensor.buffer(), device_tensor);

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    auto desc = device_tensor.desc();

    int h = bottom - top + 1;
    int w = right - left + 1;
    int c = desc.shape[3];
    auto type = desc.data_type;

    TensorShape shape{1, bottom - top + 1, right - left + 1, tensor.desc().shape[3]};
    TensorDesc dst_desc{device(), tensor.desc().data_type, shape, desc.name};
    Tensor dst_tensor{dst_desc};
    assert(device_.is_device());
    if (DataType::kINT8 == type) {
      auto input = device_tensor.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      if (3 == c) {
        impl::Crop<uint8_t, 3>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else if (1 == c) {
        impl::Crop<uint8_t, 1>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else if (DataType::kFLOAT == type) {
      auto input = static_cast<float*>(device_tensor.buffer().GetNative());
      auto output = static_cast<float*>(dst_tensor.buffer().GetNative());
      if (3 == c) {
        impl::Crop<float, 3>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else if (1 == c) {
        impl::Crop<float, 1>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else {
      MMDEPLOY_ERROR("unsupported channels {}", c);
      return Status(eNotSupported);
    }
    return dst_tensor;
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Crop, (cuda, 0), [](const Context& context) {
  return std::make_unique<CropImpl>(context);
});

}  // namespace mmdeploy::operation::cuda
