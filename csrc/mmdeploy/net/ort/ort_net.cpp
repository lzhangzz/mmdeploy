// Copyright (c) OpenMMLab. All rights reserved.

#include "ort_net.h"

#include <Windows.h>

#include <algorithm>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/net/net_module.h"
#include "onnxruntime_register.h"

extern "C" {
void default_mmdeploy_test_log() { MMDEPLOY_ERROR("default log"); }

void mmdeploy_test_log();
}

//#pragma comment(linker, "/alternatename:_mmdeploy_test_log=_default_mmdeploy_test_log")
#pragma comment(linker, "/ALTERNATENAME:mmdeploy_test_log=default_mmdeploy_test_log")


namespace mmdeploy::framework {

static TensorShape to_shape(const Ort::TypeInfo& info) {
  auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
  return {shape.begin(), shape.end()};
}

static Result<DataType> ConvertElementType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::kFLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::kHALF;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::kINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::kINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::kINT64;
    default:
      MMDEPLOY_ERROR("unsupported ONNXTensorElementDataType: {}", static_cast<int>(type));
      return Status(eNotSupported);
  }
}

// TODO: handle datatype
Result<void> OrtNet::Init(const Value& args) {
  mmdeploy_test_log();
  MMDEPLOY_ERROR("");
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();
  MMDEPLOY_ERROR("");
  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  MMDEPLOY_ERROR("");
  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  MMDEPLOY_ERROR("");
  OUTCOME_TRY(auto onnx, model.ReadFile(config.net));
  MMDEPLOY_ERROR("");
  Ort::SessionOptions options;
  options.SetLogSeverityLevel(3);

  RegisterCustomOps(options, OrtGetApiBase());
  MMDEPLOY_ERROR("");
  if (device_.is_device()) {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_.device_id();
    // TODO set compute stream
    options.AppendExecutionProvider_CUDA(cuda_options);
  }
  session_ = {env_, onnx.data(), onnx.size(), options};
  MMDEPLOY_ERROR("");
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);
  MMDEPLOY_ERROR("");
  auto n_inputs = session_.GetInputCount();
  MMDEPLOY_ERROR("");
  // force negative shape to be empty
  auto filter_shape = [](TensorShape& shape) {
    if (std::any_of(begin(shape), end(shape), [](auto x) { return x < 0; })) {
      shape = {};
    }
  };
  MMDEPLOY_ERROR("");
  for (int i = 0; i < n_inputs; ++i) {
#if ORT_API_VERSION >= 13
    auto input_name = session_.GetInputNameAllocated(i, allocator).release();
#else
    auto input_name = session_.GetInputName(i, allocator);
#endif
    MMDEPLOY_ERROR("");
    auto type_info = session_.GetInputTypeInfo(i);
    auto shape = to_shape(type_info);
    MMDEPLOY_ERROR("input {}, shape = {}", i, shape);
    filter_shape(shape);
    MMDEPLOY_ERROR("");
    OUTCOME_TRY(auto data_type,
                ConvertElementType(type_info.GetTensorTypeAndShapeInfo().GetElementType()));
    MMDEPLOY_ERROR("");
    TensorDesc t_desc{device_, data_type, shape, input_name};
    MMDEPLOY_ERROR("platform registry: {}", (void*)&gPlatformRegistry());
    MMDEPLOY_ERROR("({}, {}) {} {} {}", device_.platform_id(), device_.device_id(), data_type,
                   shape, input_name);
    Tensor t(t_desc);
    MMDEPLOY_ERROR("");
    input_tensors_.emplace_back(t);
    MMDEPLOY_ERROR("");
    allocator.Free(input_name);
  }
  MMDEPLOY_ERROR("");
  auto n_outputs = session_.GetOutputCount();

  for (int i = 0; i < n_outputs; ++i) {
#if ORT_API_VERSION >= 13
    auto output_name = session_.GetOutputNameAllocated(i, allocator).release();
#else
    auto output_name = session_.GetOutputName(i, allocator);
#endif
    auto type_info = session_.GetOutputTypeInfo(i);
    auto shape = to_shape(type_info);
    MMDEPLOY_DEBUG("output {}, shape = {}", i, shape);
    filter_shape(shape);
    OUTCOME_TRY(auto data_type,
                ConvertElementType(type_info.GetTensorTypeAndShapeInfo().GetElementType()));
    output_tensors_.emplace_back(TensorDesc{device_, data_type, shape, output_name});
    allocator.Free(output_name);
  }
  MMDEPLOY_ERROR("");
  return success();
}

Result<void> OrtNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

Result<void> OrtNet::Deinit() { return success(); }

Result<Span<Tensor>> OrtNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> OrtNet::GetOutputTensors() { return output_tensors_; }

Result<void> OrtNet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

static Ort::MemoryInfo MemoryInfo(const TensorDesc& desc) {
  const char* device_name = desc.device.is_host() ? "Cpu" : "Cuda";
  Ort::MemoryInfo memory_info(device_name, OrtDeviceAllocator, desc.device.device_id(),
                              OrtMemTypeDefault);
  return memory_info;
}

static Ort::Value AsOrtValue(Tensor& tensor) {
  auto memory_info = MemoryInfo(tensor.desc());
  std::vector<int64_t> shape(begin(tensor.shape()), end(tensor.shape()));
  return Ort::Value::CreateTensor(memory_info, tensor.data<float>(), tensor.size(), shape.data(),
                                  shape.size());
}

static Result<Tensor> AsTensor(Ort::Value& value, const Device& device) {
  auto info = value.GetTensorTypeAndShapeInfo();
  TensorDesc desc;
  desc.shape = info.GetShape();
  desc.device = device;
  OUTCOME_TRY(desc.data_type, ConvertElementType(info.GetElementType()));
  std::shared_ptr<void> data(const_cast<void*>(value.GetTensorData<void>()), [](void*) {});
  return Tensor(desc, data);
}

Result<void> OrtNet::Forward() {
  try {
    OUTCOME_TRY(stream_.Wait());
    Ort::IoBinding binding(session_);
    std::vector<Ort::Value> inputs;
    std::vector<Ort::Value> outputs;
    Ort::RunOptions options;

    inputs.reserve(input_tensors_.size());
    for (auto& t : input_tensors_) {
      inputs.push_back(AsOrtValue(t));
      binding.BindInput(t.name(), inputs.back());
    }

    // TODO: We are in the same situation as PPL.nn, the backend can't infer shapes
    //  without executing forward
    for (auto& t : output_tensors_) {
      binding.BindOutput(t.name(), MemoryInfo(t.desc()));
    }

    session_.Run({}, binding);

    outputs = binding.GetOutputValues();
    for (size_t i = 0; i < output_tensors_.size(); ++i) {
      OUTCOME_TRY(auto tmp, AsTensor(outputs[i], output_tensors_[i].device()));
      output_tensors_[i].Reshape(tmp.shape());
      OUTCOME_TRY(tmp.CopyTo(output_tensors_[i], stream_));
    }

    OUTCOME_TRY(stream_.Wait());
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR(e.what());
    return Status(eFail);
  }
  return success();
}

static std::unique_ptr<Net> Create(const Value& args) {
  try {
    auto p = std::make_unique<OrtNet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating OrtNet: {}", r.error().message().c_str());
      return nullptr;
    }
  } catch (const Exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating ORTNet: {}", e.what());
    return nullptr;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating ORTNet: {}", e.what());
    return nullptr;
  }
}

// MMDEPLOY_REGISTER_FACTORY_FUNC(Net, (onnxruntime, 0), Create);

typedef void* (*get_net_registry_t)();

// static ::mmdeploy::Registerer register_onnx{
//     [creator = ::mmdeploy::SimpleCreator<Net>("onnxruntime", 0, Create)]() mutable {
//       auto handle = GetModuleHandle("mmdeploy_python.pyd");
//       fprintf(stderr, "handle = %p\n", handle);
//       auto proc = (get_net_registry_t)GetProcAddress(handle, "mmdeploy_net_registry");
//       fprintf(stderr, "proc = %p\n", proc);
//       auto& registry = *(Registry<Net>*)proc();
//       fprintf(stderr, "registry = %p\n", &registry);
//       registry.Add(creator);
//     }};

}  // namespace mmdeploy::framework

extern "C" {
MMDEPLOY_EXPORT int init(void* registry) {
  static mmdeploy::SimpleCreator<mmdeploy::framework::Net> creator("onnxruntime", 0,
                                                                   mmdeploy::framework::Create);
  static_cast<mmdeploy::Registry<mmdeploy::framework::Net>*>(registry)->Add(creator);
  return 0;
}
}