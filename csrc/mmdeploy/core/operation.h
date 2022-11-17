// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"

namespace mmdeploy::operation {

using namespace mmdeploy::framework;
using std::string_view;
using std::unique_ptr;

class MMDEPLOY_API Context {
 public:
  explicit Context(Device device);
  explicit Context(Stream stream);
  explicit Context(Device device, Stream stream);
  ~Context();

  Context(const Context&) = delete;
  Context(Context&&) noexcept = delete;
  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) noexcept = delete;

  void Track(const Tensor& tensor) { buffers_.push_back(tensor.buffer()); }
  void Track(const Mat& mat) { buffers_.push_back(mat.buffer()); };
  void Track(const Buffer& buffer) { buffers_.push_back(buffer); };

  template <typename T, typename... Args>
  T Create(Args&&... args) {
    return _track(T((Args &&) args...));
  }

  const Device& device() const noexcept { return device_; }
  Stream& stream() noexcept { return stream_; }
  const std::vector<Buffer>& buffers() const noexcept { return buffers_; }

 private:
  Tensor&& _track(Tensor&& tensor) {
    Track(tensor);
    return std::move(tensor);
  }
  Mat&& _track(Mat&& mat) {
    Track(mat);
    return std::move(mat);
  }
  Buffer&& _track(Buffer&& buffer) {
    Track(buffer);
    return std::move(buffer);
  }

 private:
  Device device_;
  Stream stream_;
  std::vector<Buffer> buffers_;
  Context* parent_;
};

MMDEPLOY_API Context& gContext();

template <typename T, typename... Args>
static unique_ptr<T> Create(Args&&... args) {
  std::vector<Device> candidates{gContext().device()};
  if (candidates[0].is_device()) {
    candidates.emplace_back(0);
  }
  for (const auto& device : candidates) {
    if (auto platform = GetPlatformName(device)) {
      if (auto creator = gRegistry<T>().Get(platform)) {
        Context context(device);
        return creator->Create((Args &&) args...);
      }
    }
  }
  return nullptr;
}

class Operation {
 public:
  Operation() : device_(gContext().device()) {}
  virtual ~Operation() = default;

  const Device& device() const noexcept { return device_; }
  static Stream& stream() noexcept { return gContext().stream(); }

 protected:
  Device device_;
};

namespace _apply {

inline Result<void> Copy(const Buffer& src, Buffer& dst, size_t size, Stream& stream) {
  OUTCOME_TRY(stream.Copy(src, dst, size));
  if (dst.GetDevice() != stream.GetDevice()) {
    OUTCOME_TRY(stream.Wait());
  }
  return success();
}

inline Result<Tensor> Secure(const Tensor& val, const Device& device, Stream& stream) {
  if (val.device() == device) {
    return val;
  }

  TensorDesc desc{device, val.data_type(), val.shape(), val.name()};
  Tensor dst(desc);

  OUTCOME_TRY(Copy(val.buffer(), dst.buffer(), val.byte_size(), stream));

  gContext().Track(dst);
  return dst;
}

inline Result<Mat> Secure(const Mat& val, const Device& device, Stream& stream) {
  if (val.device() == device) {
    return val;
  }

  Mat dst{val.height(), val.width(), val.pixel_format(), val.type(), device};

  OUTCOME_TRY(Copy(val.buffer(), dst.buffer(), val.byte_size(), stream));

  gContext().Track(dst);
  return dst;
}

template <typename T>
struct _base_handler {
  using type = T;
  static T input(T x, const Device&, Stream&) { return x; }
  static T pass(T x) { return x; }
  static void output(T) {}
};

template <typename T>
struct _handler : _base_handler<T> {};

template <>
struct _handler<const Tensor&> : _base_handler<const Tensor&> {
  using type = Result<Tensor>;
  static type input(const Tensor& tensor, const Device& device, Stream& stream) {
    return Secure(tensor, device, stream);
  }
  static const Tensor& pass(const type& tensor) { return tensor.value(); }
  static void output(const Result<Tensor>&) {}
};

template <>
struct _handler<const Mat&> {
  using type = Result<Mat>;
  static type input(const Mat& mat, const Device& device, Stream& stream) {
    return Secure(mat, device, stream);
  }
  static const Mat& pass(const type& mat) { return mat.value(); }
  static void output(const type&) {}
};

template <>
struct _handler<const std::vector<Tensor>&> {
  using type = Result<std::vector<Tensor>>;
  static type input(const std::vector<Tensor>& tensors, const Device& device, Stream& stream) {
    std::vector<Tensor> rets(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      OUTCOME_TRY(rets[i], Secure(tensors[i], device, stream));
    }
    return rets;
  }
  static const std::vector<Tensor>& pass(const type& tensors) { return tensors.value(); }
  static void output(const type&) {}
};

template <>
struct _handler<Tensor&> : _base_handler<Tensor&> {
  static void output(Tensor& tensor) { gContext().Track(tensor); }
};

template <>
struct _handler<Mat&> : _base_handler<Mat&> {
  static void output(Mat& mat) { gContext().Track(mat); }
};

inline Result<void> _check() { return success(); }

template <typename T, typename... Ts>
Result<void> _check(T&& x, Ts&&... xs) {
  return _check((Ts &&) xs...);
}

template <typename T, typename... Ts>
Result<void> _check(Result<T>& x, Ts&&... xs) {
  OUTCOME_TRY(x);
  return _check((Ts &&) xs...);
}

template <typename Sig>
struct apply_impl {
  static_assert(!std::is_same_v<Sig, Sig>, "Not a member function pointer");
};

template <typename Ret, typename C, typename... Args>
struct apply_impl<Ret (C::*)(Args...)> {
  const Device& device;
  Stream& stream;

  template <typename Op, typename... As>
  Result<void> operator()(Op& op, As&&... as) const {
    return apply(op, std::index_sequence_for<Args...>{}, (As &&) as...);
  }

  template <typename Op, typename... As, size_t... Is>
  Result<void> apply(Op& op, std::index_sequence<Is...>, As&&... as) const {
    // transform input args and store them in a tuple
    std::tuple<typename _handler<Args>::type...> tmps{
        _handler<Args>::input((As &&) as, device, stream)...};

    // check if any copy operations are failed
    OUTCOME_TRY(_check(std::get<Is>(tmps)...));

    // apply the operation
    OUTCOME_TRY(op.apply(_handler<Args>::pass(std::get<Is>(tmps))...));

    // track output data (Tensor& and Mat&)
    (_handler<Args>::output(std::get<Is>(tmps)), ...);
    return success();
  }
};

template <typename Op, typename... Args>
Result<void> apply(Op& op, Args&&... args) {
  _apply::apply_impl<decltype(&Op::apply)> impl{op.device(), op.stream()};
  return impl(op, (Args &&) args...);
}

}  // namespace _apply

template <typename Op>
class Managed {
 public:
  Managed() = default;

  explicit Managed(std::unique_ptr<Op> op) : op_(std::move(op)) {}

  template <typename... Args>
  Result<void> Apply(Args&&... args) {
    assert(op_);
    return _apply::apply(*op_, (Args &&) args...);
  }

  template <typename... Args>
  static Managed<Op> Create(Args&&... args) {
    return Managed<Op>(operation::Create<Op>((Args &&) args...));
  }

 private:
  std::unique_ptr<Op> op_;
};

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
