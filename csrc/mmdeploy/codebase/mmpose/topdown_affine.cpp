// Copyright (c) OpenMMLab. All rights reserved.

#include <set>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgproc.hpp"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy::mmpose {

cv::Point2f operator*(const cv::Point2f& a, const cv::Point2f& b) {
  cv::Point2f c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

class TopDownAffine : public transform::Transform {
 public:
  explicit TopDownAffine(const Value& args) noexcept {
    use_udp_ = args.value("use_udp", use_udp_);
    backend_ = args.contains("backend") && args["backend"].is_string()
                   ? args["backend"].get<string>()
                   : backend_;
    stream_ = args["context"]["stream"].get<Stream>();
    assert(args.contains("image_size"));
    from_value(args["image_size"], image_size_);
    warp_affine_ = operation::Managed<operation::WarpAffine>::Create("bilinear");
    resize_ = operation::Managed<operation::Resize>::Create("bilinear");
    crop_ = operation::Managed<operation::Crop>::Create();
    pad_ = operation::Managed<operation::Pad>::Create("constant", 0);
  }

  ~TopDownAffine() override = default;

  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("top_down_affine input: {}", data);

    auto img = data["img"].get<Tensor>();

    // prepare data
    vector<float> bbox;
    vector<float> c;  // center
    vector<float> s;  // scale
    if (data.contains("center") && data.contains("scale")) {
      // after mmpose v0.26.0
      from_value(data["center"], c);
      from_value(data["scale"], s);
    } else {
      // before mmpose v0.26.0
      from_value(data["bbox"], bbox);
      Box2cs(bbox, c, s);
    }
    // end prepare data

    auto r = data["rotation"].get<float>();

    Tensor dst;
    if (use_udp_) {
      cv::Mat trans =
          GetWarpMatrix(r, {c[0] * 2.f, c[1] * 2.f}, {image_size_[0] - 1.f, image_size_[1] - 1.f},
                        {s[0] * 200.f, s[1] * 200.f});
      OUTCOME_TRY(warp_affine_.Apply(img, dst, trans.ptr<float>(), image_size_[1], image_size_[0]));
    } else if (0) {
      cv::Mat trans =
          GetAffineTransform({c[0], c[1]}, {s[0], s[1]}, r, {image_size_[0], image_size_[1]});
      OUTCOME_TRY(warp_affine_.Apply(img, dst, trans.ptr<float>(), image_size_[1], image_size_[0]));
    } else if (0) {
      auto w_roi = static_cast<int>(std::round(s[0] * 200));
      auto h_roi = static_cast<int>(std::round(s[1] * 200));
      auto x_offset = .5f * (w_roi - 1) + .5f * w_roi / image_size_[0];
      auto y_offset = .5f * (h_roi - 1) + .5f * h_roi / image_size_[1];
      auto x0_roi = static_cast<int>(std::round(c[0] - x_offset));
      auto y0_roi = static_cast<int>(std::round(c[1] - y_offset));
      c[0] = x0_roi + x_offset;
      c[1] = y0_roi + y_offset;
      s[0] = w_roi / 200.f;
      s[1] = h_roi / 200.f;
      auto t = 0, l = 0, b = 0, r = 0;
      l = std::max(-x0_roi, 0);
      t = std::max(-y0_roi, 0);
      x0_roi += l;
      y0_roi += t;
      auto x1_roi = x0_roi + w_roi - 1;
      auto y1_roi = y0_roi + h_roi - 1;
      r = std::max(x1_roi - static_cast<int>(img.shape(2)) + 1, 0);
      b = std::max(y1_roi - static_cast<int>(img.shape(1)) + 1, 0);
      Tensor pad = img;
      if (t + l + b + r) {
        Tensor tmp;
        OUTCOME_TRY(pad_.Apply(img, tmp, t, l, b, r));
        pad = tmp;
      }
      Tensor crop;
      OUTCOME_TRY(crop_.Apply(pad, crop, y0_roi, x0_roi, y1_roi, x1_roi));
      OUTCOME_TRY(resize_.Apply(crop, dst, image_size_[1], image_size_[0]));
    } else {
      s[0] *= 200;
      s[1] *= 200;
      const std::array img_roi{0, 0, (int)img.shape(2), (int)img.shape(1)};
      const std::array tmp_roi{0, 0, (int)image_size_[0], (int)image_size_[1]};
      auto roi = round({c[0] - s[0] / 2.f, c[1] - s[1] / 2.f, s[0], s[1]});
      auto src_roi = intersect(roi, img_roi);
      // prior scale factor
      auto factor = (float)image_size_[0] / s[0];
      // rounded dst roi
      auto dst_roi = round({(src_roi[0] - roi[0]) * factor,  //
                            (src_roi[1] - roi[1]) * factor,  //
                            src_roi[2] * factor,             //
                            src_roi[3] * factor});
      dst_roi = intersect(dst_roi, tmp_roi);
      // exact scale factors
      auto factor_x = (float)dst_roi[2] / src_roi[2];
      auto factor_y = (float)dst_roi[3] / src_roi[3];
      // center of src roi
      auto c_src_x = src_roi[0] + (src_roi[2] - 1) / 2.f;
      auto c_src_y = src_roi[1] + (src_roi[3] - 1) / 2.f;
      // center of dst roi
      auto c_dst_x = dst_roi[0] + (dst_roi[2] - 1) / 2.f;
      auto c_dst_y = dst_roi[1] + (dst_roi[3] - 1) / 2.f;
      // vector from c_dst to (w/2, h/2)
      auto v_dst_x = image_size_[0] / 2.f - c_dst_x;
      auto v_dst_y = image_size_[1] / 2.f - c_dst_y;
      // vector from c_src to corrected center
      auto v_src_x = v_dst_x / factor_x;
      auto v_src_y = v_dst_y / factor_y;
      // corrected center
      c[0] = c_src_x + v_src_x;
      c[1] = c_src_y + v_src_y;
      // corrected scale
      s[0] = image_size_[0] / factor_x / 200.f;
      s[1] = image_size_[1] / factor_y / 200.f;
      Tensor crop, resize;
      OUTCOME_TRY(crop_.Apply(img, crop, src_roi[1], src_roi[0], src_roi[1] + src_roi[3] - 1,
                              src_roi[0] + src_roi[2] - 1));
      OUTCOME_TRY(resize_.Apply(crop, resize, dst_roi[3], dst_roi[2]));
      auto pad_t = dst_roi[1];
      auto pad_l = dst_roi[0];
      auto pad_b = image_size_[1] - dst_roi[3] - pad_t;
      auto pad_r = image_size_[0] - dst_roi[2] - pad_l;
      if (pad_t + pad_l + pad_b + pad_r) {
        OUTCOME_TRY(pad_.Apply(resize, dst, pad_t, pad_l, pad_b, pad_r));
      } else {
        dst = resize;
      }
    }

    data["img_shape"] = {1, image_size_[1], image_size_[0], dst.shape(3)};
    data["img"] = std::move(dst);
    data["center"] = to_value(c);
    data["scale"] = to_value(s);
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

  static std::array<int, 4> round(const std::array<float, 4>& a) {
    return {
        static_cast<int>(std::round(a[0])),
        static_cast<int>(std::round(a[1])),
        static_cast<int>(std::round(a[2])),
        static_cast<int>(std::round(a[3])),
    };
  }

  // xywh
  template <typename T>
  static std::array<T, 4> intersect(std::array<T, 4> a, std::array<T, 4> b) {
    auto x1 = std::max(a[0], b[0]);
    auto y1 = std::max(a[1], b[1]);
    a[2] = std::min(a[0] + a[2], b[0] + b[2]) - x1;
    a[3] = std::min(a[1] + a[3], b[1] + b[3]) - y1;
    a[0] = x1;
    a[1] = y1;
    if (a[2] <= 0 || a[3] <= 0) {
      a = {};
    }
    return a;
  }

  void Box2cs(vector<float>& box, vector<float>& center, vector<float>& scale) {
    // bbox_xywh2cs
    float x = box[0];
    float y = box[1];
    float w = box[2];
    float h = box[3];
    float aspect_ratio = image_size_[0] * 1.0 / image_size_[1];
    center.push_back(x + w * 0.5);
    center.push_back(y + h * 0.5);
    if (w > aspect_ratio * h) {
      h = w * 1.0 / aspect_ratio;
    } else if (w < aspect_ratio * h) {
      w = h * aspect_ratio;
    }
    scale.push_back(w / 200 * 1.25);
    scale.push_back(h / 200 * 1.25);
  }

  cv::Mat GetWarpMatrix(float theta, cv::Size2f size_input, cv::Size2f size_dst,
                        cv::Size2f size_target) {
    theta = theta * 3.1415926 / 180;
    float scale_x = size_dst.width / size_target.width;
    float scale_y = size_dst.height / size_target.height;
    cv::Mat matrix = cv::Mat(2, 3, CV_32F);
    matrix.at<float>(0, 0) = std::cos(theta) * scale_x;
    matrix.at<float>(0, 1) = -std::sin(theta) * scale_x;
    matrix.at<float>(0, 2) =
        scale_x * (-0.5f * size_input.width * std::cos(theta) +
                   0.5f * size_input.height * std::sin(theta) + 0.5f * size_target.width);
    matrix.at<float>(1, 0) = std::sin(theta) * scale_y;
    matrix.at<float>(1, 1) = std::cos(theta) * scale_y;
    matrix.at<float>(1, 2) =
        scale_y * (-0.5f * size_input.width * std::sin(theta) -
                   0.5f * size_input.height * std::cos(theta) + 0.5f * size_target.height);
    return matrix;
  }

  cv::Mat GetAffineTransform(cv::Point2f center, cv::Point2f scale, float rot, cv::Size output_size,
                             cv::Point2f shift = {0.f, 0.f}, bool inv = false) {
    cv::Point2f scale_tmp = scale * 200;
    float src_w = scale_tmp.x;
    int dst_w = output_size.width;
    int dst_h = output_size.height;
    float rot_rad = 3.1415926 * rot / 180;
    cv::Point2f src_dir = rotate_point({0.f, src_w * -0.5f}, rot_rad);
    cv::Point2f dst_dir = {0.f, dst_w * -0.5f};

    cv::Point2f src_points[3];
    src_points[0] = center + scale_tmp * shift;
    src_points[1] = center + src_dir + scale_tmp * shift;
    src_points[2] = Get3rdPoint(src_points[0], src_points[1]);

    cv::Point2f dst_points[3];
    dst_points[0] = {dst_w * 0.5f, dst_h * 0.5f};
    dst_points[1] = dst_dir + cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);
    dst_points[2] = Get3rdPoint(dst_points[0], dst_points[1]);

    cv::Mat trans = inv ? cv::getAffineTransform(dst_points, src_points)
                        : cv::getAffineTransform(src_points, dst_points);
    trans.convertTo(trans, CV_32F);
    return trans;
  }

  cv::Point2f rotate_point(cv::Point2f pt, float angle_rad) {
    float sn = std::sin(angle_rad);
    float cs = std::cos(angle_rad);
    float new_x = pt.x * cs - pt.y * sn;
    float new_y = pt.x * sn + pt.y * cs;
    return {new_x, new_y};
  }

  cv::Point2f Get3rdPoint(cv::Point2f a, cv::Point2f b) {
    cv::Point2f direction = a - b;
    cv::Point2f third_pt = b + cv::Point2f(-direction.y, direction.x);
    return third_pt;
  }

 protected:
  operation::Managed<operation::WarpAffine> warp_affine_;
  operation::Managed<operation::Resize> resize_;
  operation::Managed<operation::Pad> pad_;
  operation::Managed<operation::Crop> crop_;
  bool use_udp_{false};
  vector<int> image_size_;
  std::string backend_;
  Stream stream_;
};

MMDEPLOY_REGISTER_TRANSFORM(TopDownAffine);

}  // namespace mmdeploy::mmpose
