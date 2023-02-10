// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_RANGE_H
#define MMDEPLOY_RANGE_H

#include <utility>

namespace mmdeploy {

template <typename Iterator>
class iterator_range {
 public:
  explicit iterator_range(Iterator first, Iterator last) : first_(first), last_(last) {}
  explicit iterator_range(std::pair<Iterator, Iterator> range)
      : iterator_range(range.first, range.second) {}

  auto begin() noexcept { return first_; }
  auto end() noexcept { return last_; }

 private:
  Iterator first_;
  Iterator last_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_RANGE_H
