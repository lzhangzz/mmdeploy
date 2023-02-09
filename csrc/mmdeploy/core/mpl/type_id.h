// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TYPE_ID_H
#define MMDEPLOY_TYPE_ID_H

#include "mmdeploy/core/mpl/tag_invoke.h"

namespace mmdeploy {

namespace traits {

using type_id_t = uint64_t;

template <typename T>
struct get_type_id_t {
  auto operator()() const { return tag_invoke(*this); }
};

template <typename T>
inline constexpr get_type_id_t<T> GetTypeId{};

template <typename T>
inline constexpr auto has_type_id = tag_invocable<get_type_id_t<T>>;

#define MMDEPLOY_REGISTER_TYPE_ID(type, id)                  \
  inline constexpr ::mmdeploy::traits::type_id_t tag_invoke( \
      ::mmdeploy::traits::get_type_id_t<type>) {             \
    return id;                                               \
  }

MMDEPLOY_REGISTER_TYPE_ID(void, static_cast<mmdeploy::traits::type_id_t>(-1));

static_assert(has_type_id<void>);
static_assert(!has_type_id<struct tmp>);

}  // namespace traits

}  // namespace mmdeploy

#endif  // MMDEPLOY_TYPE_ID_H
