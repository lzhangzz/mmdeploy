// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_SERIALIZATION_H_
#define MMDEPLOY_SRC_CORE_SERIALIZATION_H_

#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/mpl/detected.h"
#include "mmdeploy/core/mpl/tag_invoke.h"
#include "mmdeploy/core/mpl/type_traits.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy {

#define MMDEPLOY_ARCHIVE_NVP(archive, ...) archive(MMDEPLOY_PP_MAP(MMDEPLOY_NVP, __VA_ARGS__))

#define MMDEPLOY_ARCHIVE_MEMBERS(...)           \
  template <typename Archive>                   \
  void serialize(Archive& archive) {            \
    MMDEPLOY_ARCHIVE_NVP(archive, __VA_ARGS__); \
  }

#define MMDEPLOY_NVP(var) \
  ::mmdeploy::serialization::NamedValue { std::forward_as_tuple(#var, var) }

namespace serialization {

struct save_cpo {
  template <typename Archive, typename T>
  auto operator()(Archive&& a, T&& v) const
      noexcept(noexcept(tag_invoke(*this, (Archive &&) a, (T &&) v)))
          -> decltype(tag_invoke(*this, (Archive &&) a, (T &&) v)) {
    return tag_invoke(*this, (Archive &&) a, (T &&) v);
  }
};

inline constexpr save_cpo save{};

struct load_cpo {
  template <typename Archive, typename T>
  auto operator()(Archive&& a, T&& v) const
      noexcept(noexcept(tag_invoke(*this, (Archive &&) a, (T &&) v)))
          -> decltype(tag_invoke(*this, (Archive &&) a, (T &&) v)) {
    return tag_invoke(*this, (Archive &&) a, (T &&) v);
  }
};

inline constexpr load_cpo load{};

struct serialize_cpo {
  template <typename Archive, typename T>
  auto operator()(Archive&& a, T&& v) const
      noexcept(noexcept(tag_invoke(*this, (Archive &&) a, (T &&) v)))
          -> decltype(tag_invoke(*this, (Archive &&) a, (T &&) v)) {
    return tag_invoke(*this, (Archive &&) a, (T &&) v);
  }
};

inline constexpr serialize_cpo serialize{};

template <typename NameT, typename ValueT>
class NamedValue {
 public:
  explicit NamedValue(std::tuple<NameT, ValueT>&& data) : data_(std::move(data)) {}
  template <typename Archive>
  void serialize(Archive& archive) {
    archive.named_value(std::forward<NameT>(std::get<0>(data_)),
                        std::forward<ValueT>(std::get<1>(data_)));
  }
  std::tuple<NameT, ValueT>& data() { return data_; }

 private:
  std::tuple<NameT, ValueT> data_;
};

template <typename T>
struct array_tag {
  explicit array_tag(std::size_t size) : size_(size) {}
  std::size_t size() const { return size_; }
  std::size_t size_;
};

template <typename T>
struct object_tag {};

template <typename T>
using mapped_type_t = typename T::mapped_type;

template <typename T>
using has_mapped_type = detail::is_detected<mapped_type_t, T>;

template <typename T>
using get_size_t = decltype(std::declval<T>().size());

template <typename T>
using has_size = detail::is_detected<get_size_t, T>;

template <typename T>
using reserve_t = decltype(std::declval<T>().reserve(std::size_t{0}));

template <typename T>
using has_reserve = detail::is_detected<reserve_t, T>;

// range -> array archive
template <typename Archive, typename T, typename U = remove_cvref_t<T>,
          typename ValueType = typename U::value_type,
          std::enable_if_t<!std::is_same_v<U, std::string>, int> = 0>
auto tag_invoke(save_cpo, Archive& archive, T&& iterable)
    -> std::void_t<decltype(iterable.begin(), iterable.end())> {
  if constexpr (has_size<T>::value) {
    archive.init(array_tag<ValueType>(iterable.size()));
  }
  for (auto&& x : iterable) {
    archive.item(std::forward<decltype(x)>(x));
  }
}

template <typename T, typename = void>
struct is_range : std::false_type {};

template <typename T>
struct is_range<T, decltype(std::declval<T&>().begin(), std::declval<T&>().end(), void())>
    : std::true_type {};

template <typename T>
inline constexpr bool range = is_range<T>::value;

template <typename T>
inline constexpr bool disable_sized_range = false;

template <typename T>
inline constexpr bool sized_range = !disable_sized_range<T> && range<T> && has_size<T>::value;

template <typename T, typename = void>
struct has_data : std::false_type {};

template <typename T>
struct has_data<T, decltype(std::declval<T&>().data(), void())> : std::true_type {};

template <typename T>
inline constexpr bool contiguous_range = range<T> && has_data<T>::value;

//template <typename Archive, typename T, std::enable_if_t<contiguous_range<T>, int> = 0>
//auto tag_invoke(save_cpo, Archive& archive, T&& range) {
//  std::copy(((T &&) range).begin(), ((T&&)range).end(), )
//}

// template <typename T0, typename T1>
// class KeyValue {
//  public:
//   explicit KeyValue(std::tuple<T0, T1>&& data) : data_(std::move(data)) {}
//   template <typename Archive>
//   void serialize(Archive& archive) {
//     archive.named_value("key", std::forward<T0>(std::get<0>(data_)));
//     archive.named_value("value", std::forward<T1>(std::get<1>(data_)));
//   }
//   std::tuple<T0, T1>& data() { return data_; }
//
//  private:
//   std::tuple<T0, T1> data_;
// };

// template <typename Archive, typename T, typename U = remove_cvref_t<T>,
//           typename KeyType = typename U::key_type, typename MappedType = typename
//           U::mapped_type, std::enable_if_t<!std::is_constructible_v<std::string, KeyType>,
//           int> = 0>
// auto tag_invoke(save_cpo, Archive& archive, T& object)
//     -> std::void_t<decltype(object.begin(), object.end())> {
//   if constexpr (has_size<T>::value) {
//     // TODO: provide meaningful type info
//     archive.init(array_tag<void>(object.size()));
//   }
//   for (auto&& [k, v] : object) {
//     archive.item(KeyValue{
//         std::forward_as_tuple(std::forward<decltype(k)>(k), std::forward<decltype(v)>(v))});
//   }
// }

// object -> object archive
template <typename Archive, typename T, typename U = remove_cvref_t<T>,
          typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
          std::enable_if_t<std::is_constructible_v<std::string, KeyType>, int> = 0>
auto tag_invoke(save_cpo, Archive& archive, T& object)
    -> std::void_t<decltype(object.begin(), object.end())> {
  if constexpr (has_size<T>::value) {
    archive.init(object_tag<MappedType>());
  }
  for (auto&& [k, v] : object) {
    archive.named_value(std::forward<decltype(k)>(k), std::forward<decltype(v)>(v));
  }
}

namespace detail {

template <typename Archive, typename T, std::size_t... Is>
void save_tuple_impl(Archive& archive, T&& tuple, std::index_sequence<Is...>) {
  (archive.item(std::get<Is>(std::forward<T>(tuple))), ...);
}

template <typename Archive, typename T, size_t... Is>
void load_tuple_impl(Archive& archive, T& tuple, std::index_sequence<Is...>) {
  (archive.item(std::get<Is>(tuple)), ...);
}

}  // namespace detail

// tuple -> array archive
template <typename Archive, typename... Ts>
void tag_invoke(save_cpo, Archive& archive, const std::tuple<Ts...>& tuple) {
  detail::save_tuple_impl(archive, tuple, std::index_sequence_for<Ts...>{});
}

// fixed-size array -> array archive
template <typename Archive, typename T, std::size_t N>
void tag_invoke(save_cpo, Archive& archive, T (&v)[N]) {
  archive.init(array_tag<T>(N));
  for (std::size_t i = 0; i < N; ++i) {
    archive.item(v[i]);
  }
}

// array archive -> tuple
template <typename Archive, typename... Ts>
void tag_invoke(load_cpo, Archive& archive, std::tuple<Ts...>& tuple) {
  std::size_t size{};
  archive.init(size);
  if (size != sizeof...(Ts)) {
    throw_exception(eShapeMismatch);
  }
  detail::load_tuple_impl(archive, tuple, std::index_sequence_for<Ts...>{});
}

// array archive -> std::vector
template <typename Archive, typename T, typename U = remove_cvref_t<T>,
          typename ValueType = typename U::value_type,
          std::enable_if_t<!std::is_same_v<U, std::string>, int> = 0>
auto tag_invoke(load_cpo, Archive&& archive, T&& vec)
    -> std::void_t<decltype(vec.push_back(ValueType{}))> {
  std::size_t size{};
  archive.init(size);
  vec.clear();
  for (std::size_t i = 0; i < size; ++i) {
    ValueType v{};
    archive.item(v);
    vec.push_back(std::move(v));
  }
}

// array archive -> std::array
template <typename Archive, typename T, std::size_t N>
void tag_invoke(load_cpo, Archive& archive, std::array<T, N>& v) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    archive.item(v[i]);
  }
}

// array archive -> fixed-size array
template <typename Archive, typename T, std::size_t N>
void tag_invoke(load_cpo, Archive& archive, T (&v)[N]) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    archive.item(v[i]);
  }
}

// array archive -> set
template <typename Archive, typename T, typename U = remove_cvref_t<T>,
          typename ValueType = typename U::value_type,
          std::enable_if_t<std::conjunction_v<std::is_default_constructible<ValueType>,
                                              std::negation<has_mapped_type<U>>>,
                           int> = 0>
auto tag_invoke(load_cpo, Archive&& archive, T&& set)
    -> std::void_t<decltype(set.insert(std::declval<ValueType>()))> {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    ValueType v{};
    archive.item(v);
    set.insert(std::move(v));
  }
}

// key-value array archive -> map<K, V>
// template <
//    typename Archive, typename T, typename U = remove_cvref_t<T>,
//    typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
//    std::enable_if_t<std::conjunction_v<std::negation<std::is_constructible<KeyType,
//    std::string>>,
//                                        std::is_default_constructible<KeyType>,
//                                        std::is_default_constructible<MappedType>>,
//                     int> = 0>
// void tag_invoke(load_cpo, Archive&& archive, T&& object) {
//  std::size_t size{};
//  archive.init(size);
//  for (std::size_t i = 0; i < size; ++i) {
//    KeyType key;
//    MappedType mapped;
//    archive.item(KeyValue{std::tie(key, mapped)});
//    object.insert({std::move(key), std::move(mapped)});
//  };
//}

// object archive -> map<string, T>
template <typename Archive, typename T, typename U = remove_cvref_t<T>,
          typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
          std::enable_if_t<std::conjunction_v<std::is_constructible<KeyType, std::string>,
                                              std::is_default_constructible<MappedType>>,
                           int> = 0>
void tag_invoke(load_cpo, Archive&& archive, T&& object) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    std::string name;
    MappedType value{};
    archive.named_value(name, value);
    object.insert({std::move(name), std::move(value)});
  }
}

}  // namespace serialization

};  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_SERIALIZATION_H_
