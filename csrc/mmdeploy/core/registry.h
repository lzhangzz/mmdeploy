// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_REGISTRY_H
#define MMDEPLOY_REGISTRY_H

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/mpl/range.h"
#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/mpl/tag_invoke.h"

namespace mmdeploy {

namespace _registry {

using std::optional;
using std::string_view;

template <typename T, typename = void>
struct _get_signature {
  static_assert(!std::is_same_v<T, T>, "tag T is not associated with a signature");
};

template <typename T>
using get_signature_t = decltype(get_signature(type_identity<T>{}));

template <typename T>
struct _get_signature<T, std::void_t<get_signature_t<T>>> {
  using type = typename get_signature_t<T>::type;
};

template <typename T>
using GetSignature = typename _get_signature<T>::type;

template <typename Tag>
class Creator;

template <>
class MMDEPLOY_API Creator<void> {
 public:
  virtual ~Creator() = default;
  virtual string_view name() const noexcept = 0;
  virtual int version() const noexcept { return 0; }
};

template <typename Ret, typename... Args>
class MMDEPLOY_API Creator<Ret(Args...)> : public Creator<void> {
 public:
  virtual Ret Create(Args... args) = 0;
};

template <typename Tag>
class SimpleCreator;

template <typename Ret, typename... Args>
class SimpleCreator<Ret(Args...)> : public Creator<Ret(Args...)> {
 public:
  using FunctionType = std::function<Ret(Args...)>;

  SimpleCreator(const string_view& name, int version, FunctionType func)
      : name_(name), version_(version), func_(std::move(func)) {}

  string_view name() const noexcept override { return name_; }
  int version() const noexcept override { return version_; }
  Ret Create(Args... args) override { return func_(args...); }

 private:
  std::string name_;
  int version_;
  FunctionType func_;
};

template <typename Tag>
class Registry;

template <>
class MMDEPLOY_API Registry<void> {
 public:
  Registry();

  ~Registry();

  bool AddCreator(Creator<void>& creator);

  Creator<void>* GetCreator(const string_view& name, int version);

  Span<Creator<void>*> Creators();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

template <typename Signature>
struct _result_of;

template <typename R, typename... As>
struct _result_of<R(As...)> {
  using type = R;
};

template <typename Tag>
class Registry : public Registry<void> {
 public:
  using Signature = GetSignature<Tag>;
  using CreatorType = Creator<Signature>;

  bool Add(CreatorType& creator) & { return AddCreator(creator); }

  CreatorType* Get(const string_view& name, int version) & {
    return static_cast<CreatorType*>(GetCreator(name, version));
  }

  CreatorType* Get(const string_view& name) & { return Get(name, -1); }

  template <typename... Args>
  auto Create(const std::pair<string_view, int>& desc,
              Args&&... args) & -> optional<typename _result_of<Signature>::type> {
    if (auto creator = Get(desc.first, desc.second); creator) {
      return creator->Create((Args &&) args...);
    } else {
      return std::nullopt;
    }
  }

  template <typename... Args>
  auto Create(const string_view& name, Args&&... args) & {
    return Create(std::pair{name, -1}, (Args &&) args...);
  }

  Span<CreatorType*> Creators() & {
    auto creators = Registry<void>::Creators();
    return {reinterpret_cast<CreatorType**>(creators.data()), creators.size()};
  }

  auto List() & {
    std::vector<std::pair<string_view, int>> list;
    for (const auto& creator : Creators()) {
      list.emplace_back(creator->name(), creator->version());
    }
    return list;
  }
};

template <typename F>
class Registerer {
 public:
  explicit Registerer(F f) : func_(std::move(f)) { func_(); }

 private:
  F func_;
};

template <typename T>
struct get_registry_cpo {
  auto& operator()() const { return tag_invoke(*this); }
};

template <typename T>
inline constexpr get_registry_cpo<T> gRegistry{};

}  // namespace _registry

using _registry::gRegistry;
using _registry::Registerer;
using _registry::Registry;

template <typename Tag>
using Creator = _registry::Creator<_registry::GetSignature<Tag>>;

template <typename Tag>
using SimpleCreator = _registry::SimpleCreator<_registry::GetSignature<Tag>>;

}  // namespace mmdeploy

// Specify creator signature for tag
#define MMDEPLOY_CREATOR_SIGNATURE(tag, signature) \
  ::mmdeploy::type_identity<signature> get_signature(::mmdeploy::type_identity<tag>);

#define MMDEPLOY_DECLARE_REGISTRY(tag, signature) \
  MMDEPLOY_CREATOR_SIGNATURE(tag, signature)      \
  MMDEPLOY_API ::mmdeploy::Registry<tag>& tag_invoke(::mmdeploy::_registry::get_registry_cpo<tag>);

#define MMDEPLOY_DECLARE_REGISTRY_EXPAND(tag, signature)        \
  MMDEPLOY_CREATOR_SIGNATURE(tag, MMDEPLOY_PP_EXPAND signature) \
  MMDEPLOY_API ::mmdeploy::Registry<tag>& tag_invoke(::mmdeploy::_registry::get_registry_cpo<tag>);

#define MMDEPLOY_DEFINE_REGISTRY(tag)                                                   \
  ::mmdeploy::Registry<tag>& tag_invoke(::mmdeploy::_registry::get_registry_cpo<tag>) { \
    static ::mmdeploy::Registry<tag> instance{};                                        \
    return instance;                                                                    \
  }

#define MMDEPLOY_REGISTER_CREATOR(tag, creator_type)                    \
  static ::mmdeploy::Registerer MMDEPLOY_ANONYMOUS_VARIABLE(register_){ \
      [creator = creator_type{}]() mutable { ::mmdeploy::gRegistry<tag>().Add(creator); }};

#define MMDEPLOY_CREATOR_DESC(name, version) #name, version

#define MMDEPLOY_REGISTER_FACTORY_FUNC(tag, creator_desc, func)                                  \
  static ::mmdeploy::Registerer MMDEPLOY_ANONYMOUS_VARIABLE(register_){                          \
      [creator =                                                                                 \
           ::mmdeploy::SimpleCreator<tag>(MMDEPLOY_CREATOR_DESC creator_desc, func)]() mutable { \
        ::mmdeploy::gRegistry<tag>().Add(creator);                                               \
      }};

#endif  // MMDEPLOY_REGISTRY_H
