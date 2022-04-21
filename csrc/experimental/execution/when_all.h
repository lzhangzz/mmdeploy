// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_H_

#include "utility.h"

namespace mmdeploy {

namespace __when_all {

template <typename... Senders>
using __concat_t = decltype(std::tuple_cat(std::declval<completion_signature_for_t<Senders>>()...));

template <typename CvrefReceiver, typename... Senders>
struct _Operation {
  struct type;
};
template <typename CvrefReceiver, typename... Senders>
using Operation = typename _Operation<CvrefReceiver, Senders...>::type;

template <typename CvrefReceiver, size_t Index, typename... Senders>
struct _Receiver {
  struct type;
};
template <typename CvrefReceiver, size_t Index, typename... Senders>
using receiver_t = typename _Receiver<CvrefReceiver, Index, Senders...>::type;

template <typename CvrefReceiver, size_t Index, typename... Senders>
struct _Receiver<CvrefReceiver, Index, Senders...>::type {
  using Receiver = std::decay_t<CvrefReceiver>;
  Operation<CvrefReceiver, Senders...>* op_state_;

  template <typename... As>
  friend void SetValue(type&& self, As&&... as) noexcept {
    std::get<Index>(self.op_state_->vals_).emplace((As &&) as...);
    self.op_state_->_Arrive();
  }
};

template <typename CvrefReceiver, typename... Senders>
struct _Operation<CvrefReceiver, Senders...>::type {
  using Receiver = std::decay_t<CvrefReceiver>;

  template <size_t Index>
  using _receiver_t = receiver_t<CvrefReceiver, Index, Senders...>;

  template <typename Sender, size_t Index>
  using _ChildOpState = connect_result_t<_copy_cvref_t<CvrefReceiver, Sender>, _receiver_t<Index>>;

  using _Indices = std::index_sequence_for<Senders...>;

  // workaround for a bug in GCC7 that `Is` in a lambda is treated as unexpanded parameter pack
  template <typename Sender, typename Receiver>
  static auto _Connect1(Sender&& sender, Receiver&& receiver) {
    return __conv{[&]() mutable { return Connect((Sender &&) sender, (Receiver &&) receiver); }};
  }

  template <size_t... Is, typename... _Senders>
  static auto _ConnectChildren(type* self, std::index_sequence<Is...>, _Senders&&... senders)
      -> std::tuple<_ChildOpState<Senders, Is>...> {
    return {_Connect1((_Senders &&) senders, _receiver_t<Is>{self})...};
  }

  using _ChildOpStates = decltype(_ConnectChildren(
      nullptr, _Indices{}, std::declval<_copy_cvref_t<CvrefReceiver, Senders>>()...));

  using _ChildValueTuple = std::tuple<std::optional<completion_signature_for_t<Senders>>...>;

  void _Arrive() noexcept {
    if (0 == --count_) {
      _Complete();
    }
  }

  void _Complete() noexcept {
    std::apply(
        [this](auto&... opt_vals) -> void {
          std::apply(
              [this](auto&... all_vals) -> void {
                SetValue((Receiver &&) receiver_, std::move(all_vals)...);
              },
              std::tuple_cat(
                  std::apply([](auto&... vals) { return std::tie(vals...); }, *opt_vals)...));
        },
        vals_);
  }

  template <typename... _Senders>
  explicit type(Receiver&& receiver, _Senders&&... senders)
      : child_states_{_ConnectChildren(this, _Indices{}, (_Senders &&) senders...)},
        receiver_(std::move(receiver)) {}

  friend void Start(type& self) noexcept {
    std::apply([](auto&&... child_ops) noexcept -> void { (Start(child_ops), ...); },
               self.child_states_);
  }

  type(const type&) = delete;
  type(type&&) = delete;
  type& operator=(const type&) = delete;
  type& operator=(type&&) = delete;

  _ChildOpStates child_states_;
  Receiver receiver_;
  std::atomic<size_t> count_{sizeof...(Senders)};
  _ChildValueTuple vals_;
};

template <typename... Senders>
struct _Sender {
  struct type;
};
template <typename... Senders>
using Sender = typename _Sender<std::decay_t<Senders>...>::type;

template <typename... Senders>
struct _Sender<Senders...>::type {
  using value_type = __concat_t<Senders...>;

  template <typename Receiver>
  using operation_t = Operation<Receiver, Senders...>;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver)
      -> operation_t<_copy_cvref_t<Self, std::decay_t<Receiver>>> {  // cvref encoded in receiver
                                                                     // type
    return std::apply(
        [&](auto&&... senders) {
          return operation_t<_copy_cvref_t<Self, std::decay_t<Receiver>>>(
              (Receiver &&) receiver, (decltype(senders)&&)senders...);
        },
        ((Self &&) self).senders_);
  }

  std::tuple<Senders...> senders_;
};

struct when_all_t {
  template <typename... Senders, std::enable_if_t<(_decays_to_sender<Senders> && ...), int> = 0>
  Sender<Senders...> operator()(Senders&&... senders) const {
    return {{(Senders &&) senders...}};
  }
};

}  // namespace __when_all

using __when_all::when_all_t;
inline constexpr when_all_t WhenAll{};

}

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_H_
