//
// Created by li on 2022/3/17.
//

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_

#include "core/value.h"
#include "execution.h"

namespace mmdeploy {

class AbstractScheduler;
class AbstractSender;
class AbstractOperation;
class AbstractReceiver;

template <class _Scheduler>
class TypeErasedScheduler;

template <class _Sender>
class TypeErasedSender;

template <class _Operation>
class TypeErasedOperation;

template <class _Receiver>
class TypeErasedReceiver;

template <class _Sender, class T = std::remove_reference_t<_Sender>>
TypeErasedSender<T>* MakeTypeErasedSender(_Sender&&);

template <class _Scheduler, class T = std::remove_reference_t<_Scheduler>>
TypeErasedScheduler<T>* MakeTypeErasedScheduler(_Scheduler&&);

template <class _Receiver, class T = std::remove_reference_t<_Receiver>>
TypeErasedReceiver<T>* MakeTypeErasedReceiver(_Receiver&&);

// eliminate recursive of type erasers
inline AbstractSender* MakeTypeErasedSender(AbstractSender* s) { return s; }
inline AbstractScheduler* MakeTypeErasedScheduler(AbstractScheduler* s) { return s; }
inline AbstractReceiver* MakeTypeErasedReceiver(AbstractReceiver* r) { return r; }

class AbstractScheduler {
 public:
  virtual ~AbstractScheduler() = default;

  virtual AbstractSender* _Schedule() = 0;
  //  virtual Value _SyncWait(AbstractSender*) = 0;

  friend AbstractSender* Schedule(AbstractScheduler* self) { return self->_Schedule(); }

  //  friend Value SyncWait(AbstractScheduler* self, AbstractSender* sender) {
  //    return self->_SyncWait(sender);
  //  }
};

class AbstractSender {
 public:
  virtual ~AbstractSender() = default;

  virtual AbstractOperation* _Connect(AbstractReceiver* r) = 0;
  virtual AbstractScheduler* _GetCompletionScheduler() = 0;

  template <class R>
  friend AbstractOperation* Connect(AbstractSender* self, R rcvr) {
    return self->_Connect(MakeTypeErasedReceiver((R &&) rcvr));
  }
  friend AbstractScheduler* GetCompletionScheduler(AbstractSender* self) {
    return self->_GetCompletionScheduler();
  }
  //  friend Value SyncWait(AbstractSender* self) {
  //    if (auto sched = self->_GetCompletionScheduler(); sched != nullptr) {
  //      return SyncWait(sched, self);
  //    } else {
  //      return std::get<Value>(_SyncWaitDefault(self));
  //    }
  //  }
};

class AbstractOperation {
 public:
  virtual ~AbstractOperation() = default;

  virtual void _Start() = 0;

  friend void Start(AbstractOperation* self) { self->_Start(); }
};

class AbstractReceiver {
 public:
  virtual ~AbstractReceiver() = default;

  virtual void _SetValue(Value) = 0;

  friend void SetValue(AbstractReceiver* self, Value v) { self->_SetValue(std::move(v)); }

  friend void SetValue(AbstractReceiver* self) { self->_SetValue(Value::kNull); }
};

template <class _Scheduler>
class TypeErasedScheduler : public AbstractScheduler {
 public:
  explicit TypeErasedScheduler(_Scheduler&& scheduler) : scheduler_(std::move(scheduler)) {}
  template <class S>
  explicit TypeErasedScheduler(TypeErasedScheduler<S>*) = delete;
  explicit TypeErasedScheduler(AbstractScheduler*) = delete;

  AbstractSender* _Schedule() override { return MakeTypeErasedSender(Schedule(scheduler_)); }

  //  Value _SyncWait(AbstractSender* sender) override { return SyncWait(scheduler_, sender); }

 private:
  _Scheduler scheduler_;
};

template <class _Sender>
class TypeErasedSender : public AbstractSender {
 public:
  explicit TypeErasedSender(_Sender&& s) : s_(std::move(s)) {}

  template <class S>
  explicit TypeErasedSender(TypeErasedSender<S>*) = delete;
  explicit TypeErasedSender(AbstractSender*) = delete;

  AbstractOperation* _Connect(AbstractReceiver* r) override {
    // most operation states are non-movable, use copy elision to initialize erased operations
    using _Operation = decltype(Connect(s_, r));
    return new TypeErasedOperation<_Operation>([&] { return Connect(s_, r); });
  }

  AbstractScheduler* _GetCompletionScheduler() override {
    if constexpr (_has_completion_scheduler<_Sender>) {
      auto sched = GetCompletionScheduler(s_);
      return MakeTypeErasedScheduler(std::move(sched));
    } else {
      return nullptr;
    }
  }

 private:
  _Sender s_;
};

template <class _Operation>
class TypeErasedOperation : public AbstractOperation {
 public:
  template <class F>
  explicit TypeErasedOperation(F f) : operation_(f()) {}

  template <class T>
  explicit TypeErasedOperation(TypeErasedOperation<T>*) = delete;
  explicit TypeErasedOperation(AbstractOperation*) = delete;

  void _Start() override { Start(operation_); }

 private:
  _Operation operation_;
};

template <class _Receiver>
class TypeErasedReceiver : public AbstractReceiver {
 public:
  explicit TypeErasedReceiver(_Receiver&& r) : r_(std::move(r)) {}

  template <class R>
  explicit TypeErasedReceiver(TypeErasedReceiver<R>*) = delete;
  explicit TypeErasedReceiver(AbstractReceiver*) = delete;

  void _SetValue(Value v) override {
    if constexpr (detail::is_detected_v<_set_value_t, _Receiver>) {
      SetValue(std::move(r_), std::move(v));
    } else {
      SetValue(std::move(r_));
    }
  }

 private:
  template <class T>
  using _set_value_t = decltype(SetValue(std::declval<T>(), std::declval<Value>()));

  _Receiver r_;
};

template <class _Sender, class T>
TypeErasedSender<T>* MakeTypeErasedSender(_Sender&& sender) {
  return new TypeErasedSender<T>{(_Sender &&) sender};
}

template <class _Scheduler, class T>
TypeErasedScheduler<T>* MakeTypeErasedScheduler(_Scheduler&& scheduler) {
  return new TypeErasedScheduler<T>{(_Scheduler &&) scheduler};
}

template <class _Receiver, class T>
TypeErasedReceiver<T>* MakeTypeErasedReceiver(_Receiver&& receiver) {
  return new TypeErasedReceiver<T>{(_Receiver &&) receiver};
}

}  // namespace mmdeploy

#if __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_value* mmdeploy_value_t;
typedef mmdeploy_value_t (*mmdeploy_invocable_t)(mmdeploy_value_t, void*);

struct mmdeploy_sender;
struct mmdeploy_scheduler;

typedef mmdeploy_sender (*mmdeploy_kleisli_t)(mmdeploy_value_t, void*);

typedef struct mmdeploy_sender* mmdeploy_sender_t;
typedef struct mmdeploy_scheduler* mmdeploy_scheduler_t;

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_inline_scheduler();

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                                          mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input,
                                                      mmdeploy_invocable_t fn, void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_let_value(mmdeploy_sender_t input,
                                                           mmdeploy_kleisli_t kleisli,
                                                           void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n);

MMDEPLOY_API mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
