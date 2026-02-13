#pragma once

#include <boost/sml.hpp>
#include <array>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace emel {

namespace policy {

struct inline_scheduler {
  static constexpr bool guarantees_fifo = true;
  static constexpr bool single_consumer = true;
  static constexpr bool run_to_completion = true;

  template <class Fn>
  void schedule(Fn && fn) noexcept(noexcept(std::forward<Fn>(fn)())) {
    std::forward<Fn>(fn)();
  }
};

template <std::size_t Capacity = 1024, std::size_t InlineTaskBytes = 64>
class fifo_scheduler {
 public:
  static_assert(Capacity > 1, "fifo_scheduler capacity must be greater than 1");
  static_assert(
    (Capacity & (Capacity - 1)) == 0, "fifo_scheduler capacity must be a power of two");
  static_assert(InlineTaskBytes > 0, "fifo_scheduler inline storage must be non-zero");

  static constexpr bool guarantees_fifo = true;
  static constexpr bool single_consumer = true;
  static constexpr bool run_to_completion = true;

  fifo_scheduler() = default;
  ~fifo_scheduler() { clear(); }

  fifo_scheduler(const fifo_scheduler &) = delete;
  fifo_scheduler & operator=(const fifo_scheduler &) = delete;

  fifo_scheduler(fifo_scheduler &&) = delete;
  fifo_scheduler & operator=(fifo_scheduler &&) = delete;

  template <class Fn>
  bool try_run_immediate(Fn && fn) noexcept(noexcept(std::forward<Fn>(fn)())) {
    if (draining_ || !empty()) {
      return false;
    }

    draining_ = true;
    std::forward<Fn>(fn)();
    drain_pending();
    draining_ = false;
    return true;
  }

  template <class Fn>
  void schedule(Fn && fn) noexcept {
    if (try_run_immediate(std::forward<Fn>(fn))) {
      return;
    }

    enqueue(std::forward<Fn>(fn));
    if (draining_) {
      return;
    }

    draining_ = true;
    drain_pending();
    draining_ = false;
  }

 private:
  struct task_slot {
    using invoke_fn = void (*)(void *) noexcept;
    using destroy_fn = void (*)(void *) noexcept;

    alignas(std::max_align_t) std::array<std::byte, InlineTaskBytes> storage = {};
    invoke_fn invoke = nullptr;
    destroy_fn destroy = nullptr;

    template <class Fn>
    void set(Fn && fn) noexcept {
      using fn_type = std::decay_t<Fn>;
      static_assert(
        sizeof(fn_type) <= InlineTaskBytes, "Scheduled task exceeds inline storage capacity");
      static_assert(
        alignof(fn_type) <= alignof(std::max_align_t),
        "Scheduled task alignment exceeds scheduler storage alignment");

      new (storage.data()) fn_type(std::forward<Fn>(fn));
      invoke = [](void * ptr) noexcept { (*static_cast<fn_type *>(ptr))(); };
      destroy = [](void * ptr) noexcept { static_cast<fn_type *>(ptr)->~fn_type(); };
    }

    void run() noexcept {
      invoke(storage.data());
      destroy(storage.data());
      invoke = nullptr;
      destroy = nullptr;
    }

    void reset() noexcept {
      if (destroy != nullptr) {
        destroy(storage.data());
      }
      invoke = nullptr;
      destroy = nullptr;
    }
  };

  static constexpr std::size_t next(const std::size_t index) noexcept {
    return (index + 1) & (Capacity - 1);
  }

  bool empty() const noexcept { return head_ == tail_; }

  bool full() const noexcept { return next(tail_) == head_; }

  template <class Fn>
  void enqueue(Fn && fn) noexcept {
    if (full()) {
      std::terminate();
    }

    tasks_[tail_].set(std::forward<Fn>(fn));
    tail_ = next(tail_);
  }

  void drain_pending() noexcept {
    while (!empty()) {
      task_slot & task = tasks_[head_];
      head_ = next(head_);
      task.run();
    }
  }

  void clear() noexcept {
    while (!empty()) {
      tasks_[head_].reset();
      head_ = next(head_);
    }
    tail_ = head_;
    draining_ = false;
  }

  std::array<task_slot, Capacity> tasks_ = {};
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  bool draining_ = false;
};

template <class Scheduler>
struct coroutine_scheduler {
  using scheduler_type = Scheduler;
};

struct heap_coroutine_allocator {
  void * allocate(const std::size_t size, const std::size_t alignment) {
    return ::operator new(size, std::align_val_t(alignment));
  }

  void deallocate(
      void * ptr, const std::size_t size, const std::size_t alignment) noexcept {
    ::operator delete(ptr, size, std::align_val_t(alignment));
  }
};

template <std::size_t SlotSize = 1024, std::size_t SlotCount = 64>
class pooled_coroutine_allocator {
 public:
  static_assert(SlotSize > 0, "pooled_coroutine_allocator slot size must be non-zero");
  static_assert(SlotCount > 0, "pooled_coroutine_allocator slot count must be non-zero");

  pooled_coroutine_allocator() noexcept { reset_freelist(); }

  void * allocate(const std::size_t size, const std::size_t alignment) {
    if (size <= SlotSize && alignment <= alignof(pool_slot) && free_head_ != INVALID_INDEX) {
      const std::size_t slot_index = free_head_;
      free_head_ = next_free_[slot_index];
      return static_cast<void *>(slots_[slot_index].storage.data());
    }

    return ::operator new(size, std::align_val_t(alignment));
  }

  void deallocate(
      void * ptr, const std::size_t size, const std::size_t alignment) noexcept {
    if (ptr == nullptr) {
      return;
    }

    if (size <= SlotSize && alignment <= alignof(pool_slot) && is_pool_pointer(ptr)) {
      const std::size_t slot_index = slot_index_for(ptr);
      next_free_[slot_index] = free_head_;
      free_head_ = slot_index;
      return;
    }

    ::operator delete(ptr, size, std::align_val_t(alignment));
  }

 private:
  static constexpr std::size_t INVALID_INDEX = SlotCount;

  struct pool_slot {
    alignas(std::max_align_t) std::array<std::byte, SlotSize> storage = {};
  };

  bool is_pool_pointer(void * ptr) const noexcept {
    const auto * begin = reinterpret_cast<const std::byte *>(slots_.data());
    const auto * end = begin + sizeof(slots_);
    const auto * candidate = static_cast<const std::byte *>(ptr);
    if (candidate < begin || candidate >= end) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(candidate - begin);
    return (offset % sizeof(pool_slot)) == 0;
  }

  std::size_t slot_index_for(void * ptr) const noexcept {
    const auto * begin = reinterpret_cast<const std::byte *>(slots_.data());
    const auto * candidate = static_cast<const std::byte *>(ptr);
    const std::size_t offset = static_cast<std::size_t>(candidate - begin);
    return offset / sizeof(pool_slot);
  }

  void reset_freelist() noexcept {
    for (std::size_t i = 0; i + 1 < SlotCount; ++i) {
      next_free_[i] = i + 1;
    }
    next_free_[SlotCount - 1] = INVALID_INDEX;
    free_head_ = 0;
  }

  std::array<pool_slot, SlotCount> slots_ = {};
  std::array<std::size_t, SlotCount> next_free_ = {};
  std::size_t free_head_ = 0;
};

template <class Allocator>
struct coroutine_allocator {
  using allocator_type = Allocator;
};

template <class SchedulerPolicy>
concept valid_coroutine_scheduler_policy = requires {
  typename SchedulerPolicy::scheduler_type;
};

template <class Scheduler>
concept valid_coroutine_scheduler = requires(Scheduler scheduler, void (*fn)()) {
  scheduler.schedule(fn);
};

template <class Scheduler>
concept strict_ordering_scheduler_contract =
  requires {
    { Scheduler::guarantees_fifo } -> std::convertible_to<bool>;
    { Scheduler::single_consumer } -> std::convertible_to<bool>;
    { Scheduler::run_to_completion } -> std::convertible_to<bool>;
  } && static_cast<bool>(Scheduler::guarantees_fifo) &&
  static_cast<bool>(Scheduler::single_consumer) &&
  static_cast<bool>(Scheduler::run_to_completion);

template <class Scheduler>
concept has_try_run_immediate =
  requires(Scheduler scheduler) { { scheduler.try_run_immediate(+[]() noexcept {}) } -> std::same_as<bool>; };

template <class AllocatorPolicy>
concept valid_coroutine_allocator_policy = requires {
  typename AllocatorPolicy::allocator_type;
};

template <class Allocator>
concept valid_coroutine_allocator =
  requires(Allocator allocator, void * ptr, std::size_t size, std::size_t alignment) {
    { allocator.allocate(size, alignment) } -> std::same_as<void *>;
    { allocator.deallocate(ptr, size, alignment) } noexcept;
  };

}  // namespace policy

template <class Model, class... Policies>
class sm {
 public:
  using model_type = Model;
  using state_machine_type = boost::sml::sm<Model, Policies...>;

  sm() = default;
  ~sm() = default;

  sm(const sm &) = default;
  sm(sm &&) = default;
  sm & operator=(const sm &) = default;
  sm & operator=(sm &&) = default;

  template <class... Args>
  explicit sm(Args &&... args) : state_machine_(std::forward<Args>(args)...) {}

  template <class Event>
  bool process_event(const Event & ev) {
    return state_machine_.process_event(ev);
  }

  template <class State>
  bool is(State state = {}) const {
    return state_machine_.is(state);
  }

  template <class Visitor>
  void visit_current_states(Visitor && visitor) {
    state_machine_.visit_current_states(std::forward<Visitor>(visitor));
  }

 protected:
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  state_machine_type state_machine_;
};

class bool_task {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    using allocate_fn = void * (*)(void *, std::size_t, std::size_t);
    using deallocate_fn = void (*)(void *, void *, std::size_t, std::size_t) noexcept;

    struct frame_header {
      void * allocator_ctx = nullptr;
      deallocate_fn deallocate = nullptr;
      void * allocation_ptr = nullptr;
      std::size_t allocation_size = 0;
      std::size_t allocation_alignment = 0;
    };

    static void * allocate_frame(
        const std::size_t frame_size,
        void * allocator_ctx,
        allocate_fn allocate,
        deallocate_fn deallocate) {
      constexpr std::size_t frame_align = alignof(promise_type);
      constexpr std::size_t header_align = alignof(frame_header);
      constexpr std::size_t alloc_align = frame_align > header_align ? frame_align : header_align;

      const std::size_t allocation_size = frame_size + sizeof(frame_header) + alloc_align - 1;
      void * raw = allocate(allocator_ctx, allocation_size, alloc_align);
      if (raw == nullptr) {
        throw std::bad_alloc();
      }

      void * aligned_frame = static_cast<std::byte *>(raw) + sizeof(frame_header);
      std::size_t space = allocation_size - sizeof(frame_header);
      if (std::align(frame_align, frame_size, aligned_frame, space) == nullptr) {
        deallocate(allocator_ctx, raw, allocation_size, alloc_align);
        throw std::bad_alloc();
      }

      auto * header = reinterpret_cast<frame_header *>(
        static_cast<std::byte *>(aligned_frame) - sizeof(frame_header));
      header->allocator_ctx = allocator_ctx;
      header->deallocate = deallocate;
      header->allocation_ptr = raw;
      header->allocation_size = allocation_size;
      header->allocation_alignment = alloc_align;
      return aligned_frame;
    }

    template <class Allocator>
    static void * allocate_frame_with_allocator(std::size_t frame_size, Allocator & allocator) {
      static_assert(
        policy::valid_coroutine_allocator<Allocator>,
        "Coroutine allocator must provide allocate(size, alignment) and "
        "noexcept deallocate(ptr, size, alignment)");

      return allocate_frame(
        frame_size,
        &allocator,
        [](void * ctx, const std::size_t size, const std::size_t alignment) -> void * {
          return static_cast<Allocator *>(ctx)->allocate(size, alignment);
        },
        [](void * ctx, void * ptr, const std::size_t size, const std::size_t alignment) noexcept {
          static_cast<Allocator *>(ctx)->deallocate(ptr, size, alignment);
        });
    }

    static void deallocate_frame(void * frame_ptr) noexcept {
      if (frame_ptr == nullptr) {
        return;
      }

      auto * header = reinterpret_cast<frame_header *>(
        static_cast<std::byte *>(frame_ptr) - sizeof(frame_header));
      header->deallocate(
        header->allocator_ctx,
        header->allocation_ptr,
        header->allocation_size,
        header->allocation_alignment);
    }

    static void * operator new(std::size_t frame_size) {
      return allocate_frame(
        frame_size,
        nullptr,
        [](void *, const std::size_t size, const std::size_t alignment) -> void * {
          return ::operator new(size, std::align_val_t(alignment));
        },
        [](void *, void * ptr, const std::size_t size, const std::size_t alignment) noexcept {
          ::operator delete(ptr, size, std::align_val_t(alignment));
        });
    }

    template <class Allocator, class... Args>
    static void * operator new(
        std::size_t frame_size,
        std::allocator_arg_t,
        Allocator & allocator,
        Args &&...) {
      return allocate_frame_with_allocator(frame_size, allocator);
    }

    static void operator delete(void * frame_ptr) noexcept { deallocate_frame(frame_ptr); }

    static void operator delete(void * frame_ptr, std::size_t) noexcept {
      deallocate_frame(frame_ptr);
    }

    template <class Allocator, class... Args>
    static void operator delete(
        void * frame_ptr,
        std::allocator_arg_t,
        Allocator &,
        Args &&...) noexcept {
      deallocate_frame(frame_ptr);
    }

    bool value = false;
    std::exception_ptr exception = nullptr;
    std::coroutine_handle<> continuation = std::noop_coroutine();

    bool_task get_return_object() noexcept { return bool_task{handle_type::from_promise(*this)}; }

    std::suspend_never initial_suspend() noexcept { return {}; }

    auto final_suspend() noexcept {
      struct continuation_awaiter {
        bool await_ready() const noexcept { return false; }
        std::coroutine_handle<> await_suspend(handle_type h) const noexcept {
          return h.promise().continuation;
        }
        void await_resume() const noexcept {}
      };
      return continuation_awaiter{};
    }

    void return_value(const bool v) noexcept { value = v; }
    void unhandled_exception() noexcept { exception = std::current_exception(); }
  };

  bool_task() = default;
  explicit bool_task(handle_type handle) noexcept : handle_(handle) {}

  static bool_task from_value(const bool value) noexcept {
    bool_task task{};
    task.has_immediate_value_ = true;
    task.immediate_value_ = value;
    return task;
  }

  ~bool_task() {
    if (handle_) {
      handle_.destroy();
    }
  }

  bool_task(const bool_task &) = delete;
  bool_task & operator=(const bool_task &) = delete;

  bool_task(bool_task && other) noexcept
      : handle_(std::exchange(other.handle_, {})),
        has_immediate_value_(other.has_immediate_value_),
        immediate_value_(other.immediate_value_) {
    other.has_immediate_value_ = false;
    other.immediate_value_ = false;
  }

  bool_task & operator=(bool_task && other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (handle_) {
      handle_.destroy();
    }
    handle_ = std::exchange(other.handle_, {});
    has_immediate_value_ = other.has_immediate_value_;
    immediate_value_ = other.immediate_value_;
    other.has_immediate_value_ = false;
    other.immediate_value_ = false;
    return *this;
  }

  bool await_ready() const noexcept {
    if (has_immediate_value_) {
      return true;
    }
    return !handle_ || handle_.done();
  }

  std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
    if (!handle_) {
      return std::noop_coroutine();
    }
    handle_.promise().continuation = awaiting;
    return handle_;
  }

  bool await_resume() { return result(); }

  bool result() {
    if (has_immediate_value_) {
      return immediate_value_;
    }
    if (!handle_) {
      return false;
    }
    if (!handle_.done()) {
      throw std::logic_error("bool_task result() called before coroutine completion");
    }
    if (handle_.promise().exception) {
      std::rethrow_exception(handle_.promise().exception);
    }
    return handle_.promise().value;
  }

 private:
  handle_type handle_{};
  bool has_immediate_value_ = false;
  bool immediate_value_ = false;
};

template <
    class Model,
    class SchedulerPolicy = policy::coroutine_scheduler<policy::fifo_scheduler<>>,
    class AllocatorPolicy = policy::coroutine_allocator<policy::pooled_coroutine_allocator<>>,
    class... Policies>
class co_sm {
 public:
  static_assert(
    policy::valid_coroutine_scheduler_policy<SchedulerPolicy>,
    "SchedulerPolicy must define scheduler_type");
  static_assert(
    policy::valid_coroutine_allocator_policy<AllocatorPolicy>,
    "AllocatorPolicy must define allocator_type");

  using model_type = Model;
  using scheduler_policy_type = SchedulerPolicy;
  using scheduler_type = typename scheduler_policy_type::scheduler_type;
  using allocator_policy_type = AllocatorPolicy;
  using allocator_type = typename allocator_policy_type::allocator_type;
  using state_machine_type = boost::sml::sm<Model, Policies...>;

  static_assert(
    policy::valid_coroutine_scheduler<scheduler_type>,
    "scheduler_type must provide schedule(Fn)");
  static_assert(
    policy::strict_ordering_scheduler_contract<scheduler_type>,
    "scheduler_type must guarantee FIFO ordering, single-consumer dispatch, and run-to-completion");
  static_assert(
    policy::valid_coroutine_allocator<allocator_type>,
    "allocator_type must provide allocate(size, alignment) and "
    "noexcept deallocate(ptr, size, alignment)");

  co_sm() = default;
  ~co_sm() = default;

  co_sm(const co_sm &) = default;
  co_sm(co_sm &&) = default;
  co_sm & operator=(const co_sm &) = default;
  co_sm & operator=(co_sm &&) = default;

  explicit co_sm(const scheduler_type & scheduler) : scheduler_(scheduler) {}
  explicit co_sm(const allocator_type & allocator) : allocator_(allocator) {}
  co_sm(const scheduler_type & scheduler, const allocator_type & allocator)
      : scheduler_(scheduler), allocator_(allocator) {}

  template <class... Args>
  explicit co_sm(Args &&... args) : state_machine_(std::forward<Args>(args)...) {}

  template <class... Args>
  co_sm(const scheduler_type & scheduler, Args &&... args)
      : state_machine_(std::forward<Args>(args)...), scheduler_(scheduler) {}

  template <class... Args>
  co_sm(const allocator_type & allocator, Args &&... args)
      : state_machine_(std::forward<Args>(args)...), allocator_(allocator) {}

  template <class... Args>
  co_sm(const scheduler_type & scheduler, const allocator_type & allocator, Args &&... args)
      : state_machine_(std::forward<Args>(args)...), scheduler_(scheduler), allocator_(allocator) {}

  template <class Event>
  bool process_event(const Event & ev) {
    return state_machine_.process_event(ev);
  }

  template <class Event>
  bool_task process_event_async(const Event & ev) {
    if constexpr (std::is_same_v<scheduler_type, policy::inline_scheduler>) {
      return bool_task::from_value(state_machine_.process_event(ev));
    }
    if constexpr (policy::has_try_run_immediate<scheduler_type>) {
      bool accepted = false;
      if (scheduler_.try_run_immediate([this, &ev, &accepted]() {
            accepted = state_machine_.process_event(ev);
          })) {
        return bool_task::from_value(accepted);
      }
    }
    return process_event_async_impl(std::allocator_arg, allocator_, *this, ev);
  }

  template <class State>
  bool is(State state = {}) const {
    return state_machine_.is(state);
  }

  template <class Visitor>
  void visit_current_states(Visitor && visitor) {
    state_machine_.visit_current_states(std::forward<Visitor>(visitor));
  }

  scheduler_type & scheduler() noexcept { return scheduler_; }
  const scheduler_type & scheduler() const noexcept { return scheduler_; }
  allocator_type & allocator() noexcept { return allocator_; }
  const allocator_type & allocator() const noexcept { return allocator_; }

 protected:
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  template <class Event>
  static bool_task process_event_async_impl(
      std::allocator_arg_t, allocator_type & allocator, co_sm & self, const Event & ev) {
    (void)allocator;
    co_return co_await process_event_awaitable<Event>{self, ev};
  }

  template <class Event>
  struct process_event_awaitable {
    co_sm & self;
    const Event & event;
    bool accepted = false;

    bool await_ready() noexcept {
      if constexpr (std::is_same_v<scheduler_type, policy::inline_scheduler>) {
        accepted = self.state_machine_.process_event(event);
        return true;
      }
      return false;
    }

    void await_suspend(std::coroutine_handle<> handle) {
      self.scheduler_.schedule([this, handle]() mutable {
        accepted = self.state_machine_.process_event(event);
        handle.resume();
      });
    }

    bool await_resume() const noexcept { return accepted; }
  };

  state_machine_type state_machine_;
  scheduler_type scheduler_{};
  allocator_type allocator_{};
};

}  // namespace emel
