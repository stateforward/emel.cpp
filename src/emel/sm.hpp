#pragma once

#include <boost/sml.hpp>
#include <algorithm>
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
#include <tuple>
#include <utility>

namespace emel {

namespace detail {

template <class event>
constexpr bool normalize_event_result(const event & ev, const bool accepted) noexcept {
  if (!accepted) {
    return false;
  }
  if constexpr (requires { ev.error_out; }) {
    if (ev.error_out != nullptr && *ev.error_out != 0) {
      return false;
    }
  }
  return true;
}

template <class owner, class process>
struct process_support {
  struct immediate_queue {
    using container_type = void;
    owner * owner_ptr = nullptr;

    template <class event>
    void push(const event & ev) noexcept {
      if (owner_ptr != nullptr) {
        owner_ptr->process_event(ev);
      }
    }
  };

  explicit process_support(owner * owner_ptr) : queue_{owner_ptr}, process_{queue_} {}

  immediate_queue queue_{};
  process process_;
};

template <class... types, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<types...>, fn && visitor) {
  (visitor.template operator()<types>(), ...);
}

template <class list>
struct type_list_size;

template <class... types>
struct type_list_size<boost::sml::aux::type_list<types...>>
    : std::integral_constant<std::size_t, sizeof...(types)> {};

template <class list>
inline constexpr std::size_t type_list_size_v = type_list_size<list>::value;

template <class list>
struct type_list_max;

template <class... types>
struct type_list_max<boost::sml::aux::type_list<types...>> {
  static constexpr std::size_t size = std::max({sizeof(types)...});
  static constexpr std::size_t align = std::max({alignof(types)...});
};

template <class type, class list>
struct type_list_contains;

template <class type, class... types>
struct type_list_contains<type, boost::sml::aux::type_list<types...>>
    : std::bool_constant<(std::is_same_v<type, types> || ...)> {};

template <class sm>
sm * sm_any_ptr(void * storage) noexcept {
  return std::launder(reinterpret_cast<sm *>(storage));
}

template <class sm>
const sm * sm_any_ptr(const void * storage) noexcept {
  return std::launder(reinterpret_cast<const sm *>(storage));
}

template <class sm>
void sm_any_construct(void * storage) {
  new (storage) sm();
}

template <class sm>
void sm_any_destroy(void * storage) noexcept {
  sm_any_ptr<sm>(storage)->~sm();
}

template <class sm, class event>
bool sm_any_process_event(void * storage, const event & ev) {
  return sm_any_ptr<sm>(storage)->process_event(ev);
}

template <class list>
struct sm_any_construct_table;

template <class... types>
struct sm_any_construct_table<boost::sml::aux::type_list<types...>> {
  using fn = void (*)(void *);
  inline static constexpr std::array<fn, sizeof...(types)> table = {
      &sm_any_construct<types>...};
};

template <class list>
struct sm_any_destroy_table;

template <class... types>
struct sm_any_destroy_table<boost::sml::aux::type_list<types...>> {
  using fn = void (*)(void *) noexcept;
  inline static constexpr std::array<fn, sizeof...(types)> table = {
      &sm_any_destroy<types>...};
};

template <class event, class list>
struct sm_any_event_table;

template <class event, class... types>
struct sm_any_event_table<event, boost::sml::aux::type_list<types...>> {
  using fn = bool (*)(void *, const event &);
  inline static constexpr std::array<fn, sizeof...(types)> table = {
      &sm_any_process_event<types, event>...};
};

template <class event_list>
struct sm_any_process_event_table;

template <class... events>
struct sm_any_process_event_table<boost::sml::aux::type_list<events...>> {
  template <class event>
  using fn = bool (*)(void *, const event &);

  std::tuple<fn<events>...> fns{};

  template <class event>
  void set(fn<event> fn_value) noexcept {
    std::get<fn<event>>(fns) = fn_value;
  }

  template <class event>
  bool call(void * storage, const event & ev) const {
    return std::get<fn<event>>(fns)(storage, ev);
  }
};

template <class list>
struct sm_any_visit;

template <class... types>
struct sm_any_visit<boost::sml::aux::type_list<types...>> {
  template <std::size_t idx, class visitor, class first, class... rest>
  static void apply_index(std::size_t target, void * storage, visitor && visitor_fn) {
    if (target == idx) {
      visitor_fn(*sm_any_ptr<first>(storage));
      return;
    }
    if constexpr (sizeof...(rest) > 0) {
      apply_index<idx + 1, visitor, rest...>(
          target, storage, std::forward<visitor>(visitor_fn));
    }
  }

  template <std::size_t idx, class visitor, class first, class... rest>
  static void apply_index(std::size_t target, const void * storage,
                          visitor && visitor_fn) {
    if (target == idx) {
      visitor_fn(*sm_any_ptr<first>(storage));
      return;
    }
    if constexpr (sizeof...(rest) > 0) {
      apply_index<idx + 1, visitor, rest...>(
          target, storage, std::forward<visitor>(visitor_fn));
    }
  }

  template <class visitor>
  static void apply(std::size_t target, void * storage, visitor && visitor_fn) {
    apply_index<0, visitor, types...>(target, storage,
                                      std::forward<visitor>(visitor_fn));
  }

  template <class visitor>
  static void apply(std::size_t target, const void * storage,
                    visitor && visitor_fn) {
    apply_index<0, visitor, types...>(target, storage,
                                      std::forward<visitor>(visitor_fn));
  }
};

}  // namespace detail

namespace policy {

struct inline_scheduler {
  static constexpr bool guarantees_fifo = true;
  static constexpr bool single_consumer = true;
  static constexpr bool run_to_completion = true;

  template <class fn>
  void schedule(fn && fn_value) noexcept(noexcept(std::forward<fn>(fn_value)())) {
    std::forward<fn>(fn_value)();
  }
};

template <std::size_t capacity = 1024, std::size_t inline_task_bytes = 64>
class fifo_scheduler {
 public:
  static_assert(capacity > 1, "fifo_scheduler capacity must be greater than 1");
  static_assert(
    (capacity & (capacity - 1)) == 0, "fifo_scheduler capacity must be a power of two");
  static_assert(inline_task_bytes > 0, "fifo_scheduler inline storage must be non-zero");

  static constexpr bool guarantees_fifo = true;
  static constexpr bool single_consumer = true;
  static constexpr bool run_to_completion = true;

  fifo_scheduler() = default;
  ~fifo_scheduler() { clear(); }

  fifo_scheduler(const fifo_scheduler &) = delete;
  fifo_scheduler & operator=(const fifo_scheduler &) = delete;

  fifo_scheduler(fifo_scheduler &&) = delete;
  fifo_scheduler & operator=(fifo_scheduler &&) = delete;

  template <class fn>
  bool try_run_immediate(fn && fn_value) noexcept(noexcept(std::forward<fn>(fn_value)())) {
    if (draining_ || !empty()) {
      return false;
    }

    draining_ = true;
    std::forward<fn>(fn_value)();
    drain_pending();
    draining_ = false;
    return true;
  }

  template <class fn>
  void schedule(fn && fn_value) noexcept {
    if (try_run_immediate(std::forward<fn>(fn_value))) {
      return;
    }

    enqueue(std::forward<fn>(fn_value));
    if (draining_) {
      return;
    }

    draining_ = true;
    drain_pending();
    draining_ = false;
  }

 private:
  // GCOVR_EXCL_START
  struct task_slot {
    using invoke_fn = void (*)(void *) noexcept;
    using destroy_fn = void (*)(void *) noexcept;

    alignas(std::max_align_t) std::array<std::byte, inline_task_bytes> storage = {};
    invoke_fn invoke = nullptr;
    destroy_fn destroy = nullptr;

    template <class fn>
    void set(fn && fn_value) noexcept {
      using fn_type = std::decay_t<fn>;
      static_assert(
        sizeof(fn_type) <= inline_task_bytes, "scheduled task exceeds inline storage capacity");
      static_assert(
        alignof(fn_type) <= alignof(std::max_align_t),
        "scheduled task alignment exceeds scheduler storage alignment");

      new (storage.data()) fn_type(std::forward<fn>(fn_value));
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
  // GCOVR_EXCL_STOP

  static constexpr std::size_t next(const std::size_t index) noexcept {
    return (index + 1) & (capacity - 1);
  }

  bool empty() const noexcept { return head_ == tail_; }

  bool full() const noexcept { return next(tail_) == head_; }

  template <class fn>
  void enqueue(fn && fn_value) noexcept {
    if (full()) {
      std::terminate();  // GCOVR_EXCL_LINE
    }

    tasks_[tail_].set(std::forward<fn>(fn_value));
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

  std::array<task_slot, capacity> tasks_ = {};
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  bool draining_ = false;
};

template <class scheduler>
struct coroutine_scheduler {
  using scheduler_type = scheduler;
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

template <std::size_t slot_size = 1024, std::size_t slot_count = 64>
class pooled_coroutine_allocator {
 public:
  static_assert(slot_size > 0, "pooled_coroutine_allocator slot size must be non-zero");
  static_assert(slot_count > 0, "pooled_coroutine_allocator slot count must be non-zero");

  pooled_coroutine_allocator() noexcept { reset_freelist(); }

  void * allocate(const std::size_t size, const std::size_t alignment) {
    if (size <= slot_size && alignment <= alignof(pool_slot) && free_head_ != INVALID_INDEX) {
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

    if (size <= slot_size && alignment <= alignof(pool_slot) && is_pool_pointer(ptr)) {
      const std::size_t slot_index = slot_index_for(ptr);
      next_free_[slot_index] = free_head_;
      free_head_ = slot_index;
      return;
    }

    ::operator delete(ptr, size, std::align_val_t(alignment));
  }

 private:
  static constexpr std::size_t INVALID_INDEX = slot_count;

  struct pool_slot {
    alignas(std::max_align_t) std::array<std::byte, slot_size> storage = {};
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
    for (std::size_t i = 0; i + 1 < slot_count; ++i) {
      next_free_[i] = i + 1;
    }
    next_free_[slot_count - 1] = INVALID_INDEX;
    free_head_ = 0;
  }

  std::array<pool_slot, slot_count> slots_ = {};
  std::array<std::size_t, slot_count> next_free_ = {};
  std::size_t free_head_ = 0;
};

template <class allocator>
struct coroutine_allocator {
  using allocator_type = allocator;
};

template <class scheduler_policy>
concept valid_coroutine_scheduler_policy = requires {
  typename scheduler_policy::scheduler_type;
};

template <class scheduler>
concept valid_coroutine_scheduler = requires(scheduler scheduler_value, void (*fn_ptr)()) {
  scheduler_value.schedule(fn_ptr);
};

template <class scheduler>
concept strict_ordering_scheduler_contract =
  requires {
    { scheduler::guarantees_fifo } -> std::convertible_to<bool>;
    { scheduler::single_consumer } -> std::convertible_to<bool>;
    { scheduler::run_to_completion } -> std::convertible_to<bool>;
  } && static_cast<bool>(scheduler::guarantees_fifo) &&
  static_cast<bool>(scheduler::single_consumer) &&
  static_cast<bool>(scheduler::run_to_completion);

template <class scheduler>
concept has_try_run_immediate =
  requires(scheduler scheduler_value) {
    { scheduler_value.try_run_immediate(+[]() noexcept {}) } -> std::same_as<bool>;
  };

template <class allocator_policy>
concept valid_coroutine_allocator_policy = requires {
  typename allocator_policy::allocator_type;
};

template <class allocator>
concept valid_coroutine_allocator =
  requires(allocator allocator_value, void * ptr, std::size_t size, std::size_t alignment) {
    { allocator_value.allocate(size, alignment) } -> std::same_as<void *>;
    { allocator_value.deallocate(ptr, size, alignment) } noexcept;
  };

}  // namespace policy

template <class model, class... policies>
class sm {
 public:
  using model_type = model;
  using state_machine_type = boost::sml::sm<model, policies...>;

  sm() = default;
  ~sm() = default;

  sm(const sm &) = default;
  sm(sm &&) = default;
  sm & operator=(const sm &) = default;
  sm & operator=(sm &&) = default;

  template <class... args>
  explicit sm(args &&... args_in) : state_machine_(std::forward<args>(args_in)...) {}

  template <class event>
  bool process_event(const event & ev) {
    const bool accepted = state_machine_.process_event(ev);
    return detail::normalize_event_result(ev, accepted);
  }

  template <class state>
  bool is(state state_value = {}) const {
    return state_machine_.is(state_value);
  }

  template <class visitor>
  void visit_current_states(visitor && visitor_fn) {
    state_machine_.visit_current_states(std::forward<visitor>(visitor_fn));
  }

 protected:
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  state_machine_type state_machine_;
};

template <class kind_enum, class sm_list, class event_list>
class sm_any {
 public:
  sm_any() { construct(default_index()); }
  explicit sm_any(const kind_enum kind) { construct(index_from_kind(kind)); }

  sm_any(const sm_any &) = delete;
  sm_any & operator=(const sm_any &) = delete;
  sm_any(sm_any &&) = delete;
  sm_any & operator=(sm_any &&) = delete;

  ~sm_any() { destroy(); }

  void set_kind(const kind_enum kind) {
    const std::size_t next = index_from_kind(kind);
    if (next == index_) {
      return;
    }
    destroy();
    construct(next);
  }

  kind_enum kind() const noexcept { return kind_; }

  template <class event>
  bool process_event(const event & ev) {
    static_assert(detail::type_list_contains<event, event_list>::value,
                  "sm_any event type must appear in event_list");
    return process_.template call<event>(storage(), ev);
  }

  template <class visitor>
  void visit(visitor && visitor_fn) {
    detail::sm_any_visit<sm_list>::apply(index_, storage(),
                                         std::forward<visitor>(visitor_fn));
  }

  template <class visitor>
  void visit(visitor && visitor_fn) const {
    detail::sm_any_visit<sm_list>::apply(index_, storage_const(),
                                         std::forward<visitor>(visitor_fn));
  }

 private:
  using storage_t = std::aligned_storage_t<detail::type_list_max<sm_list>::size,
                                          detail::type_list_max<sm_list>::align>;

  static constexpr std::size_t k_sm_count = detail::type_list_size_v<sm_list>;
  static_assert(k_sm_count > 0, "sm_any requires at least one state machine");

  using construct_fn = void (*)(void *);
  using destroy_fn = void (*)(void *) noexcept;

  static constexpr std::size_t default_index() noexcept { return 0; }

  static constexpr std::size_t index_from_kind(const kind_enum kind) noexcept {
    const std::size_t idx = static_cast<std::size_t>(kind);
    return idx < k_sm_count ? idx : default_index();
  }

  void construct(const std::size_t idx) {
    detail::sm_any_construct_table<sm_list>::table[idx](storage());
    detail::for_each_type(event_list{}, process_setter{this, idx});
    index_ = idx;
    kind_ = static_cast<kind_enum>(idx);
  }

  void destroy() noexcept {
    detail::sm_any_destroy_table<sm_list>::table[index_](storage());
  }

  void * storage() noexcept { return &storage_; }
  const void * storage_const() const noexcept { return &storage_; }

  storage_t storage_{};
  std::size_t index_ = 0;
  kind_enum kind_ = static_cast<kind_enum>(0);
  detail::sm_any_process_event_table<event_list> process_{};

  struct process_setter {
    sm_any * self = nullptr;
    std::size_t index = 0;

    template <class event>
    void operator()() const {
      self->process_.template set<event>(
          detail::sm_any_event_table<event, sm_list>::table[index]);
    }
  };
};

template <class kind_enum, class sm_list, class event_list>
using SmAny = sm_any<kind_enum, sm_list, event_list>;

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

    template <class allocator>
    static void * allocate_frame_with_allocator(
        std::size_t frame_size,
        allocator & allocator_value) {
      static_assert(
        policy::valid_coroutine_allocator<allocator>,
        "coroutine allocator must provide allocate(size, alignment) and "
        "noexcept deallocate(ptr, size, alignment)");

      return allocate_frame(
        frame_size,
        &allocator_value,
        [](void * ctx, const std::size_t size, const std::size_t alignment) -> void * {
          return static_cast<allocator *>(ctx)->allocate(size, alignment);
        },
        [](void * ctx, void * ptr, const std::size_t size, const std::size_t alignment) noexcept {
          static_cast<allocator *>(ctx)->deallocate(ptr, size, alignment);
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

    template <class allocator, class... args>
    static void * operator new(
        std::size_t frame_size,
        std::allocator_arg_t,
        allocator & allocator_value,
        args &&...) {
      return allocate_frame_with_allocator(frame_size, allocator_value);
    }

    static void operator delete(void * frame_ptr) noexcept { deallocate_frame(frame_ptr); }

    static void operator delete(void * frame_ptr, std::size_t) noexcept {
      deallocate_frame(frame_ptr);
    }

    template <class allocator, class... args>
    static void operator delete(
        void * frame_ptr,
        std::allocator_arg_t,
        allocator &,
        args &&...) noexcept {
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
    class model,
    class scheduler_policy = policy::coroutine_scheduler<policy::fifo_scheduler<>>,
    class allocator_policy = policy::coroutine_allocator<policy::pooled_coroutine_allocator<>>,
    class... policies>
class co_sm {
 public:
  static_assert(
    policy::valid_coroutine_scheduler_policy<scheduler_policy>,
    "scheduler_policy must define scheduler_type");
  static_assert(
    policy::valid_coroutine_allocator_policy<allocator_policy>,
    "allocator_policy must define allocator_type");

  using model_type = model;
  using scheduler_policy_type = scheduler_policy;
  using scheduler_type = typename scheduler_policy_type::scheduler_type;
  using allocator_policy_type = allocator_policy;
  using allocator_type = typename allocator_policy_type::allocator_type;
  using state_machine_type = boost::sml::sm<model, policies...>;

  static_assert(
    policy::valid_coroutine_scheduler<scheduler_type>,
    "scheduler_type must provide schedule(fn)");
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

  template <class... args>
  explicit co_sm(args &&... args_in) : state_machine_(std::forward<args>(args_in)...) {}

  template <class... args>
  co_sm(const scheduler_type & scheduler, args &&... args_in)
      : state_machine_(std::forward<args>(args_in)...), scheduler_(scheduler) {}

  template <class... args>
  co_sm(const allocator_type & allocator, args &&... args_in)
      : state_machine_(std::forward<args>(args_in)...), allocator_(allocator) {}

  template <class... args>
  co_sm(const scheduler_type & scheduler, const allocator_type & allocator, args &&... args_in)
      : state_machine_(std::forward<args>(args_in)...),
        scheduler_(scheduler),
        allocator_(allocator) {}

  template <class event>
  bool process_event(const event & ev) {
    const bool accepted = state_machine_.process_event(ev);
    return detail::normalize_event_result(ev, accepted);
  }

  template <class event>
  bool_task process_event_async(const event & ev) {
    if constexpr (std::is_same_v<scheduler_type, policy::inline_scheduler>) {
      const bool accepted = detail::normalize_event_result(ev, state_machine_.process_event(ev));
      return bool_task::from_value(accepted);
    }
    if constexpr (policy::has_try_run_immediate<scheduler_type>) {
      bool accepted = false;
      if (scheduler_.try_run_immediate([this, &ev, &accepted]() {
            accepted = detail::normalize_event_result(ev, state_machine_.process_event(ev));
          })) {
        return bool_task::from_value(accepted);
      }
    }
    return process_event_async_impl(std::allocator_arg, allocator_, *this, ev);
  }

  template <class state>
  bool is(state state_value = {}) const {
    return state_machine_.is(state_value);
  }

  template <class visitor>
  void visit_current_states(visitor && visitor_fn) {
    state_machine_.visit_current_states(std::forward<visitor>(visitor_fn));
  }

  scheduler_type & scheduler() noexcept { return scheduler_; }
  const scheduler_type & scheduler() const noexcept { return scheduler_; }
  allocator_type & allocator() noexcept { return allocator_; }
  const allocator_type & allocator() const noexcept { return allocator_; }

 protected:
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  template <class event>
  static bool_task process_event_async_impl(
      std::allocator_arg_t,
      allocator_type & allocator_value,
      co_sm & self,
      const event & ev) {
    (void)allocator_value;
    co_return co_await process_event_awaitable<event>{self, ev};
  }

  template <class event>
  struct process_event_awaitable {
    co_sm & self;
    const event & event_value;
    bool accepted = false;

    bool await_ready() noexcept {
      if constexpr (std::is_same_v<scheduler_type, policy::inline_scheduler>) {
        accepted =
          detail::normalize_event_result(event_value, self.state_machine_.process_event(event_value));
        return true;
      }
      return false;
    }

    void await_suspend(std::coroutine_handle<> handle) {
      self.scheduler_.schedule([this, handle]() mutable {
        accepted = detail::normalize_event_result(
          event_value,
          self.state_machine_.process_event(event_value));
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
