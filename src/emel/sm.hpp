#pragma once

#include <stateforward/sml.hpp>
#include <stateforward/sml/utility/co_sm.hpp>
#include <stateforward/sml/utility/external_completion.hpp>
#include <stateforward/sml/utility/thread_pool_scheduler.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <tuple>
#include <utility>

namespace emel {

namespace policy {

using inline_scheduler = stateforward::sml::utility::policy::inline_scheduler;

template <std::size_t capacity = 1024, std::size_t inline_task_bytes = 64>
using fifo_scheduler =
    stateforward::sml::utility::policy::fifo_scheduler<capacity, inline_task_bytes>;

template <class scheduler>
using coroutine_scheduler =
    stateforward::sml::utility::policy::coroutine_scheduler<scheduler>;

template <class allocator>
using coroutine_allocator =
    stateforward::sml::utility::policy::coroutine_allocator<allocator>;

template <std::size_t slot_size = 1024, std::size_t slot_count = 64>
class fixed_coroutine_allocator {
 public:
  static_assert(slot_size > 0,
                "fixed_coroutine_allocator slot size must be non-zero");
  static_assert(slot_count > 0,
                "fixed_coroutine_allocator slot count must be non-zero");

  fixed_coroutine_allocator() noexcept { reset_freelist(); }

  void * allocate(const std::size_t size, const std::size_t alignment) noexcept {
    const bool fits = size <= slot_size && alignment <= alignof(pool_slot);
    const bool available = free_head_ != invalid_index;
    if (fits && available) {
      const std::size_t slot_index = free_head_;
      free_head_ = next_free_[slot_index];
      return static_cast<void *>(slots_[slot_index].storage.data());
    }
    return nullptr;
  }

  void deallocate(void * ptr, const std::size_t size,
                  const std::size_t alignment) noexcept {
    const bool reusable = ptr != nullptr && size <= slot_size &&
                          alignment <= alignof(pool_slot) &&
                          is_pool_pointer(ptr);
    if (reusable) {
      const std::size_t slot_index = slot_index_for(ptr);
      next_free_[slot_index] = free_head_;
      free_head_ = slot_index;
    }
  }

 private:
  static constexpr std::size_t invalid_index = slot_count;

  struct pool_slot {
    alignas(std::max_align_t) std::array<unsigned char, slot_size> storage{};
  };

  bool is_pool_pointer(void * ptr) const noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(slots_.data());
    const auto end = begin + sizeof(slots_);
    const auto candidate = reinterpret_cast<std::uintptr_t>(ptr);
    const bool in_range = candidate >= begin && candidate < end;
    if (in_range) {
      const std::size_t offset = static_cast<std::size_t>(candidate - begin);
      return (offset % sizeof(pool_slot)) == 0;
    }
    return false;
  }

  std::size_t slot_index_for(void * ptr) const noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(slots_.data());
    const auto candidate = reinterpret_cast<std::uintptr_t>(ptr);
    const std::size_t offset = static_cast<std::size_t>(candidate - begin);
    return offset / sizeof(pool_slot);
  }

  void reset_freelist() noexcept {
    for (std::size_t i = 0; i + 1 < slot_count; ++i) {
      next_free_[i] = i + 1;
    }
    next_free_[slot_count - 1] = invalid_index;
    free_head_ = 0;
  }

  std::array<pool_slot, slot_count> slots_{};
  std::array<std::size_t, slot_count> next_free_{};
  std::size_t free_head_ = 0;
};

template <std::size_t worker_count = 2, std::size_t capacity = 1024,
          std::size_t inline_task_bytes = 64>
using thread_pool_scheduler =
    stateforward::sml::utility::policy::thread_pool_scheduler<
        worker_count, capacity, inline_task_bytes>;

using stateforward::sml::utility::policy::cpu_relax;

template <std::size_t worker_count, std::size_t inline_task_bytes = 128,
          std::size_t idle_spin_budget = 1048576>
class fork_join_lane_pool {
 public:
  static_assert(worker_count > 0, "fork_join_lane_pool needs workers");
  static_assert(inline_task_bytes > 0,
                "fork_join_lane_pool inline storage must be non-zero");

  static constexpr std::size_t static_worker_count = worker_count;

  fork_join_lane_pool() { start_workers(); }

  ~fork_join_lane_pool() { stop_workers(); }

  fork_join_lane_pool(const fork_join_lane_pool &) = delete;
  fork_join_lane_pool & operator=(const fork_join_lane_pool &) = delete;
  fork_join_lane_pool(fork_join_lane_pool &&) = delete;
  fork_join_lane_pool & operator=(fork_join_lane_pool &&) = delete;

  class join_group {
   public:
    join_group() = default;

    join_group(const join_group &) = delete;
    join_group & operator=(const join_group &) = delete;

    bool wait() noexcept {
      while (pending_.load(std::memory_order_acquire) != 0u) {
        cpu_relax();
      }
      const bool accepted = accepted_.load(std::memory_order_acquire);
      accepted_.store(true, std::memory_order_release);
      return accepted;
    }

   private:
    friend class fork_join_lane_pool;

    void start_one() noexcept {
      pending_.fetch_add(1u, std::memory_order_acq_rel);
    }

    void reject_one() noexcept {
      accepted_.store(false, std::memory_order_release);
      complete_one();
    }

    void reject() noexcept {
      accepted_.store(false, std::memory_order_release);
    }

    void complete_one() noexcept {
      pending_.fetch_sub(1u, std::memory_order_acq_rel);
    }

    static void complete_one(void *ctx) noexcept {
      static_cast<join_group *>(ctx)->complete_one();
    }

    std::atomic<std::uint32_t> pending_ = 0u;
    std::atomic<bool> accepted_ = true;
  };

  template <class fn>
  bool try_submit(join_group & group, fn && fn_in) noexcept {
    if (running_on_this_worker()) {
      group.reject();
      return false;
    }

    group.start_one();
    const bool submitted = try_submit_with_completion(
        std::forward<fn>(fn_in), &group, join_group::complete_one);
    if (!submitted) {
      group.reject_one();
      return false;
    }
    return true;
  }

  bool is_current_thread_worker() const noexcept {
    return running_on_this_worker();
  }

 private:
  struct task_slot {
    using invoke_fn = void (*)(void *) noexcept;
    using destroy_fn = void (*)(void *) noexcept;

    alignas(std::max_align_t) std::array<unsigned char, inline_task_bytes>
        storage{};
    invoke_fn invoke = nullptr;
    destroy_fn destroy = nullptr;
    void *completion_ctx = nullptr;
    void (*completion_fn)(void *) noexcept = nullptr;

    template <class fn>
    void set(fn && fn_in, void *ctx,
             void (*completion)(void *) noexcept) noexcept {
      using fn_type = std::decay_t<fn>;
      static_assert(sizeof(fn_type) <= inline_task_bytes,
                    "fork_join_lane_pool task exceeds inline storage");
      static_assert(alignof(fn_type) <= alignof(std::max_align_t),
                    "fork_join_lane_pool task alignment is too large");

      new (storage.data()) fn_type(std::forward<fn>(fn_in));
      invoke = [](void *ptr) noexcept { (*static_cast<fn_type *>(ptr))(); };
      destroy = [](void *ptr) noexcept {
        static_cast<fn_type *>(ptr)->~fn_type();
      };
      completion_ctx = ctx;
      completion_fn = completion;
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
      completion_ctx = nullptr;
      completion_fn = nullptr;
    }
  };

  struct worker_slot {
    task_slot task = {};
    std::counting_semaphore<> ready{0};
    std::thread thread = {};
    std::atomic<bool> idle = true;
    std::atomic<bool> stopping = false;
  };

  void start_workers() {
    try {
      for (std::size_t index = 0u; index < worker_count; ++index) {
        workers_[index].thread =
            std::thread([this, index]() noexcept { worker_loop(index); });
      }
    } catch (...) {
      stop_workers();
      throw;
    }
  }

  void stop_workers() noexcept {
    for (auto &worker : workers_) {
      worker.stopping.store(true, std::memory_order_release);
      worker.ready.release();
    }
    for (auto &worker : workers_) {
      if (worker.thread.joinable()) {
        worker.thread.join();
      }
      worker.task.reset();
    }
  }

  template <class fn>
  bool try_submit_with_completion(fn && fn_in, void *completion_ctx,
                                  void (*completion_fn)(void *) noexcept)
      noexcept {
    const std::size_t start =
        next_worker_.fetch_add(1u, std::memory_order_relaxed) % worker_count;
    for (std::size_t offset = 0u; offset < worker_count; ++offset) {
      worker_slot &worker = workers_[(start + offset) % worker_count];
      bool expected = true;
      if (!worker.idle.compare_exchange_strong(
              expected, false, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        continue;
      }
      worker.task.set(std::forward<fn>(fn_in), completion_ctx, completion_fn);
      std::atomic_thread_fence(std::memory_order_release);
      worker.ready.release();
      return true;
    }
    return false;
  }

  void worker_loop(const std::size_t index) noexcept {
    struct worker_scope {
      const fork_join_lane_pool *previous;
      explicit worker_scope(const fork_join_lane_pool *current) noexcept
          : previous(active_worker_pool_) {
        active_worker_pool_ = current;
      }
      ~worker_scope() noexcept { active_worker_pool_ = previous; }
    } scope{this};

    worker_slot &worker = workers_[index];
    for (;;) {
      bool claimed = false;
      for (std::size_t spin = 0u; spin < idle_spin_budget; ++spin) {
        if (worker.ready.try_acquire()) {
          claimed = true;
          break;
        }
        if (worker.stopping.load(std::memory_order_acquire)) {
          return;
        }
        cpu_relax();
      }
      if (!claimed) {
        worker.ready.acquire();
      }
      if (worker.stopping.load(std::memory_order_acquire) &&
          worker.task.invoke == nullptr) {
        return;
      }

      worker.task.run();
      void *completion_ctx = worker.task.completion_ctx;
      void (*completion_fn)(void *) noexcept = worker.task.completion_fn;
      worker.task.completion_ctx = nullptr;
      worker.task.completion_fn = nullptr;
      worker.idle.store(true, std::memory_order_release);
      if (completion_fn != nullptr) {
        completion_fn(completion_ctx);
      }
    }
  }

  bool running_on_this_worker() const noexcept {
    return active_worker_pool_ == this;
  }

  std::array<worker_slot, worker_count> workers_{};
  std::atomic<std::size_t> next_worker_ = 0u;
  inline static thread_local const fork_join_lane_pool *active_worker_pool_ =
      nullptr;
};

class fork_join_start_gate {
 public:
  fork_join_start_gate() = default;

  fork_join_start_gate(const fork_join_start_gate &) = delete;
  fork_join_start_gate & operator=(const fork_join_start_gate &) = delete;

  void wait() const noexcept {
    while (!open_.load(std::memory_order_acquire)) {
      cpu_relax();
    }
  }

  void arrive_and_wait() noexcept {
    arrived_.fetch_add(1u, std::memory_order_acq_rel);
    wait();
  }

  void open() noexcept { open_.store(true, std::memory_order_release); }

  void open_after_arrivals(const std::size_t expected_arrivals) noexcept {
    while (arrived_.load(std::memory_order_acquire) < expected_arrivals) {
      cpu_relax();
    }
    open();
  }

 private:
  std::atomic<std::size_t> arrived_ = 0u;
  std::atomic<bool> open_ = false;
};

using completion_source = stateforward::sml::utility::policy::completion_source;

template <std::size_t source_count = 8>
using external_completion_scheduler =
    stateforward::sml::utility::policy::external_completion_scheduler<source_count>;

template <class scheduler>
concept external_completion_scheduler_contract =
    stateforward::sml::utility::policy::external_completion_scheduler_contract<scheduler>;

using default_coroutine_scheduler = coroutine_scheduler<inline_scheduler>;
using default_coroutine_allocator =
    coroutine_allocator<fixed_coroutine_allocator<>>;

template <std::size_t source_count = 8>
using external_completion_co_policy =
    coroutine_scheduler<external_completion_scheduler<source_count>>;

template <class scheduler>
concept strict_ordering_scheduler_contract =
    requires {
      { scheduler::guarantees_fifo } -> std::convertible_to<bool>;
      { scheduler::single_consumer } -> std::convertible_to<bool>;
      { scheduler::run_to_completion } -> std::convertible_to<bool>;
    } && static_cast<bool>(scheduler::guarantees_fifo) &&
    static_cast<bool>(scheduler::single_consumer) &&
    static_cast<bool>(scheduler::run_to_completion);

}  // namespace policy

using bool_task = stateforward::sml::utility::bool_task;

namespace event {

// Generic completion trigger delivered by an external-completion co_sm:
// source_index names the policy::completion_source that fired; machines map it
// onto domain state (for example a window slot) via their own guards/actions.
using completion = stateforward::sml::utility::completion;

}  // namespace event

namespace detail {

class dispatch_scope {
 public:
  explicit dispatch_scope(std::atomic<bool> & active) noexcept
      : active_(&active) {
    bool expected = false;
    acquired_ = active_->compare_exchange_strong(
        expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  dispatch_scope(const dispatch_scope &) = delete;
  dispatch_scope & operator=(const dispatch_scope &) = delete;

  ~dispatch_scope() noexcept {
    if (acquired_) {
      active_->store(false, std::memory_order_release);
    }
  }

  explicit operator bool() const noexcept { return acquired_; }

 private:
  std::atomic<bool> * active_ = nullptr;
  bool acquired_ = false;
};

template <class owner, class process>
struct process_support {
  struct immediate_queue {
    using container_type = void;
    owner * owner_ptr = nullptr;

    template <class event>
    void push(const event & ev) noexcept {
      while (owner_ptr != nullptr) {
        (void)owner_ptr->process_event(ev);
        break;
      }
    }
  };

  explicit process_support(owner * owner_ptr) : queue_{owner_ptr}, process_{queue_} {}

  immediate_queue queue_{};
  process process_;
};

template <class... types, class fn>
constexpr void for_each_type(stateforward::sml::aux::type_list<types...>, fn && visitor) {
  (visitor.template operator()<types>(), ...);
}

template <class list>
struct type_list_size;

template <class... types>
struct type_list_size<stateforward::sml::aux::type_list<types...>>
    : std::integral_constant<std::size_t, sizeof...(types)> {};

template <class list>
inline constexpr std::size_t type_list_size_v = type_list_size<list>::value;

template <class list>
struct type_list_max;

template <class... types>
struct type_list_max<stateforward::sml::aux::type_list<types...>> {
  static constexpr std::size_t size = std::max({sizeof(types)...});
  static constexpr std::size_t align = std::max({alignof(types)...});
};

template <class type, class list>
struct type_list_contains;

template <class type, class... types>
struct type_list_contains<type, stateforward::sml::aux::type_list<types...>>
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
struct sm_any_construct_table<stateforward::sml::aux::type_list<types...>> {
  using fn = void (*)(void *);
  inline static constexpr std::array<fn, sizeof...(types)> table = {
      &sm_any_construct<types>...};
};

template <class list>
struct sm_any_destroy_table;

template <class... types>
struct sm_any_destroy_table<stateforward::sml::aux::type_list<types...>> {
  using fn = void (*)(void *) noexcept;
  inline static constexpr std::array<fn, sizeof...(types)> table = {
      &sm_any_destroy<types>...};
};

template <class event, class list>
struct sm_any_event_table;

template <class event, class... types>
struct sm_any_event_table<event, stateforward::sml::aux::type_list<types...>> {
  using fn = bool (*)(void *, const event &);
  inline static constexpr std::array<fn, sizeof...(types)> table = {
      &sm_any_process_event<types, event>...};
};

template <class event_list>
struct sm_any_process_event_table;

template <class... events>
struct sm_any_process_event_table<stateforward::sml::aux::type_list<events...>> {
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
struct sm_any_visit<stateforward::sml::aux::type_list<types...>> {
  template <std::size_t idx, class visitor, class first, class... rest>
  static void apply_index(std::size_t target, void * storage, visitor && visitor_fn) {
    const bool matched = target == idx;
    while (matched) {
      visitor_fn(*sm_any_ptr<first>(storage));
      return;
    }
    if constexpr (sizeof...(rest) > 0) {
      while (!matched) {
        apply_index<idx + 1, visitor, rest...>(
            target, storage, std::forward<visitor>(visitor_fn));
        break;
      }
    }
  }

  template <std::size_t idx, class visitor, class first, class... rest>
  static void apply_index(std::size_t target, const void * storage,
                          visitor && visitor_fn) {
    const bool matched = target == idx;
    while (matched) {
      visitor_fn(*sm_any_ptr<first>(storage));
      return;
    }
    if constexpr (sizeof...(rest) > 0) {
      while (!matched) {
        apply_index<idx + 1, visitor, rest...>(
            target, storage, std::forward<visitor>(visitor_fn));
        break;
      }
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

template <class model, class context = void, class... policies>
class sm;

template <class model, class context = void,
          class scheduler_policy = policy::default_coroutine_scheduler,
          class allocator_policy = policy::default_coroutine_allocator,
          class... policies>
class co_sm;

template <class model, class... policies>
class sm<model, void, policies...> {
 public:
  using model_type = model;
  using context_type = void;
  using state_machine_type = stateforward::sml::sm<model, policies...>;

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
    return state_machine_.process_event(ev);
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
  // sml's policy-less back-end embeds aux::pool<> (a zero-size array); g++
  // -Wpedantic flags members of such types at the declaration site.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  state_machine_type state_machine_;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

template <class model, class context, class... policies>
class sm {
 public:
  static_assert(!std::is_void_v<context>, "contextful sm requires a non-void context type");

  using model_type = model;
  using context_type = context;
  using state_machine_type = stateforward::sml::sm<model, policies...>;

  sm() : state_machine_(context_) {}
  template <class... context_args>
  explicit sm(std::in_place_t, context_args &&... context_args_in)
      : context_(std::forward<context_args>(context_args_in)...),
        state_machine_(context_) {}
  explicit sm(const context_type & context_in)
      : context_(context_in), state_machine_(context_) {}
  explicit sm(context_type && context_in)
      : context_(std::move(context_in)), state_machine_(context_) {}

  template <class... args>
    requires (sizeof...(args) > 0)
  explicit sm(const context_type & context_in, args &&... args_in)
      : context_(context_in),
        state_machine_(context_, std::forward<args>(args_in)...) {}

  template <class... args>
    requires (sizeof...(args) > 0)
  explicit sm(context_type && context_in, args &&... args_in)
      : context_(std::move(context_in)),
        state_machine_(context_, std::forward<args>(args_in)...) {}

  sm(const sm &) = default;
  sm(sm &&) = default;
  sm & operator=(const sm &) = default;
  sm & operator=(sm &&) = default;
  ~sm() = default;

  template <class event>
  bool process_event(const event & ev) {
    return state_machine_.process_event(ev);
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
  context_type context_{};
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  // sml's policy-less back-end embeds aux::pool<> (a zero-size array); g++
  // -Wpedantic flags members of such types at the declaration site.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  state_machine_type state_machine_;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

namespace detail {

template <class scheduler, class = void>
struct is_multi_consumer_scheduler : std::false_type {};

template <class scheduler>
struct is_multi_consumer_scheduler<
    scheduler, std::void_t<decltype(scheduler::multi_consumer)>>
    : std::bool_constant<static_cast<bool>(scheduler::multi_consumer)> {};

template <class scheduler>
inline constexpr bool is_multi_consumer_scheduler_v =
    is_multi_consumer_scheduler<scheduler>::value;

template <class model, class scheduler_policy, class allocator_policy,
          class... policies>
using co_sm_backend_t =
    stateforward::sml::utility::co_sm<model, scheduler_policy,
                                      allocator_policy, policies...>;

}  // namespace detail

template <class model, class scheduler_policy, class allocator_policy,
          class... policies>
class co_sm<model, void, scheduler_policy, allocator_policy, policies...> {
 public:
  using model_type = model;
  using context_type = void;
  using scheduler_policy_type = scheduler_policy;
  using scheduler_type = typename scheduler_policy_type::scheduler_type;
  using allocator_policy_type = allocator_policy;
  using allocator_type = typename allocator_policy_type::allocator_type;
  using state_machine_type =
      detail::co_sm_backend_t<model, scheduler_policy, allocator_policy,
                              policies...>;

  co_sm() = default;
  ~co_sm() = default;

  co_sm(const co_sm &) = default;
  co_sm(co_sm &&) = default;
  co_sm & operator=(const co_sm &) = default;
  co_sm & operator=(co_sm &&) = default;

  template <class... args>
  explicit co_sm(args &&... args_in)
      : state_machine_(std::forward<args>(args_in)...) {}

  explicit co_sm(scheduler_type scheduler_in)
    requires detail::is_multi_consumer_scheduler_v<scheduler_type>
      : state_machine_(scheduler_in) {}

  template <class... args>
  explicit co_sm(scheduler_type scheduler_in, args &&... args_in)
    requires detail::is_multi_consumer_scheduler_v<scheduler_type>
      : state_machine_(scheduler_in, std::forward<args>(args_in)...) {}

  template <class event>
  bool process_event(const event & ev) {
    return state_machine_.process_event(ev);
  }

  template <class event>
  bool_task process_event_async(const event & ev) {
    if constexpr (std::is_same_v<scheduler_type, policy::inline_scheduler>) {
      const bool accepted = state_machine_.process_event(ev);
      return bool_task::from_value(accepted);
    } else if constexpr (policy::external_completion_scheduler_contract<scheduler_type>) {
      // The upstream backend drives suspension to completion on this thread
      // before returning, so the task is always immediately ready.
      const bool accepted = state_machine_.process_event(ev);
      return bool_task::from_value(accepted);
    } else if constexpr (detail::is_multi_consumer_scheduler_v<scheduler_type>) {
      return state_machine_.process_event_async(ev);
    } else if constexpr (requires(scheduler_type & scheduler) {
                           {
                             scheduler.try_run_immediate([]() noexcept {})
                           } -> std::same_as<bool>;
                         }) {
      bool accepted = false;
      const bool completed = state_machine_.scheduler().try_run_immediate(
          [this, &ev, &accepted]() {
            accepted = state_machine_.process_event(ev);
          });
      return bool_task::from_value(completed && accepted);
    } else {
      return state_machine_.process_event_async(ev);
    }
  }

  template <class state>
  bool is(state state_value = {}) const {
    return state_machine_.is(state_value);
  }

  template <class visitor>
  void visit_current_states(visitor && visitor_fn) {
    state_machine_.visit_current_states(std::forward<visitor>(visitor_fn));
  }

  scheduler_type & scheduler() noexcept { return state_machine_.scheduler(); }
  const scheduler_type & scheduler() const noexcept {
    return state_machine_.scheduler();
  }

  allocator_type & allocator() noexcept { return state_machine_.allocator(); }
  const allocator_type & allocator() const noexcept {
    return state_machine_.allocator();
  }

 protected:
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  // sml's policy-less back-end embeds aux::pool<> (a zero-size array); g++
  // -Wpedantic flags members of such types at the declaration site.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  state_machine_type state_machine_;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

template <class model, class context, class scheduler_policy,
          class allocator_policy, class... policies>
class co_sm {
 public:
  static_assert(!std::is_void_v<context>, "contextful co_sm requires a non-void context type");

  using model_type = model;
  using context_type = context;
  using scheduler_policy_type = scheduler_policy;
  using scheduler_type = typename scheduler_policy_type::scheduler_type;
  using allocator_policy_type = allocator_policy;
  using allocator_type = typename allocator_policy_type::allocator_type;
  using state_machine_type =
      detail::co_sm_backend_t<model, scheduler_policy, allocator_policy,
                              policies...>;

  co_sm() : state_machine_(context_) {}
  explicit co_sm(const context_type & context_in)
      : context_(context_in), state_machine_(context_) {}
  explicit co_sm(context_type && context_in)
      : context_(std::move(context_in)), state_machine_(context_) {}

  explicit co_sm(scheduler_type scheduler_in)
    requires detail::is_multi_consumer_scheduler_v<scheduler_type>
      : state_machine_(scheduler_in, context_) {}

  co_sm(scheduler_type scheduler_in, const context_type & context_in)
    requires detail::is_multi_consumer_scheduler_v<scheduler_type>
      : context_(context_in), state_machine_(scheduler_in, context_) {}

  co_sm(scheduler_type scheduler_in, context_type && context_in)
    requires detail::is_multi_consumer_scheduler_v<scheduler_type>
      : context_(std::move(context_in)), state_machine_(scheduler_in, context_) {}

  template <class... args>
    requires (sizeof...(args) > 0)
  explicit co_sm(const context_type & context_in, args &&... args_in)
      : context_(context_in),
        state_machine_(context_, std::forward<args>(args_in)...) {}

  template <class... args>
    requires (sizeof...(args) > 0)
  explicit co_sm(context_type && context_in, args &&... args_in)
      : context_(std::move(context_in)),
        state_machine_(context_, std::forward<args>(args_in)...) {}

  co_sm(const co_sm &) = default;
  co_sm(co_sm &&) = default;
  co_sm & operator=(const co_sm &) = default;
  co_sm & operator=(co_sm &&) = default;
  ~co_sm() = default;

  template <class event>
  bool process_event(const event & ev) {
    return state_machine_.process_event(ev);
  }

  template <class event>
  bool_task process_event_async(const event & ev) {
    if constexpr (std::is_same_v<scheduler_type, policy::inline_scheduler>) {
      const bool accepted = state_machine_.process_event(ev);
      return bool_task::from_value(accepted);
    } else if constexpr (policy::external_completion_scheduler_contract<scheduler_type>) {
      // The upstream backend drives suspension to completion on this thread
      // before returning, so the task is always immediately ready.
      const bool accepted = state_machine_.process_event(ev);
      return bool_task::from_value(accepted);
    } else if constexpr (detail::is_multi_consumer_scheduler_v<scheduler_type>) {
      return state_machine_.process_event_async(ev);
    } else if constexpr (requires(scheduler_type & scheduler) {
                           {
                             scheduler.try_run_immediate([]() noexcept {})
                           } -> std::same_as<bool>;
                         }) {
      bool accepted = false;
      const bool completed = state_machine_.scheduler().try_run_immediate(
          [this, &ev, &accepted]() {
            accepted = state_machine_.process_event(ev);
          });
      return bool_task::from_value(completed && accepted);
    } else {
      return state_machine_.process_event_async(ev);
    }
  }

  template <class state>
  bool is(state state_value = {}) const {
    return state_machine_.is(state_value);
  }

  template <class visitor>
  void visit_current_states(visitor && visitor_fn) {
    state_machine_.visit_current_states(std::forward<visitor>(visitor_fn));
  }

  scheduler_type & scheduler() noexcept { return state_machine_.scheduler(); }
  const scheduler_type & scheduler() const noexcept {
    return state_machine_.scheduler();
  }

  allocator_type & allocator() noexcept { return state_machine_.allocator(); }
  const allocator_type & allocator() const noexcept {
    return state_machine_.allocator();
  }

 protected:
  context_type context_{};
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  // sml's policy-less back-end embeds aux::pool<> (a zero-size array); g++
  // -Wpedantic flags members of such types at the declaration site.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  state_machine_type state_machine_;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

template <class kind_enum, class sm_list, class event_list>
class sm_any {
 public:
  sm_any() { construct(default_index()); }
  explicit sm_any(const kind_enum kind) {
    construct(index_from_kind(kind));
    kind_ = kind;
  }

  sm_any(const sm_any &) = delete;
  sm_any & operator=(const sm_any &) = delete;
  sm_any(sm_any &&) = delete;
  sm_any & operator=(sm_any &&) = delete;

  ~sm_any() { destroy(); }

  void set_kind(const kind_enum kind) {
    const std::size_t next = index_from_kind(kind);
    const bool changed = next != index_;
    while (changed) {
      destroy();
      construct(next);
      break;
    }
    kind_ = kind;
  }

  // Reports the kind the owner requested, even when it does not name a real
  // variant. The active machine index is clamped to the first variant for such
  // kinds (safe construction), but the requested kind is preserved so
  // is_supported_kind(kind()) stays truthful and owners can reject dispatch to
  // an unsupported facade instead of silently running the default variant.
  kind_enum kind() const noexcept { return kind_; }

  // A kind is supported only when it names a real variant. Out-of-range kinds
  // (including the owner-defined `unsupported` sentinel) are clamped to the
  // first variant by index_from_kind, so owners must reject them before
  // dispatch instead of silently running the default variant.
  static constexpr bool is_supported_kind(const kind_enum kind) noexcept {
    return static_cast<std::size_t>(kind) < k_sm_count;
  }

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
    const std::size_t in_range = static_cast<std::size_t>(idx < k_sm_count);
    return in_range * idx + (std::size_t{1} - in_range) * default_index();
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

}  // namespace emel
