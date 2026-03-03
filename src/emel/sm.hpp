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
  const bool accepted_ok = accepted;
  if constexpr (requires { ev.error_out; }) {
    using error_member = std::remove_reference_t<decltype(ev.error_out)>;
    if constexpr (std::is_pointer_v<error_member>) {
      const bool error_is_clear = ev.error_out == nullptr || *ev.error_out == 0;
      return accepted_ok && error_is_clear;
    } else {
      const bool error_is_clear = ev.error_out == 0;
      return accepted_ok && error_is_clear;
    }
  }
  return accepted_ok;
}

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

template <class model, class... policies>
class sm<model, void, policies...> {
 public:
  using model_type = model;
  using context_type = void;
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

template <class model, class context, class... policies>
class sm {
 public:
  static_assert(!std::is_void_v<context>, "contextful sm requires a non-void context type");

  using model_type = model;
  using context_type = context;
  using state_machine_type = boost::sml::sm<model, policies...>;

  sm() : state_machine_(context_) {}
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
  context_type context_{};
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
    const bool changed = next != index_;
    while (changed) {
      destroy();
      construct(next);
      return;
    }
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
