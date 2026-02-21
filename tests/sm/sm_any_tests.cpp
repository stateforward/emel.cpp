#include <cstdint>
#include <type_traits>

#include <doctest/doctest.h>

#include "emel/sm.hpp"

namespace {

struct event_alpha {
  int32_t * value = nullptr;
};

struct event_beta {
  int32_t * value = nullptr;
};

struct Idle {};

struct set_alpha_first {
  void operator()(const event_alpha & ev) const noexcept {
    if (ev.value != nullptr) {
      *ev.value = 11;
    }
  }
};

struct set_beta_first {
  void operator()(const event_beta & ev) const noexcept {
    if (ev.value != nullptr) {
      *ev.value = 12;
    }
  }
};

struct set_alpha_second {
  void operator()(const event_alpha & ev) const noexcept {
    if (ev.value != nullptr) {
      *ev.value = 21;
    }
  }
};

struct set_beta_second {
  void operator()(const event_beta & ev) const noexcept {
    if (ev.value != nullptr) {
      *ev.value = 22;
    }
  }
};

struct first_model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
        *sml::state<Idle> + sml::event<event_alpha> / set_alpha_first{} =
            sml::state<Idle>,
        sml::state<Idle> + sml::event<event_beta> / set_beta_first{} =
            sml::state<Idle>);
  }
};

struct second_model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
        *sml::state<Idle> + sml::event<event_alpha> / set_alpha_second{} =
            sml::state<Idle>,
        sml::state<Idle> + sml::event<event_beta> / set_beta_second{} =
            sml::state<Idle>);
  }
};

using first_sm = emel::sm<first_model>;
using second_sm = emel::sm<second_model>;

enum class test_kind : uint8_t {
  first = 0,
  second = 1,
};

using sm_list = boost::sml::aux::type_list<first_sm, second_sm>;
using event_list = boost::sml::aux::type_list<event_alpha, event_beta>;
using any_sm = emel::sm_any<test_kind, sm_list, event_list>;

}  // namespace

TEST_CASE("sm_any_dispatches_by_kind") {
  any_sm any{test_kind::first};

  int32_t value = 0;
  any.process_event(event_alpha{&value});
  CHECK(value == 11);

  any.process_event(event_beta{&value});
  CHECK(value == 12);

  any.set_kind(test_kind::second);
  any.process_event(event_alpha{&value});
  CHECK(value == 21);

  any.process_event(event_beta{&value});
  CHECK(value == 22);
}

TEST_CASE("sm_any_visit_hits_active_sm") {
  any_sm any{test_kind::first};
  bool is_first = false;
  any.visit([&](const auto & sm) {
    using sm_type = std::decay_t<decltype(sm)>;
    is_first = std::is_same_v<sm_type, first_sm>;
  });
  CHECK(is_first);

  any.set_kind(test_kind::second);
  bool is_second = false;
  any.visit([&](const auto & sm) {
    using sm_type = std::decay_t<decltype(sm)>;
    is_second = std::is_same_v<sm_type, second_sm>;
  });
  CHECK(is_second);
}

TEST_CASE("sm_any_normalizes_invalid_kind_to_default") {
  any_sm any{test_kind::second};
  any.set_kind(static_cast<test_kind>(42));

  int32_t value = 0;
  any.process_event(event_alpha{&value});
  CHECK(value == 11);
}
