#include "bench_cases.hpp"

#include <cstdint>
#include <cstdlib>

#include "emel/sm.hpp"

namespace {

bool bench_internal_enabled() {
  const char * value = std::getenv("EMEL_BENCH_INTERNAL");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] != '0';
}

struct event_alpha {
  volatile int32_t * sink = nullptr;
};

struct event_beta {
  volatile int32_t * sink = nullptr;
};

struct idle_state {};

struct set_alpha_first {
  void operator()(const event_alpha & ev) const noexcept {
    if (ev.sink != nullptr) {
      *ev.sink = 11;
    }
  }
};

struct set_beta_first {
  void operator()(const event_beta & ev) const noexcept {
    if (ev.sink != nullptr) {
      *ev.sink = 12;
    }
  }
};

struct set_alpha_second {
  void operator()(const event_alpha & ev) const noexcept {
    if (ev.sink != nullptr) {
      *ev.sink = 21;
    }
  }
};

struct set_beta_second {
  void operator()(const event_beta & ev) const noexcept {
    if (ev.sink != nullptr) {
      *ev.sink = 22;
    }
  }
};

struct first_model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
        *sml::state<idle_state> + sml::event<event_alpha> / set_alpha_first{} =
            sml::state<idle_state>,
        sml::state<idle_state> + sml::event<event_beta> / set_beta_first{} =
            sml::state<idle_state>);
  }
};

struct second_model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
        *sml::state<idle_state> + sml::event<event_alpha> / set_alpha_second{} =
            sml::state<idle_state>,
        sml::state<idle_state> + sml::event<event_beta> / set_beta_second{} =
            sml::state<idle_state>);
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

namespace emel::bench {

void append_emel_sm_any_cases(std::vector<result> & results, const config & cfg) {
  if (!bench_internal_enabled()) {
    return;
  }

  volatile int32_t sink = 0;
  event_alpha alpha{&sink};

  {
    first_sm direct{};
    auto fn = [&]() { (void)direct.process_event(alpha); };
    results.push_back(measure_case("sm_any/direct", cfg, fn));
  }

  {
    any_sm any{test_kind::first};
    auto fn = [&]() { (void)any.process_event(alpha); };
    results.push_back(measure_case("sm_any/any", cfg, fn));
  }
}

void append_reference_sm_any_cases(std::vector<result> & results, const config & cfg) {
  if (!bench_internal_enabled()) {
    return;
  }

  volatile int32_t sink = 0;
  event_alpha alpha{&sink};

  {
    first_sm direct{};
    auto fn = [&]() { (void)direct.process_event(alpha); };
    results.push_back(measure_case("sm_any/direct", cfg, fn));
  }

  {
    first_sm direct{};
    auto fn = [&]() { (void)direct.process_event(alpha); };
    results.push_back(measure_case("sm_any/any", cfg, fn));
  }
}

}  // namespace emel::bench
