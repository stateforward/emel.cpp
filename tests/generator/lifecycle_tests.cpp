#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>
#include <string>

#include "emel/docs/detail.hpp"
#include "emel/emel.h"
#include "emel/generator/errors.hpp"
#include "emel/generator/sm.hpp"

namespace {

struct callback_tracker {
  bool done_called = false;
  bool error_called = false;
  int32_t tokens_generated = -1;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
};

void dispatch_done_test(void * owner, const emel::generator::events::generation_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->done_called = true;
  tracker->tokens_generated = ev.tokens_generated;
}

void dispatch_error_test(void * owner, const emel::generator::events::generation_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->error_called = true;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->err = static_cast<emel::error::type>(ev.err);
}

template <class... Ts, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

}  // namespace

TEST_CASE("generator_starts_ready") {
  emel::generator::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::generator::ready>));
}

TEST_CASE("generator_valid_generate_dispatches_done") {
  emel::generator::sm machine{};
  emel::error::type error = emel::error::cast(emel::generator::error::backend);
  callback_tracker tracker{};
  const auto on_done = emel::callback<void(const emel::generator::events::generation_done &)>(
      &tracker, dispatch_done_test);
  const auto on_error = emel::callback<void(const emel::generator::events::generation_error &)>(
      &tracker, dispatch_error_test);

  emel::generator::event::generate ev{
    .prompt = "hello",
    .max_tokens = 3,
    .error_out = &error,
    .on_done = on_done,
    .on_error = on_error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::generator::ready>));
  CHECK(tracker.done_called);
  CHECK_FALSE(tracker.error_called);
  CHECK(error == emel::error::cast(emel::generator::error::none));
  CHECK(tracker.tokens_generated == 3);
}

TEST_CASE("generator_invalid_generate_dispatches_error") {
  emel::generator::sm machine{};
  emel::error::type error = emel::error::cast(emel::generator::error::none);
  callback_tracker tracker{};
  const auto on_done = emel::callback<void(const emel::generator::events::generation_done &)>(
      &tracker, dispatch_done_test);
  const auto on_error = emel::callback<void(const emel::generator::events::generation_error &)>(
      &tracker, dispatch_error_test);

  emel::generator::event::generate ev{
    .prompt = "hello",
    .max_tokens = 0,
    .error_out = &error,
    .on_done = on_done,
    .on_error = on_error,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::generator::ready>));
  CHECK_FALSE(tracker.done_called);
  CHECK(tracker.error_called);
  CHECK(error == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(tracker.tokens_generated == 0);
}

TEST_CASE("generator_docs_table_uses_typed_completion_event_name") {
  using machine_t = boost::sml::sm<emel::generator::model>;
  using transitions = typename machine_t::transitions;

  bool has_completion_transition = false;
  bool has_typed_completion = false;

  for_each_type(transitions{}, [&]<class transition_t>() {
    using event = typename transition_t::event;
    const std::string event_name = emel::docs::detail::table_event_name<event>();
    if (event_name == "completion") {
      has_completion_transition = true;
      return;
    }
    if (event_name == "completion<generate_run>") {
      has_completion_transition = true;
      has_typed_completion = true;
    }
  });

  CHECK(has_completion_transition);
  CHECK(has_typed_completion);
}

TEST_CASE("docs_detail_shortens_lambda_type_names_for_mermaid") {
  using emel::docs::detail::shorten_type_name;

  CHECK(shorten_type_name("lambda at /tmp/path/my_action.cpp:42:7>") == "lambda_my_action_42_7");
  CHECK(shorten_type_name("lambda at my_action.cpp:42>") == "lambda_my_action_42");
  CHECK(shorten_type_name("lambda at my_action.cpp>") == "lambda_my_action");
}

TEST_CASE("docs_detail_table_event_name_supports_non_completion_event") {
  const auto event_name =
      emel::docs::detail::table_event_name<emel::generator::event::generate_run>();
  CHECK(event_name == "generate_run");
}
