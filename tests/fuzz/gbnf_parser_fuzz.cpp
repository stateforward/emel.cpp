#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"

namespace {

struct fuzz_dispatch_state {
  bool done_called = false;
  bool error_called = false;
};

bool on_done(void * owner, const emel::gbnf::rule_parser::events::parsing_done &) noexcept {
  auto * state = static_cast<fuzz_dispatch_state *>(owner);
  state->done_called = true;
  return true;
}

bool on_error(void * owner, const emel::gbnf::rule_parser::events::parsing_error &) noexcept {
  auto * state = static_cast<fuzz_dispatch_state *>(owner);
  state->error_called = true;
  return true;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
  emel::gbnf::grammar grammar{};
  emel::gbnf::rule_parser::sm machine{};
  fuzz_dispatch_state dispatch_state{};

  std::string_view input(reinterpret_cast<const char *>(data), size);
  const emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_done &)> done_cb{
      &dispatch_state,
      on_done};
  const emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_error &)> error_cb{
      &dispatch_state,
      on_error};
  emel::gbnf::rule_parser::event::parse ev{};
  ev.grammar_text = input;
  ev.grammar_out = &grammar;
  ev.dispatch_done = done_cb;
  ev.dispatch_error = error_cb;

  (void)machine.process_event(ev);
  return 0;
}
