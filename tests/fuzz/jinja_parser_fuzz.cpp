#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/sm.hpp"

namespace {

struct dispatch_state {
  bool done_called = false;
  bool error_called = false;
};

bool on_done(void * owner, const emel::text::jinja::events::parsing_done &) noexcept {
  auto * state = static_cast<dispatch_state *>(owner);
  state->done_called = true;
  return true;
}

bool on_error(void * owner, const emel::text::jinja::events::parsing_error &) noexcept {
  auto * state = static_cast<dispatch_state *>(owner);
  state->error_called = true;
  return true;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
  emel::text::jinja::program program{};
  emel::text::jinja::parser::action::context parse_ctx{};
  emel::text::jinja::parser::sm parser{parse_ctx};
  dispatch_state state{};
  int32_t parse_err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t parse_error_pos = 0;

  const emel::text::jinja::event::parse::done_callback done_cb{&state, on_done};
  const emel::text::jinja::event::parse::error_callback error_cb{&state, on_error};
  const std::string_view input(reinterpret_cast<const char *>(data), size);
  const emel::text::jinja::event::parse parse_ev{
      input,
      program,
      done_cb,
      error_cb,
      parse_err,
      parse_error_pos,
  };

  (void)parser.process_event(parse_ev);
  return 0;
}
