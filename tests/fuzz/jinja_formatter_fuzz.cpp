#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/text/jinja/formatter/errors.hpp"
#include "emel/text/jinja/formatter/events.hpp"
#include "emel/text/jinja/formatter/sm.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/sm.hpp"

namespace {

struct parser_dispatch_state {
  bool done_called = false;
  bool error_called = false;
};

struct formatter_dispatch_state {
  bool done_called = false;
  bool error_called = false;
};

bool on_parse_done(void * owner, const emel::text::jinja::events::parsing_done &) noexcept {
  auto * state = static_cast<parser_dispatch_state *>(owner);
  state->done_called = true;
  return true;
}

bool on_parse_error(void * owner, const emel::text::jinja::events::parsing_error &) noexcept {
  auto * state = static_cast<parser_dispatch_state *>(owner);
  state->error_called = true;
  return true;
}

bool on_render_done(void * owner,
                    const emel::text::jinja::events::rendering_done &) noexcept {
  auto * state = static_cast<formatter_dispatch_state *>(owner);
  state->done_called = true;
  return true;
}

bool on_render_error(void * owner,
                     const emel::text::jinja::events::rendering_error &) noexcept {
  auto * state = static_cast<formatter_dispatch_state *>(owner);
  state->error_called = true;
  return true;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
  emel::text::jinja::program program{};
  parser_dispatch_state parse_state{};
  formatter_dispatch_state render_state{};

  emel::text::jinja::parser::action::context parse_ctx{};
  emel::text::jinja::parser::sm parser{parse_ctx};
  emel::text::jinja::formatter::action::context formatter_ctx{};
  emel::text::jinja::formatter::sm formatter{formatter_ctx};

  int32_t parse_err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t parse_error_pos = 0;
  const std::string_view input(reinterpret_cast<const char *>(data), size);
  const emel::text::jinja::event::parse::done_callback parse_done_cb{
      &parse_state,
      on_parse_done};
  const emel::text::jinja::event::parse::error_callback parse_error_cb{
      &parse_state,
      on_parse_error};
  const emel::text::jinja::event::parse parse_ev{
      input,
      program,
      parse_done_cb,
      parse_error_cb,
      parse_err,
      parse_error_pos,
  };
  (void)parser.process_event(parse_ev);

  std::array<char, 4096> output_buffer = {};
  size_t output_len = 0;
  bool output_truncated = false;
  int32_t render_err = static_cast<int32_t>(emel::text::jinja::formatter::error::none);
  size_t render_error_pos = 0;
  const emel::text::jinja::event::render::done_callback render_done_cb{
      &render_state,
      on_render_done};
  const emel::text::jinja::event::render::error_callback render_error_cb{
      &render_state,
      on_render_error};
  const emel::text::jinja::event::render render_ev{
      program,
      input,
      output_buffer[0],
      output_buffer.size(),
      render_done_cb,
      render_error_cb,
      nullptr,
      &output_len,
      &output_truncated,
      &render_err,
      &render_error_pos,
  };
  (void)formatter.process_event(render_ev);

  return 0;
}
