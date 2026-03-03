#pragma once

#include <cstring>
#include <string>
#include <string_view>

#include "emel/text/jinja/formatter/detail.hpp"
#include "emel/text/jinja/formatter/events.hpp"
#include "emel/text/jinja/formatter/errors.hpp"

namespace emel::text::jinja::formatter::action {

namespace runtime_detail {

inline const emel::text::jinja::value *
lookup_global(const emel::text::jinja::object_value * const globals,
             const std::string_view key) noexcept {
  if (!globals || globals->count == 0) {
    return nullptr;
  }

  for (size_t i = 0; i < globals->count; ++i) {
    const auto & entry = globals->entries[i];
    if (entry.key.type == emel::text::jinja::value_type::string &&
        entry.key.string_v.view == key) {
      return &entry.val;
    }
  }
  return nullptr;
}

inline void append_value(std::string & output,
                        const emel::text::jinja::value & value) {
  switch (value.type) {
    case emel::text::jinja::value_type::string:
      output += value.string_v.view;
      break;
    case emel::text::jinja::value_type::integer:
      output += std::to_string(value.int_v);
      break;
    case emel::text::jinja::value_type::floating:
      output += std::to_string(value.float_v);
      break;
    case emel::text::jinja::value_type::boolean:
      output += value.bool_v ? "true" : "false";
      break;
    default:
      break;
  }
}

inline void render_node(std::string & output,
                       const emel::text::jinja::ast_node * const node,
                       const emel::text::jinja::object_value * const globals) {
  if (!node) {
    return;
  }

  if (const auto * text = dynamic_cast<const emel::text::jinja::string_literal *>(node);
      text != nullptr) {
    output += text->value;
    return;
  }
  if (const auto * comment = dynamic_cast<const emel::text::jinja::comment_statement *>(node);
      comment != nullptr) {
    (void)comment;
    return;
  }
  if (const auto * id = dynamic_cast<const emel::text::jinja::identifier *>(node);
      id != nullptr) {
    const auto * value = lookup_global(globals, id->name);
    if (value != nullptr) {
      append_value(output, *value);
    }
    return;
  }
  if (dynamic_cast<const emel::text::jinja::noop_statement *>(node) != nullptr) {
    return;
  }
}

inline std::string render_program(const emel::text::jinja::program & program,
                                 const emel::text::jinja::object_value * const globals) {
  std::string output;
  output.reserve(program.body.size() * 16);
  for (const auto & node : program.body) {
    render_node(output, node.get(), globals);
  }
  return output;
}

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

}  // namespace runtime_detail

struct reject_invalid_render {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::mark_error(runtime_ev.ctx, error::invalid_request, false, 0);
  }
};

struct begin_render {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::reset_result(runtime_ev.ctx);
  }
};

struct mark_empty_output {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::mark_done(runtime_ev.ctx, 0, false);
  }
};

struct copy_source_text {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    const auto & request = runtime_ev.request;
    const auto rendered = runtime_detail::render_program(request.program, request.globals);
    if (rendered.size() > request.output_capacity) {
      detail::mark_error(runtime_ev.ctx, error::invalid_request, true, 0);
      return;
    }
    if (!rendered.empty()) {
      std::memcpy(&request.output, rendered.data(), rendered.size());
    }
    detail::mark_done(runtime_ev.ctx, rendered.size(), false);
  }
};

struct mark_capacity_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::mark_error(runtime_ev.ctx, error::invalid_request, true, 0);
  }
};

struct dispatch_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::emit_done(runtime_ev.request, runtime_ev.ctx);
  }
};

struct dispatch_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::emit_error(runtime_ev.request, runtime_ev.ctx);
  }
};

struct on_unexpected {
  void operator()(const event::render_runtime & ev) const noexcept {
    detail::mark_error(ev.ctx, error::invalid_request, true, 0);
    detail::emit_error(ev.request, ev.ctx);
  }

  template <class event_type>
  void operator()(const event_type &) const noexcept {
  }
};

inline constexpr reject_invalid_render reject_invalid_render{};
inline constexpr begin_render begin_render{};
inline constexpr mark_empty_output mark_empty_output{};
inline constexpr copy_source_text copy_source_text{};
inline constexpr mark_capacity_error mark_capacity_error{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::jinja::formatter::action
