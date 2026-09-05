#pragma once

#include "emel/model/needle/request/context.hpp"
#include "emel/model/needle/request/events.hpp"

namespace emel::model::needle::request::guard {

inline bool error_is(const emel::error::type value,
                     const error expected) noexcept {
  return value == emel::error::cast(expected);
}

struct guard_configure_valid {
  bool operator()(const event::configure_run &ev,
                  const action::context &ctx) const noexcept {
    return ctx.bound.has_tokenizer && ev.request.system.size() <=
                                      action::k_max_system_bytes &&
           !ev.request.tools_json.empty() &&
           ev.request.tools_json.size() <= action::k_max_tools_bytes &&
           action::validate_tools_json(ev.request.tools_json);
  }
};

struct guard_configure_invalid {
  bool operator()(const event::configure_run &ev,
                  const action::context &ctx) const noexcept {
    return !guard_configure_valid{}(ev, ctx);
  }
};

struct guard_reset_valid {
  bool operator()(const event::reset_run &,
                  const action::context &ctx) const noexcept {
    return ctx.assets_ready && ctx.configured;
  }
};

struct guard_reset_invalid {
  bool operator()(const event::reset_run &ev,
                  const action::context &ctx) const noexcept {
    return !guard_reset_valid{}(ev, ctx);
  }
};

struct guard_complete_valid {
  bool operator()(const event::complete_run &ev,
                  const action::context &ctx) const noexcept {
    return ctx.assets_ready && ctx.configured && ctx.reset_ready &&
           !ev.request.query.empty() &&
           ev.request.query.size() <= action::k_max_query_bytes &&
           ev.request.max_new_tokens > 0u &&
           ev.request.max_new_tokens < ctx.generated_ids.size() &&
           ctx.prompt_ids.size() >= 2u &&
           static_cast<uint64_t>(ev.request.max_new_tokens) <
               ctx.bound.geo.max_seq_len;
  }
};

struct guard_complete_invalid {
  bool operator()(const event::complete_run &ev,
                  const action::context &ctx) const noexcept {
    return !guard_complete_valid{}(ev, ctx);
  }
};

template <class runtime_event>
inline const auto &origin_event(const runtime_event &ev) noexcept {
  if constexpr (requires { ev.event_; })
    return ev.event_;
  else
    return ev;
}

struct guard_error_none {
  template <class runtime_event>
  bool operator()(const runtime_event &ev) const noexcept {
    return error_is(origin_event(ev).ctx.err, error::none);
  }
};

struct guard_error_present {
  template <class runtime_event>
  bool operator()(const runtime_event &ev) const noexcept {
    return !guard_error_none{}(ev);
  }
};

struct guard_generation_continues {
  template <class runtime_event>
  bool operator()(const runtime_event &wrapped,
                  const action::context &ctx) const noexcept {
    const auto &ev = origin_event(wrapped);
    return guard_error_none{}(ev) &&
           ctx.generated_id_count + 1u < ev.request.max_new_tokens &&
           ctx.generated_ids[ctx.generated_id_count] != ctx.vocab->eos_id;
  }
};

struct guard_generation_stops {
  template <class runtime_event>
  bool operator()(const runtime_event &wrapped,
                  const action::context &ctx) const noexcept {
    const auto &ev = origin_event(wrapped);
    return guard_error_none{}(ev) &&
           (ctx.generated_id_count + 1u >= ev.request.max_new_tokens ||
            ctx.generated_ids[ctx.generated_id_count] == ctx.vocab->eos_id);
  }
};

struct guard_callback_present {
  template <class runtime_event>
  bool operator()(const runtime_event &ev) const noexcept {
    return ev.request.on_done != nullptr;
  }
};

struct guard_callback_absent {
  template <class runtime_event>
  bool operator()(const runtime_event &ev) const noexcept {
    return ev.request.on_done == nullptr;
  }
};

struct guard_error_callback_present {
  template <class runtime_event>
  bool operator()(const runtime_event &ev) const noexcept {
    return ev.request.on_error != nullptr;
  }
};

struct guard_error_callback_absent {
  template <class runtime_event>
  bool operator()(const runtime_event &ev) const noexcept {
    return ev.request.on_error == nullptr;
  }
};

} // namespace emel::model::needle::request::guard
