#pragma once

#include "emel/logits/sampler/context.hpp"
#include "emel/logits/sampler/events.hpp"

namespace emel::logits::sampler::guard {

struct valid_config {
  bool operator()(const event::configure_runtime & ev) const noexcept {
    return ev.request.sampler_count > 0;
  }
};

struct invalid_config {
  bool operator()(const event::configure_runtime & ev) const noexcept {
    return !valid_config{}(ev);
  }
};

struct request_has_valid_sizes {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return ev.request.vocab_size > 0 &&
           ev.request.candidate_capacity >= ev.request.vocab_size;
  }
};

struct context_has_valid_sampler_table {
  bool operator()(const event::sample_logits_runtime &, const action::context & ctx) const noexcept {
    return ctx.sampler_count > 0 && ctx.sampler_fns != nullptr;
  }
};

struct valid_request {
  bool operator()(const event::sample_logits_runtime & ev, const action::context & ctx) const noexcept {
    return request_has_valid_sizes{}(ev) && context_has_valid_sampler_table{}(ev, ctx);
  }
};

struct invalid_request {
  bool operator()(const event::sample_logits_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_request{}(ev, ctx);
  }
};

struct preselected_token_valid {
  bool operator()(const event::sample_preselected_runtime & ev) const noexcept {
    return ev.request.vocab_size > 0 &&
           ev.request.selected_token_out >= 0 &&
           ev.request.selected_token_out < ev.request.vocab_size;
  }
};

struct preselected_token_invalid {
  bool operator()(const event::sample_preselected_runtime & ev) const noexcept {
    return !preselected_token_valid{}(ev);
  }
};

struct has_more_samplers {
  bool operator()(const event::sample_logits_runtime & ev, const action::context & ctx) const noexcept {
    return ev.ctx.sampler_index < ctx.sampler_count;
  }
};

struct no_more_samplers {
  bool operator()(const event::sample_logits_runtime & ev, const action::context & ctx) const noexcept {
    return ev.ctx.sampler_index >= ctx.sampler_count;
  }
};

struct sampler_fn_available {
  bool operator()(const event::sample_logits_runtime & ev, const action::context & ctx) const noexcept {
    return static_cast<bool>(ctx.sampler_fns[ev.ctx.sampler_index]);
  }
};

struct sampler_fn_missing {
  bool operator()(const event::sample_logits_runtime & ev, const action::context & ctx) const noexcept {
    return !sampler_fn_available{}(ev, ctx);
  }
};

struct sampler_call_succeeded {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return ev.ctx.sampler_call_error == emel::error::cast(error::none);
  }
};

struct sampler_call_failed {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return ev.ctx.sampler_call_error != emel::error::cast(error::none);
  }
};

struct candidate_count_valid {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return ev.ctx.candidate_count > 0 && ev.ctx.candidate_count <= ev.request.vocab_size;
  }
};

struct candidate_count_invalid {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return !candidate_count_valid{}(ev);
  }
};

struct sampler_call_succeeded_with_valid_candidate_count {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return sampler_call_succeeded{}(ev) && candidate_count_valid{}(ev);
  }
};

struct sampler_call_succeeded_with_invalid_candidate_count {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return sampler_call_succeeded{}(ev) && candidate_count_invalid{}(ev);
  }
};

struct selected_token_valid {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return ev.request.selected_token_out >= 0 &&
           ev.request.selected_token_out < ev.request.vocab_size;
  }
};

struct selected_token_missing_or_invalid {
  bool operator()(const event::sample_logits_runtime & ev) const noexcept {
    return !selected_token_valid{}(ev);
  }
};

}  // namespace emel::logits::sampler::guard
