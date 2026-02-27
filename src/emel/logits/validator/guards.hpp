#pragma once

#include "emel/logits/validator/events.hpp"

namespace emel::logits::validator::guard {

struct request_has_valid_sizes {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.request.vocab_size > 0 &&
           ev.request.candidate_capacity >= ev.request.vocab_size;
  }
};

struct valid_request {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return request_has_valid_sizes{}(ev);
  }
};

struct invalid_request {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return !valid_request{}(ev);
  }
};

struct prepare_has_more {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.ctx.build_cursor < ev.request.vocab_size;
  }
};

struct prepare_done {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.ctx.build_cursor >= ev.request.vocab_size;
  }
};

struct max_scan_has_more {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.ctx.max_cursor < ev.ctx.candidate_count;
  }
};

struct max_scan_done {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.ctx.max_cursor >= ev.ctx.candidate_count;
  }
};

struct current_score_exceeds_max {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return (&ev.request.candidate_scores)[ev.ctx.max_cursor] > ev.ctx.max_score;
  }
};

struct current_score_not_exceeds_max {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return !current_score_exceeds_max{}(ev);
  }
};

struct normalize_has_more {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.ctx.normalize_cursor < ev.ctx.candidate_count;
  }
};

struct normalize_done {
  bool operator()(const event::build_runtime & ev) const noexcept {
    return ev.ctx.normalize_cursor >= ev.ctx.candidate_count;
  }
};

}  // namespace emel::logits::validator::guard
