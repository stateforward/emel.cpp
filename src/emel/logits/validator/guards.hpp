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

}  // namespace emel::logits::validator::guard
