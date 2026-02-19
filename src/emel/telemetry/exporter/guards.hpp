#pragma once

#include "emel/telemetry/exporter/actions.hpp"

namespace emel::telemetry::exporter::guard {

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

struct has_batch {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.batch_count > 0;
  }
};

struct no_batch {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.batch_count <= 0;
  }
};

struct phase_ok_and_has_batch {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && has_batch{}(ctx);
  }
};

struct phase_ok_and_no_batch {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && no_batch{}(ctx);
  }
};

}  // namespace emel::telemetry::exporter::guard
