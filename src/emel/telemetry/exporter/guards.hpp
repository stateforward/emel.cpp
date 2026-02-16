#pragma once

#include "emel/telemetry/exporter/actions.hpp"

namespace emel::telemetry::exporter::guard {

struct has_batch {
  template <class TEvent>
  bool operator()(const TEvent &, const action::context & ctx) const noexcept {
    return ctx.batch_count > 0;
  }
};

struct no_batch {
  template <class TEvent>
  bool operator()(const TEvent &, const action::context & ctx) const noexcept {
    return ctx.batch_count <= 0;
  }
};

}  // namespace emel::telemetry::exporter::guard
