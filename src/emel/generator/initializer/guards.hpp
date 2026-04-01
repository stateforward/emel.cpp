#pragma once

#include "emel/generator/context.hpp"
#include "emel/generator/initializer/context.hpp"
#include "emel/generator/initializer/detail.hpp"
#include "emel/graph/errors.hpp"
#include "emel/memory/hybrid/errors.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/renderer/errors.hpp"

namespace emel::generator::initializer::guard {

namespace detail {

inline bool has_phase_success(const event::run & ev) noexcept {
  return ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

inline bool phase_rejected_without_code(const event::run & ev) noexcept {
  return !ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

constexpr int32_t conditioner_code(const emel::text::conditioner::error err) noexcept {
  return static_cast<int32_t>(err);
}

constexpr int32_t renderer_code(const emel::text::renderer::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t memory_code(const emel::memory::hybrid::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t graph_code(const emel::graph::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline bool conditioner_invalid_code(const int32_t code) noexcept {
  return code == conditioner_code(emel::text::conditioner::error::invalid_argument) ||
         code == conditioner_code(emel::text::conditioner::error::model_invalid) ||
         code == conditioner_code(emel::text::conditioner::error::capacity);
}

inline bool conditioner_backend_code(const int32_t code) noexcept {
  return code == conditioner_code(emel::text::conditioner::error::backend) ||
         code == conditioner_code(emel::text::conditioner::error::untracked);
}

inline bool renderer_invalid_code(const int32_t code) noexcept {
  return code == renderer_code(emel::text::renderer::error::invalid_request) ||
         code == renderer_code(emel::text::renderer::error::model_invalid);
}

inline bool renderer_backend_code(const int32_t code) noexcept {
  return code == renderer_code(emel::text::renderer::error::backend_error) ||
         code == renderer_code(emel::text::renderer::error::internal_error) ||
         code == renderer_code(emel::text::renderer::error::untracked);
}

inline bool memory_invalid_code(const int32_t code) noexcept {
  return code == memory_code(emel::memory::hybrid::error::invalid_request);
}

inline bool memory_backend_code(const int32_t code) noexcept {
  return code == memory_code(emel::memory::hybrid::error::backend_error) ||
         code == memory_code(emel::memory::hybrid::error::internal_error) ||
         code == memory_code(emel::memory::hybrid::error::out_of_memory) ||
         code == memory_code(emel::memory::hybrid::error::untracked);
}

inline bool graph_invalid_code(const int32_t code) noexcept {
  return code == graph_code(emel::graph::error::invalid_request);
}

inline bool graph_backend_code(const int32_t code) noexcept {
  return code == graph_code(emel::graph::error::assembler_failed) ||
         code == graph_code(emel::graph::error::processor_failed) ||
         code == graph_code(emel::graph::error::busy) ||
         code == graph_code(emel::graph::error::internal_error) ||
         code == graph_code(emel::graph::error::untracked);
}

}  // namespace detail

struct conditioner_bind_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct conditioner_bind_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::conditioner_invalid_code(ev.ctx.phase_code);
  }
};

struct conditioner_bind_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::conditioner_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::conditioner_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct renderer_initialize_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct renderer_initialize_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::renderer_invalid_code(ev.ctx.phase_code);
  }
};

struct renderer_initialize_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::renderer_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::renderer_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct memory_reserve_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct graph_reservation_present {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return ctx.generator.state.graph_reservation.node_count > 0u;
  }
};

struct graph_reservation_missing {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !graph_reservation_present{}(ev, ctx);
  }
};

struct memory_reserve_with_existing_graph {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return memory_reserve_ok{}(ev, ctx) && graph_reservation_present{}(ev, ctx);
  }
};

struct memory_reserve_with_missing_graph {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return memory_reserve_ok{}(ev, ctx) && graph_reservation_missing{}(ev, ctx);
  }
};

struct memory_reserve_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::memory_invalid_code(ev.ctx.phase_code);
  }
};

struct memory_reserve_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct graph_reserve_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct graph_reserve_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::graph_invalid_code(ev.ctx.phase_code);
  }
};

struct graph_reserve_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::graph_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::graph_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct uses_materialized_logits {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return ctx.generator.state.selection_mode == emel::generator::selection_mode::sample_logits;
  }
};

struct uses_preselected_argmax {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return ctx.generator.state.selection_mode ==
           emel::generator::selection_mode::preselected_argmax;
  }
};

struct sampler_configured {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.ctx.buffers_ready;
  }
};

struct sampler_config_failed {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !ev.ctx.buffers_ready;
  }
};

}  // namespace emel::generator::initializer::guard
