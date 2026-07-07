#pragma once

#include "emel/text/generator/layer/events.hpp"

namespace emel::text::generator::layer::guard {

template <event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route>
struct guard_scalar_attention_route {
  bool operator()(const event::scalar_run &ev) const noexcept {
    return ev.residual == event::residual_route::attention &&
           ev.qk_norm == qk_route && ev.v_norm == v_route;
  }
};

struct guard_scalar_shortconv_route {
  bool operator()(const event::scalar_run &ev) const noexcept {
    return ev.residual == event::residual_route::shortconv;
  }
};

template <event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route>
struct guard_chunk4_attention_route {
  bool operator()(const event::chunk4_run &ev) const noexcept {
    return ev.residual == event::residual_route::attention &&
           ev.qk_norm == qk_route && ev.v_norm == v_route;
  }
};

struct guard_chunk4_shortconv_route {
  bool operator()(const event::chunk4_run &ev) const noexcept {
    return ev.residual == event::residual_route::shortconv;
  }
};

template <event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route>
struct guard_chunk8_attention_route {
  bool operator()(const event::chunk8_run &ev) const noexcept {
    return ev.residual == event::residual_route::attention &&
           ev.qk_norm == qk_route && ev.v_norm == v_route;
  }
};

struct guard_chunk8_shortconv_route {
  bool operator()(const event::chunk8_run &ev) const noexcept {
    return ev.residual == event::residual_route::shortconv;
  }
};

struct guard_residual_ok {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  bool operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    return (*this)(ev.event_);
  }

  bool operator()(const event::scalar_run &ev) const noexcept {
    return ev.residual_ok;
  }

  bool operator()(const event::chunk4_run &ev) const noexcept {
    return ev.residual_ok;
  }

  bool operator()(const event::chunk8_run &ev) const noexcept {
    return ev.residual_ok;
  }
};

struct guard_residual_failed {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  bool operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    return (*this)(ev.event_);
  }

  bool operator()(const event::scalar_run &ev) const noexcept {
    return !guard_residual_ok{}(ev);
  }

  bool operator()(const event::chunk4_run &ev) const noexcept {
    return !guard_residual_ok{}(ev);
  }

  bool operator()(const event::chunk8_run &ev) const noexcept {
    return !guard_residual_ok{}(ev);
  }
};

struct guard_feed_forward_ok {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  bool operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    return (*this)(ev.event_);
  }

  bool operator()(const event::scalar_run &ev) const noexcept {
    return ev.feed_forward_ok;
  }

  bool operator()(const event::chunk4_run &ev) const noexcept {
    return ev.feed_forward_ok;
  }

  bool operator()(const event::chunk8_run &ev) const noexcept {
    return ev.feed_forward_ok;
  }
};

struct guard_feed_forward_failed {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  bool operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    return (*this)(ev.event_);
  }

  bool operator()(const event::scalar_run &ev) const noexcept {
    return !guard_feed_forward_ok{}(ev);
  }

  bool operator()(const event::chunk4_run &ev) const noexcept {
    return !guard_feed_forward_ok{}(ev);
  }

  bool operator()(const event::chunk8_run &ev) const noexcept {
    return !guard_feed_forward_ok{}(ev);
  }
};

} // namespace emel::text::generator::layer::guard
