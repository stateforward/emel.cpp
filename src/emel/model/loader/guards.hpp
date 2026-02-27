#pragma once

#include "emel/model/loader/context.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::guard {

struct has_model_path {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !ev.request.model_path.empty();
  }
};

struct has_file_image {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.request.file_image != nullptr && ev.request.file_size > 0;
  }
};

struct valid_request {
  bool operator()(const event::load_runtime & ev, const action::context &) const noexcept {
    return has_model_path{}(ev) || has_file_image{}(ev);
  }
};

struct invalid_request {
  bool operator()(const event::load_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_request{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct phase_failed {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct should_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !ev.request.vocab_only;
  }
};

struct skip_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.request.vocab_only;
  }
};

struct can_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.load_weights);
  }
};

struct cannot_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !can_load_weights{}(ev);
  }
};

struct can_map_layers {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.map_layers);
  }
};

struct cannot_map_layers {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !can_map_layers{}(ev);
  }
};

struct skip_validate_structure {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !ev.request.check_tensors;
  }
};

struct can_validate_structure {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.request.check_tensors && static_cast<bool>(ev.request.validate_structure);
  }
};

struct cannot_validate_structure {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.request.check_tensors && !static_cast<bool>(ev.request.validate_structure);
  }
};

struct skip_validate_architecture {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !ev.request.validate_architecture;
  }
};

struct can_validate_architecture {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.request.validate_architecture &&
           static_cast<bool>(ev.request.validate_architecture_impl);
  }
};

struct cannot_validate_architecture {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return ev.request.validate_architecture &&
           !static_cast<bool>(ev.request.validate_architecture_impl);
  }
};

struct done_callback_present {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct done_callback_absent {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !done_callback_present{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

struct phase_ok_and_should_load_weights_and_can_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && should_load_weights{}(ev) && can_load_weights{}(ev);
  }
};

struct phase_ok_and_should_load_weights_and_cannot_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && should_load_weights{}(ev) && cannot_load_weights{}(ev);
  }
};

struct phase_ok_and_skip_load_weights {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && skip_load_weights{}(ev);
  }
};

struct phase_ok_and_can_map_layers {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && can_map_layers{}(ev);
  }
};

struct phase_ok_and_cannot_map_layers {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && cannot_map_layers{}(ev);
  }
};

struct phase_ok_and_skip_validate_structure {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && skip_validate_structure{}(ev);
  }
};

struct phase_ok_and_can_validate_structure {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && can_validate_structure{}(ev);
  }
};

struct phase_ok_and_cannot_validate_structure {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && cannot_validate_structure{}(ev);
  }
};

struct phase_ok_and_skip_validate_architecture {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && skip_validate_architecture{}(ev);
  }
};

struct phase_ok_and_can_validate_architecture {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && can_validate_architecture{}(ev);
  }
};

struct phase_ok_and_cannot_validate_architecture {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return phase_ok{}(ev) && cannot_validate_architecture{}(ev);
  }
};

}  // namespace emel::model::loader::guard
