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

inline bool error_is(const event::load_runtime & ev,
                     const emel::error::type expected) noexcept {
  return ev.ctx.err == expected;
}

struct error_none {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct error_invalid_request {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct error_parse_failed {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::parse_failed));
  }
};

struct error_backend_error {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct error_model_invalid {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct error_internal_error {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct error_untracked {
  bool operator()(const event::load_runtime & ev) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct error_unclassified_code {
  bool operator()(const event::load_runtime & ev) const noexcept {
    const emel::error::type err = ev.ctx.err;
    return err != emel::error::cast(error::none) &&
           err != emel::error::cast(error::invalid_request) &&
           err != emel::error::cast(error::parse_failed) &&
           err != emel::error::cast(error::backend_error) &&
           err != emel::error::cast(error::model_invalid) &&
           err != emel::error::cast(error::internal_error) &&
           err != emel::error::cast(error::untracked);
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

}  // namespace emel::model::loader::guard
