#pragma once

#include "emel/emel.h"
#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::guard {

struct can_map_parser {
  bool operator()(const event::load & ev) const {
    return ev.parser_map != nullptr &&
           ev.parser_map->entries != nullptr &&
           ev.parser_map->count > 0;
  }
};

struct cannot_map_parser {
  bool operator()(const event::load & ev) const { return !can_map_parser{}(ev); }
};

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

struct has_request {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr;
  }
};

struct can_parse {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr &&
           ctx.parser_sm != nullptr &&
           ctx.parser_dispatch != nullptr;
  }
};

struct cannot_parse {
  bool operator()(const action::context & ctx) const noexcept {
    return !can_parse{}(ctx);
  }
};

struct should_load_weights {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && !ctx.request->vocab_only;
  }
};

struct skip_weights {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->vocab_only;
  }
};

struct can_load_weights {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr &&
           ctx.request->dispatch_load_weights != nullptr &&
           ctx.request->weight_loader_sm != nullptr;
  }
};

struct cannot_load_weights {
  bool operator()(const action::context & ctx) const noexcept {
    return !can_load_weights{}(ctx);
  }
};

struct can_map_layers {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->map_layers != nullptr;
  }
};

struct cannot_map_layers {
  bool operator()(const action::context & ctx) const noexcept {
    return !can_map_layers{}(ctx);
  }
};

struct skip_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && !ctx.request->check_tensors;
  }
};

struct can_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr &&
           ctx.request->check_tensors &&
           ctx.request->validate_structure != nullptr;
  }
};

struct cannot_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr &&
           ctx.request->check_tensors &&
           ctx.request->validate_structure == nullptr;
  }
};

struct has_arch_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->validate_architecture;
  }
};

struct no_arch_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && !ctx.request->validate_architecture;
  }
};

struct has_arch_validate_and_can_validate_architecture {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr &&
           ctx.request->validate_architecture &&
           ctx.request->validate_architecture_impl != nullptr;
  }
};

struct has_arch_validate_and_cannot_validate_architecture {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr &&
           ctx.request->validate_architecture &&
           ctx.request->validate_architecture_impl == nullptr;
  }
};

struct phase_ok_and_can_parse {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && can_parse{}(ctx);
  }
};

struct phase_ok_and_cannot_parse {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && cannot_parse{}(ctx);
  }
};

struct phase_ok_and_should_load_weights_and_can_load {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && should_load_weights{}(ctx) && can_load_weights{}(ctx);
  }
};

struct phase_ok_and_should_load_weights_and_cannot_load {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && should_load_weights{}(ctx) && cannot_load_weights{}(ctx);
  }
};

struct phase_ok_and_skip_weights_and_skip_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && skip_weights{}(ctx) && skip_validate_structure{}(ctx);
  }
};

struct phase_ok_and_skip_weights_and_can_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && skip_weights{}(ctx) && can_validate_structure{}(ctx);
  }
};

struct phase_ok_and_skip_weights_and_cannot_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && skip_weights{}(ctx) && cannot_validate_structure{}(ctx);
  }
};

struct phase_ok_and_can_map_layers {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && can_map_layers{}(ctx);
  }
};

struct phase_ok_and_cannot_map_layers {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && cannot_map_layers{}(ctx);
  }
};

struct phase_ok_and_skip_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && skip_validate_structure{}(ctx);
  }
};

struct phase_ok_and_can_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && can_validate_structure{}(ctx);
  }
};

struct phase_ok_and_cannot_validate_structure {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && cannot_validate_structure{}(ctx);
  }
};

struct phase_ok_and_has_arch_validate_and_can_validate_architecture {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && has_arch_validate_and_can_validate_architecture{}(ctx);
  }
};

struct phase_ok_and_has_arch_validate_and_cannot_validate_architecture {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && has_arch_validate_and_cannot_validate_architecture{}(ctx);
  }
};

struct phase_ok_and_no_arch_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && no_arch_validate{}(ctx);
  }
};

}  // namespace emel::model::loader::guard
