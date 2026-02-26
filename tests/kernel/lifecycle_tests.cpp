#include <doctest/doctest.h>

#include <concepts>

#include "emel/kernel/actions.hpp"
#include "emel/kernel/any.hpp"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/guards.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/kernel/x86_64/actions.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "emel/kernel/x86_64/events.hpp"
#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/aarch64/events.hpp"
#include "emel/kernel/wasm/actions.hpp"
#include "emel/kernel/wasm/sm.hpp"
#include "emel/kernel/wasm/events.hpp"
#include "emel/kernel/cuda/actions.hpp"
#include "emel/kernel/cuda/sm.hpp"
#include "emel/kernel/cuda/events.hpp"
#include "emel/kernel/metal/actions.hpp"
#include "emel/kernel/metal/sm.hpp"
#include "emel/kernel/metal/events.hpp"
#include "emel/kernel/vulkan/actions.hpp"
#include "emel/kernel/vulkan/sm.hpp"
#include "emel/kernel/vulkan/events.hpp"

namespace {

using kernel_sm = emel::kernel::sm;
using x86_64_sm = emel::kernel::x86_64::sm;
using aarch64_sm = emel::kernel::aarch64::sm;
using wasm_sm = emel::kernel::wasm::sm;
using cuda_sm = emel::kernel::cuda::sm;
using metal_sm = emel::kernel::metal::sm;
using vulkan_sm = emel::kernel::vulkan::sm;
using kernel_dispatch_event = emel::kernel::event::dispatch_scaffold;
using x86_64_dispatch_event = emel::kernel::x86_64::event::dispatch_scaffold;
using aarch64_dispatch_event = emel::kernel::aarch64::event::dispatch_scaffold;
using wasm_dispatch_event = emel::kernel::wasm::event::dispatch_scaffold;
using cuda_dispatch_event = emel::kernel::cuda::event::dispatch_scaffold;
using metal_dispatch_event = emel::kernel::metal::event::dispatch_scaffold;
using vulkan_dispatch_event = emel::kernel::vulkan::event::dispatch_scaffold;

template <class machine_type, class event_type>
concept has_public_process_event = requires(machine_type & machine, const event_type & ev) {
  { machine.process_event(ev) } -> std::convertible_to<bool>;
};

template <class machine_type>
void check_backend_op_paths(machine_type & machine,
                            const emel::kernel::event::op_dup & op_dup_ok,
                            const emel::kernel::event::op_dup & op_dup_invalid,
                            const emel::kernel::event::op_add & op_add_ok,
                            const emel::kernel::event::op_add & op_add_invalid,
                            const emel::kernel::event::op_mul & op_mul_ok,
                            const emel::kernel::event::op_mul & op_mul_invalid,
                            const emel::kernel::event::op_mul_mat & op_mul_mat_ok,
                            const emel::kernel::event::op_mul_mat & op_mul_mat_invalid,
                            const emel::kernel::event::op_rope & op_rope_ok,
                            const emel::kernel::event::op_rope & op_rope_invalid,
                            const emel::kernel::event::op_soft_max & op_soft_max_ok,
                            const emel::kernel::event::op_soft_max & op_soft_max_invalid) {
  CHECK(machine.process_event(op_dup_ok));
  CHECK_FALSE(machine.process_event(op_dup_invalid));

  CHECK(machine.process_event(op_add_ok));
  CHECK_FALSE(machine.process_event(op_add_invalid));

  CHECK(machine.process_event(op_mul_ok));
  CHECK_FALSE(machine.process_event(op_mul_invalid));

  CHECK(machine.process_event(op_mul_mat_ok));
  CHECK_FALSE(machine.process_event(op_mul_mat_invalid));

  CHECK(machine.process_event(op_rope_ok));
  CHECK_FALSE(machine.process_event(op_rope_invalid));

  CHECK(machine.process_event(op_soft_max_ok));
  CHECK_FALSE(machine.process_event(op_soft_max_invalid));
}

template <class event_type>
event_type make_smoke_op_event() {
  event_type ev{};
  static float src0[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  static float src1[4] = {3.0f, 2.0f, 1.0f, 0.0f};
  static float src2[4] = {4.0f, 5.0f, 6.0f, 7.0f};
  static float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if constexpr (requires { ev.src0; }) {
    ev.src0 = src0;
  }
  if constexpr (requires { ev.src1; }) {
    ev.src1 = src1;
  }
  if constexpr (requires { ev.src2; }) {
    ev.src2 = src2;
  }
  if constexpr (requires { ev.dst; }) {
    ev.dst = dst;
  }
  if constexpr (requires { ev.element_count; }) {
    ev.element_count = 4;
  }
  if constexpr (requires { ev.row_count; }) {
    ev.row_count = 2;
  }
  if constexpr (requires { ev.col_count; }) {
    ev.col_count = 2;
  }
  if constexpr (requires { ev.token_count; }) {
    ev.token_count = 4;
  }
  if constexpr (requires { ev.dim0; }) {
    ev.dim0 = 1;
  }
  if constexpr (requires { ev.dim1; }) {
    ev.dim1 = 1;
  }
  if constexpr (requires { ev.dim2; }) {
    ev.dim2 = 1;
  }
  if constexpr (requires { ev.dim3; }) {
    ev.dim3 = 1;
  }

  return ev;
}

}  // namespace

TEST_CASE("kernel_backend_scaffolds_accept_scaffold_event") {
  const emel::kernel::event::scaffold event{};

  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};
  kernel_sm kernel_machine{};
  emel::kernel::any any_machine{};

  CHECK(x86_64_machine.process_event(event));
  CHECK(aarch64_machine.process_event(event));
  CHECK(wasm_machine.process_event(event));
  CHECK(cuda_machine.process_event(event));
  CHECK(metal_machine.process_event(event));
  CHECK(vulkan_machine.process_event(event));
  CHECK(kernel_machine.process_event(event));
  CHECK(any_machine.process_event(event));
}

TEST_CASE("kernel_backend_scaffolds_expose_explicit_op_transitions") {
  const float src0[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  const float src1[4] = {3.0f, 2.0f, 1.0f, 0.0f};
  float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const emel::kernel::event::op_dup op_dup_ok{
    .src0 = src0,
    .dst = dst,
    .element_count = 4,
  };
  const emel::kernel::event::op_add op_add_ok{
    .src0 = src0,
    .src1 = src1,
    .dst = dst,
    .element_count = 4,
  };
  const emel::kernel::event::op_mul op_mul_ok{
    .src0 = src0,
    .src1 = src1,
    .dst = dst,
    .element_count = 4,
  };
  const emel::kernel::event::op_mul_mat op_mul_mat_ok{
    .src0 = src0,
    .src1 = src1,
    .dst = dst,
    .row_count = 2,
    .col_count = 2,
  };
  const emel::kernel::event::op_rope op_rope_ok{
    .src0 = src0,
    .dst = dst,
    .token_count = 4,
  };
  const emel::kernel::event::op_soft_max op_soft_max_ok{
    .src0 = src0,
    .dst = dst,
    .element_count = 4,
  };

  const emel::kernel::event::op_dup op_dup_invalid{
    .src0 = nullptr,
    .dst = dst,
    .element_count = 4,
  };
  const emel::kernel::event::op_add op_add_invalid{
    .src0 = src0,
    .src1 = nullptr,
    .dst = dst,
    .element_count = 4,
  };
  const emel::kernel::event::op_mul op_mul_invalid{
    .src0 = src0,
    .src1 = nullptr,
    .dst = dst,
    .element_count = 4,
  };
  const emel::kernel::event::op_mul_mat op_mul_mat_invalid{
    .src0 = src0,
    .src1 = src1,
    .dst = dst,
    .row_count = 0,
    .col_count = 2,
  };
  const emel::kernel::event::op_rope op_rope_invalid{
    .src0 = src0,
    .dst = nullptr,
    .token_count = 4,
  };
  const emel::kernel::event::op_soft_max op_soft_max_invalid{
    .src0 = src0,
    .dst = dst,
    .element_count = 0,
  };

  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};
  kernel_sm kernel_machine{};
  emel::kernel::any any_machine{};

  check_backend_op_paths(x86_64_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid, op_rope_ok,
                         op_rope_invalid, op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(aarch64_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid, op_rope_ok,
                         op_rope_invalid, op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(wasm_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid, op_rope_ok,
                         op_rope_invalid, op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(cuda_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid, op_rope_ok,
                         op_rope_invalid, op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(metal_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid, op_rope_ok,
                         op_rope_invalid, op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(vulkan_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid, op_rope_ok,
                         op_rope_invalid, op_soft_max_ok, op_soft_max_invalid);

  CHECK(any_machine.process_event(op_add_ok));
  CHECK(kernel_machine.process_event(op_add_ok));
  CHECK_FALSE(kernel_machine.process_event(op_add_invalid));
  CHECK(kernel_machine.process_event(emel::kernel::event::scaffold{}));
}

TEST_CASE("kernel_scaffold_actions_and_guards_cover_all_outcomes") {
  emel::kernel::action::context ctx{};
  emel::kernel::event::scaffold request{};
  emel::kernel::event::scaffold_ctx runtime{};
  emel::kernel::event::dispatch_scaffold dispatch{request, runtime};

  emel::kernel::action::begin_dispatch(dispatch, ctx);
  CHECK(runtime.err == static_cast<int32_t>(emel::error::cast(emel::kernel::error::none)));
  CHECK(runtime.primary_outcome == emel::kernel::event::phase_outcome::unknown);
  CHECK(runtime.secondary_outcome == emel::kernel::event::phase_outcome::unknown);
  CHECK(runtime.tertiary_outcome == emel::kernel::event::phase_outcome::unknown);
  CHECK(ctx.dispatch_generation == 1);

  emel::kernel::action::request_primary(dispatch, ctx);
  CHECK(runtime.primary_outcome == emel::kernel::event::phase_outcome::done);
  CHECK(runtime.err == static_cast<int32_t>(emel::error::cast(emel::kernel::error::none)));

  emel::kernel::action::request_secondary(dispatch, ctx);
  CHECK(runtime.secondary_outcome == emel::kernel::event::phase_outcome::done);
  CHECK(runtime.err == static_cast<int32_t>(emel::error::cast(emel::kernel::error::none)));

  emel::kernel::action::request_tertiary(dispatch, ctx);
  CHECK(runtime.tertiary_outcome == emel::kernel::event::phase_outcome::done);
  CHECK(runtime.err == static_cast<int32_t>(emel::error::cast(emel::kernel::error::none)));

  CHECK(emel::kernel::guard::valid_dispatch{}(dispatch, ctx));
  CHECK(emel::kernel::guard::phase_ok{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::phase_failed{}(dispatch, ctx));

  CHECK(emel::kernel::guard::primary_done{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::primary_unsupported{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::primary_failed{}(dispatch, ctx));

  CHECK(emel::kernel::guard::secondary_done{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::secondary_unsupported{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::secondary_failed{}(dispatch, ctx));

  CHECK(emel::kernel::guard::tertiary_done{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::tertiary_unsupported{}(dispatch, ctx));
  CHECK_FALSE(emel::kernel::guard::tertiary_failed{}(dispatch, ctx));

  runtime.primary_outcome = emel::kernel::event::phase_outcome::unsupported;
  CHECK(emel::kernel::guard::primary_unsupported{}(dispatch, ctx));
  runtime.primary_outcome = emel::kernel::event::phase_outcome::failed;
  CHECK(emel::kernel::guard::primary_failed{}(dispatch, ctx));

  runtime.secondary_outcome = emel::kernel::event::phase_outcome::unsupported;
  CHECK(emel::kernel::guard::secondary_unsupported{}(dispatch, ctx));
  runtime.secondary_outcome = emel::kernel::event::phase_outcome::failed;
  CHECK(emel::kernel::guard::secondary_failed{}(dispatch, ctx));

  runtime.tertiary_outcome = emel::kernel::event::phase_outcome::unsupported;
  CHECK(emel::kernel::guard::tertiary_unsupported{}(dispatch, ctx));
  runtime.tertiary_outcome = emel::kernel::event::phase_outcome::failed;
  CHECK(emel::kernel::guard::tertiary_failed{}(dispatch, ctx));

  emel::kernel::action::mark_unsupported(dispatch, ctx);
  CHECK(runtime.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::unsupported_op)));
  CHECK(emel::kernel::guard::phase_failed{}(dispatch, ctx));

  emel::kernel::action::dispatch_done(dispatch, ctx);
  emel::kernel::action::dispatch_error(dispatch, ctx);

  runtime.err = static_cast<int32_t>(emel::error::cast(emel::kernel::error::none));
  emel::kernel::action::on_unexpected(dispatch, ctx);
  CHECK(runtime.err == static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
}

TEST_CASE("kernel_backend_unexpected_actions_mark_backend_error") {
  const emel::kernel::event::scaffold request{};

  emel::kernel::x86_64::action::context x86_64_ctx{};
  emel::kernel::aarch64::action::context aarch64_ctx{};
  emel::kernel::wasm::action::context wasm_ctx{};
  emel::kernel::cuda::action::context cuda_ctx{};
  emel::kernel::metal::action::context metal_ctx{};
  emel::kernel::vulkan::action::context vulkan_ctx{};

  emel::kernel::x86_64::event::dispatch_ctx x86_64_dispatch_ctx{};
  emel::kernel::aarch64::event::dispatch_ctx aarch64_dispatch_ctx{};
  emel::kernel::wasm::event::dispatch_ctx wasm_dispatch_ctx{};
  emel::kernel::cuda::event::dispatch_ctx cuda_dispatch_ctx{};
  emel::kernel::metal::event::dispatch_ctx metal_dispatch_ctx{};
  emel::kernel::vulkan::event::dispatch_ctx vulkan_dispatch_ctx{};

  const emel::kernel::x86_64::event::dispatch_scaffold x86_64_dispatch{request,
                                                                         x86_64_dispatch_ctx};
  const emel::kernel::aarch64::event::dispatch_scaffold aarch64_dispatch{request,
                                                                           aarch64_dispatch_ctx};
  const emel::kernel::wasm::event::dispatch_scaffold wasm_dispatch{request, wasm_dispatch_ctx};
  const emel::kernel::cuda::event::dispatch_scaffold cuda_dispatch{request, cuda_dispatch_ctx};
  const emel::kernel::metal::event::dispatch_scaffold metal_dispatch{request,
                                                                      metal_dispatch_ctx};
  const emel::kernel::vulkan::event::dispatch_scaffold vulkan_dispatch{
    request,
    vulkan_dispatch_ctx,
  };

  emel::kernel::x86_64::action::on_unexpected(x86_64_dispatch, x86_64_ctx);
  emel::kernel::aarch64::action::on_unexpected(aarch64_dispatch, aarch64_ctx);
  emel::kernel::wasm::action::on_unexpected(wasm_dispatch, wasm_ctx);
  emel::kernel::cuda::action::on_unexpected(cuda_dispatch, cuda_ctx);
  emel::kernel::metal::action::on_unexpected(metal_dispatch, metal_ctx);
  emel::kernel::vulkan::action::on_unexpected(vulkan_dispatch, vulkan_ctx);

  CHECK(x86_64_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(aarch64_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(wasm_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(cuda_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(metal_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(vulkan_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));

  CHECK(x86_64_dispatch_ctx.outcome == emel::kernel::x86_64::events::phase_outcome::failed);
  CHECK(aarch64_dispatch_ctx.outcome == emel::kernel::aarch64::events::phase_outcome::failed);
  CHECK(wasm_dispatch_ctx.outcome == emel::kernel::wasm::events::phase_outcome::failed);
  CHECK(cuda_dispatch_ctx.outcome == emel::kernel::cuda::events::phase_outcome::failed);
  CHECK(metal_dispatch_ctx.outcome == emel::kernel::metal::events::phase_outcome::failed);
  CHECK(vulkan_dispatch_ctx.outcome == emel::kernel::vulkan::events::phase_outcome::failed);
}

TEST_CASE("kernel_backend_scaffolds_cover_all_ggml_ops") {
  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};
  kernel_sm kernel_machine{};
  emel::kernel::any any_machine{};

#define EMEL_KERNEL_CHECK_ALL_BACKENDS(op_name)                                             \
  CHECK(x86_64_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>())); \
  CHECK(aarch64_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>())); \
  CHECK(wasm_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>()));    \
  CHECK(cuda_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>()));    \
  CHECK(metal_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>()));   \
  CHECK(vulkan_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>()));  \
  CHECK(kernel_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>()));  \
  CHECK(any_machine.process_event(make_smoke_op_event<emel::kernel::event::op_name>()));
  EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_CHECK_ALL_BACKENDS)
#undef EMEL_KERNEL_CHECK_ALL_BACKENDS
}

TEST_CASE("kernel_wrappers_hide_internal_dispatch_entrypoints") {
  CHECK_FALSE((has_public_process_event<kernel_sm, kernel_dispatch_event>));

  CHECK_FALSE((has_public_process_event<x86_64_sm, x86_64_dispatch_event>));
  CHECK_FALSE((has_public_process_event<aarch64_sm, aarch64_dispatch_event>));
  CHECK_FALSE((has_public_process_event<wasm_sm, wasm_dispatch_event>));
  CHECK_FALSE((has_public_process_event<cuda_sm, cuda_dispatch_event>));
  CHECK_FALSE((has_public_process_event<metal_sm, metal_dispatch_event>));
  CHECK_FALSE((has_public_process_event<vulkan_sm, vulkan_dispatch_event>));
}
